import torch
import torch.nn as nn
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from typing import Tuple
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score

def train_one_epoch(
    model: nn.Module, 
    loader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    epoch_index: int
) -> Tuple[float, float]:
    
    model.train()
    total_loss, correct_predictions, total_samples = 0.0, 0, 0
    
    # Enable scaler only if using CUDA, otherwise it does nothing (no-op for CPU/MPS)
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    with Progress(
        TextColumn(f"[bold blue]Epoch {epoch_index}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("• Loss: {task.fields[loss]:.4f} • Acc: {task.fields[acc]:.2%}"),
        transient=True
    ) as progress:
        
        task = progress.add_task("Training", total=len(loader), loss=0.0, acc=0.0)
        
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Dynamic device type for autocast
            with torch.amp.autocast(device_type=device.type, enabled=True):
                logits = model(images)
                loss = criterion(logits, targets)
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Metrics
            batch_size = targets.size(0)
            batch_acc = (logits.argmax(dim=1) == targets).sum().item() / batch_size
            
            total_loss += loss.item() * batch_size
            correct_predictions += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size
            
            progress.update(task, advance=1, loss=loss.item(), acc=batch_acc)

    return total_loss / total_samples, correct_predictions / total_samples

@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, float, dict]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_targets, all_preds, all_logits = [], [], []
    running_loss = 0.0
    
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        
        with torch.amp.autocast(device_type=device.type, enabled=True):
            logits = model(images)
            loss = criterion(logits, targets)
            
        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_logits.extend(logits.cpu().float().numpy())
        
    y_true, y_pred = np.array(all_targets), np.array(all_preds)
    
    metrics = {
        "weighted_f1": f1_score(y_true, y_pred, average='weighted'),
        "macro_f1": f1_score(y_true, y_pred, average='macro'),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, output_dict=False, zero_division=0)
    }
    
    try:
        if len(all_logits) > 0:
            y_onehot = np.eye(len(all_logits[0]))[y_true]
            metrics["auc"] = roc_auc_score(y_onehot, all_logits, average="weighted", multi_class="ovr")
        else:
            metrics["auc"] = 0.0
    except ValueError:
        metrics["auc"] = 0.0

    return running_loss / len(loader), (y_true == y_pred).mean(), metrics