import torch
import torch.optim as optim
from rich.console import Console
from src.config import settings
from src import dataset, engine, utils, modeling

console = Console()

def main():
    utils.torch.manual_seed(settings.SEED)
    
    console.print(f"Starting {settings.PROJECT_NAME} on {settings.DEVICE}[/]")
    
    # 1. Data Setup
    loaders, class_names, loss_weights = dataset.create_dataloaders()
    loss_weights = loss_weights.to(settings.DEVICE)
    
    console.print(f"📊 Detected {len(class_names)} classes.")
    console.print(f"⚖️  Computed Loss Weights: {loss_weights.cpu().numpy().round(2)}")

    # 2. Model Setup (Clean Import)
    model = modeling.create_model(num_classes=len(class_names))
    model = model.to(settings.DEVICE)
    
    # Optional: Compile model (Linux/modern GPU only)
    try:
        model = torch.compile(model)
        console.print("[yellow]⚡ Model compiled with torch.compile()[/]")
    except Exception:
        pass

    criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = optim.AdamW(model.parameters(), lr=settings.LEARNING_RATE, weight_decay=settings.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.EPOCHS)

    # 3. Training Loop
    best_wf1 = 0.0
    history = []

    for epoch in range(1, settings.EPOCHS + 1):
        tr_loss, tr_acc = engine.train_one_epoch(
            model, loaders['train'], criterion, optimizer, settings.DEVICE, epoch
        )
        
        val_loss, val_acc, val_metrics = engine.evaluate(model, loaders['val'], settings.DEVICE)
        scheduler.step()
        
        # Tracking
        current_wf1 = val_metrics['weighted_f1']
        history.append({
            "epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc, "weighted_f1": current_wf1
        })
        
        console.print(f"   [dim]Train Loss:[/]{tr_loss:.4f} [dim]Val wF1:[/][bold cyan]{current_wf1:.4f}[/]")

        # Checkpointing
        if current_wf1 > best_wf1:
            best_wf1 = current_wf1
            torch.save(model.state_dict(), settings.CHECKPOINT_DIR / "best_model.pth")
            console.print(f"   [bold green]★ New Best Model Saved![/]")

    # 4. Final Test
    console.rule("[bold red]Final Evaluation[/]")
    # Load best weights
    model.load_state_dict(torch.load(settings.CHECKPOINT_DIR / "best_model.pth", weights_only=True))
    
    test_loss, test_acc, test_metrics = engine.evaluate(model, loaders['test'], settings.DEVICE)
    
    console.print(f"Test wF1: [bold]{test_metrics['weighted_f1']:.4f}[/]")
    console.print(test_metrics['report'])
    
    # Save artifacts
    utils.plot_performance(history)
    utils.save_confusion_matrix(test_metrics['confusion_matrix'], class_names, "test_confusion.png")

if __name__ == "__main__":
    main()