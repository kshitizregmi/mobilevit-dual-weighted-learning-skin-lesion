import torch
import numpy as np
from collections import Counter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Tuple, List, Dict
from src.config import settings

def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Defines robust augmentation pipelines."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(settings.IMG_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(settings.IMG_SIZE * 1.14)),
        transforms.CenterCrop(settings.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf

def calculate_effective_sample_weights(class_counts: List[int], beta: float = 0.999) -> torch.Tensor:
    """
    Calculates weights using Effective Number of Samples (CVPR 2019).
    Safety: Uses max(1, count) to prevent division by zero.
    """
    effective_num = 1.0 - np.power(beta, class_counts)
    # Handle the case where effective_num might be 0 (if beta=1 or count=0)
    raw_weights = (1.0 - beta) / (np.array(effective_num) + 1e-6)
    normalized_weights = raw_weights / np.mean(raw_weights)
    return torch.tensor(normalized_weights, dtype=torch.float32)

def create_dataloaders() -> Tuple[Dict[str, DataLoader], List[str], torch.Tensor]:
    train_tf, eval_tf = get_transforms()
    
    train_ds = datasets.ImageFolder(settings.DATA_ROOT / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(settings.DATA_ROOT / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(settings.DATA_ROOT / "test", transform=eval_tf)
    
    classes = train_ds.classes
    targets = [y for _, y in train_ds.samples]
    
    # 1. Loss Weights with safety check
    target_counts = Counter(targets)
    # Fix: Guard against empty classes resulting in 0 count
    count_list = [max(1, target_counts[i]) for i in range(len(classes))]
    loss_weights = calculate_effective_sample_weights(count_list, beta=settings.LOSS_BETA)
    
    # 2. Sampler Weights
    class_probs = {c: (count ** settings.SAMPLER_ALPHA) for c, count in target_counts.items()}
    total_prob = sum(class_probs.values())
    
    normalized_probs = {c: p / total_prob for c, p in class_probs.items()}
    inv_weights = {c: 1.0 / max(1e-12, p) for c, p in normalized_probs.items()}
    sample_weights = torch.DoubleTensor([inv_weights[t] for t in targets])
    
    # 3. Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=int(settings.EPOCH_SIZE_MULT * len(train_ds)), 
        replacement=True
    )

    # 4. Loaders
    loaders = {
        "train": DataLoader(train_ds, settings.BATCH_SIZE, sampler=sampler, 
                          num_workers=settings.NUM_WORKERS, pin_memory=settings.PIN_MEMORY, drop_last=True),
        "val": DataLoader(val_ds, settings.BATCH_SIZE, shuffle=False, 
                        num_workers=settings.NUM_WORKERS, pin_memory=settings.PIN_MEMORY),
        "test": DataLoader(test_ds, settings.BATCH_SIZE, shuffle=False, 
                         num_workers=settings.NUM_WORKERS, pin_memory=settings.PIN_MEMORY)
    }
    
    return loaders, classes, loss_weights