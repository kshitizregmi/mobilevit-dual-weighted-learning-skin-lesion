import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import numpy as np

def plot_performance(history: List[Dict], save_path: str = "learning_curves.png"):
    epochs = [h["epoch"] for h in history]
    metrics = [
        ("tr_loss", "val_loss", "Loss"),
        ("tr_acc", "val_acc", "Accuracy"),
        ("weighted_f1", None, "Weighted F1")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (train_key, val_key, title) in zip(axes, metrics):
        ax.plot(epochs, [h[train_key] for h in history], label="Train", linewidth=2)
        if val_key:
            ax.plot(epochs, [h[val_key] for h in history], label="Val", linewidth=2)
        else:
            ax.plot(epochs, [h[train_key] for h in history], label="Val (wF1)", color='green', linewidth=2)
            
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_confusion_matrix(cm: np.ndarray, classes: List[str], filename: str):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {filename}')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()