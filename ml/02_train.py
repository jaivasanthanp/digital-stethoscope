"""
02_train.py — ResNet-10 training on PhysioNet 2016 mel spectrograms

Usage:
    python ml/02_train.py [--epochs 50] [--batch-size 32] [--lr 1e-3]

Output:
    ml/models/checkpoints/best_model.pth
    ml/models/training_curves.png
    ml/models/confusion_matrix.png

Run ml/01_preprocess.py first.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from pathlib import Path

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.resnet10 import ResNet10
from models.dataset  import make_dataloaders, CLASS_NAMES


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * specs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += specs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        logits = model(specs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * specs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += specs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def plot_curves(train_losses, val_losses, train_accs, val_accs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses,   label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot([a * 100 for a in train_accs], label='Train')
    axes[1].plot([a * 100 for a in val_accs],   label='Val')
    axes[1].set_title('Accuracy (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Test Set)')
    plt.colorbar(im)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=10,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="ResNet-10 training")
    parser.add_argument("--data-dir",   default="ml/data")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--no-se",      action="store_true",
                        help="Disable SE blocks (slightly faster, ~3% lower accuracy)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    data_root  = Path(args.data_dir) / "processed"
    ckpt_dir   = Path("ml/models/checkpoints")
    figure_dir = Path("ml/models")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_weights = \
        make_dataloaders(str(data_root), batch_size=args.batch_size)

    # ── Model ────────────────────────────────────────────────────────────────
    model = ResNet10(n_classes=4, use_se=not args.no_se).to(device)
    print(f"Model params: {model.count_params():,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_acc  = 0.0
    train_losses, val_losses  = [], []
    train_accs,   val_accs    = [], []

    print(f"\nTraining for {args.epochs} epochs (target val acc > 80%)...")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={tr_loss:.4f}/{va_loss:.4f}  "
              f"acc={tr_acc*100:.1f}%/{va_acc*100:.1f}%  "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
            print(f"  *** New best val acc: {best_val_acc*100:.1f}% — checkpoint saved ***")

    # ── Final evaluation on test set ─────────────────────────────────────────
    print("\n=== Test set evaluation ===")
    model.load_state_dict(torch.load(ckpt_dir / "best_model.pth", weights_only=True))
    _, test_acc, y_pred, y_true = eval_epoch(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc*100:.1f}%")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # ── Figures ──────────────────────────────────────────────────────────────
    plot_curves(train_losses, val_losses, train_accs, val_accs,
                figure_dir / "training_curves.png")
    plot_confusion_matrix(y_true, y_pred, figure_dir / "confusion_matrix.png")

    # ── Save full model for export ────────────────────────────────────────────
    torch.save(model, ckpt_dir / "model_full.pth")
    print(f"\nBest val acc: {best_val_acc*100:.1f}%")
    print("Next: python ml/03_quantize.py")


if __name__ == "__main__":
    main()
