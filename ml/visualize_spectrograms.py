"""
visualize_spectrograms.py — Plot 4-class mel spectrogram samples

Loads one random spectrogram per class from the processed dataset and
plots them in a 2×2 grid. S1/S2 peaks should appear as bright horizontal
stripes in the low mel bins (~bins 10–25) with regular spacing.

Usage:
    python ml/visualize_spectrograms.py [--data-dir ml/data] [--save]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import random

CLASS_NAMES = ["absent", "present", "unknown"]
CLASS_LABELS = ["Absent", "Present", "Unknown"]


def load_random_sample(class_dir: Path) -> np.ndarray | None:
    files = sorted(class_dir.glob("*.npy"))
    if not files:
        return None
    f = random.choice(files)
    return np.load(f).astype(np.float32)


def plot_spectrograms(data_root: Path, split: str = "train", save: bool = False):
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"Log-Mel Spectrograms — PhysioNet 2016 ({split} set)\n"
                 f"64 mel bins × 64 time frames  |  4 kHz SR, n_fft=512, hop=128",
                 fontsize=12)

    gs = gridspec.GridSpec(1, 3, hspace=0.45, wspace=0.3)

    split_dir = data_root / "processed" / split
    found = 0

    for i, (class_name, class_label) in enumerate(zip(CLASS_NAMES, CLASS_LABELS)):
        class_dir = split_dir / class_name
        spec = load_random_sample(class_dir)
        if spec is None:
            print(f"  No samples found for class: {class_name}")
            continue

        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(spec, aspect='auto', origin='lower',
                       cmap='inferno', interpolation='nearest')
        ax.set_title(f"Class {i}: {class_label}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Time frame (×32 ms)", fontsize=8)
        ax.set_ylabel("Mel bin", fontsize=8)

        # Tick labels
        ax.set_xticks([0, 16, 32, 48, 63])
        ax.set_xticklabels(['0', '0.5s', '1.0s', '1.5s', '2.0s'], fontsize=7)
        ax.set_yticks([0, 16, 32, 48, 63])
        ax.set_yticklabels(['25Hz', '~100Hz', '~300Hz', '~700Hz', '2kHz'], fontsize=7)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='log₁₀ power')
        found += 1

    if found == 0:
        print("No spectrograms found. Run ml/01_preprocess.py first.")
        return

    if save:
        out = Path("ml/models/spectrogram_samples.png")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    else:
        plt.show()


def plot_class_statistics(data_root: Path):
    """Print class counts and distribution across splits."""
    processed = data_root / "processed"
    print("\n=== Dataset Statistics ===")
    print(f"{'Class':<15} {'train':>8} {'val':>8} {'test':>8} {'total':>8}")
    print("-" * 45)

    totals = [0, 0, 0]
    for class_name, class_label in zip(CLASS_NAMES, CLASS_LABELS):
        counts = []
        for split in ["train", "val", "test"]:
            d = processed / split / class_name
            n = len(list(d.glob("*.npy"))) if d.exists() else 0
            counts.append(n)
        total = sum(counts)
        totals = [t + c for t, c in zip(totals, counts)]
        print(f"{class_label:<15} {counts[0]:>8} {counts[1]:>8} {counts[2]:>8} {total:>8}")

    print("-" * 45)
    grand_total = sum(totals)
    print(f"{'TOTAL':<15} {totals[0]:>8} {totals[1]:>8} {totals[2]:>8} {grand_total:>8}")

    # Normalization params
    norm_path = data_root / "normalization_params.npy"
    if norm_path.exists():
        p = np.load(norm_path)
        print(f"\nNormalization: mean={p[0]:.4f}, std={p[1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize mel spectrograms")
    parser.add_argument("--data-dir", default="ml/data_circor")
    parser.add_argument("--split",    default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--save",     action="store_true",
                        help="Save figure instead of displaying")
    parser.add_argument("--stats",    action="store_true",
                        help="Print dataset statistics only")
    args = parser.parse_args()

    data_root = Path(args.data_dir)

    if args.stats:
        plot_class_statistics(data_root)
        return

    plot_class_statistics(data_root)
    plot_spectrograms(data_root, split=args.split, save=args.save)


if __name__ == "__main__":
    main()
