"""
PhysioNet 2016 Challenge dataloader for 4-class heart sound classification.

Classes:
  0 = Normal
  1 = SystolicMurmur
  2 = DiastolicMurmur
  3 = S3Gallop

Dataset structure expected (after running 01_preprocess.py):
  ml/data/processed/
    train/
      normal/       *.npy  (64×64 float32 spectrograms)
      systolic/     *.npy
      diastolic/    *.npy
      s3gallop/     *.npy
    val/   (same structure)
    test/  (same structure)

Run ml/01_preprocess.py first to generate these files.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

CLASS_NAMES  = ["normal", "systolic", "diastolic", "s3gallop"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


class HeartSoundDataset(Dataset):
    """Loads precomputed 64×64 log-mel spectrogram .npy files."""

    def __init__(self, root_dir: str, split: str = "train", augment: bool = False,
                 norm_mean: float = 0.0, norm_std: float = 1.0):
        """
        root_dir  : path to ml/data/processed/
        split     : 'train', 'val', or 'test'
        augment   : apply TimeShift + SpecAugment + AddNoise (training only)
        norm_mean : dataset-level mean (from normalization_params.npy)
        norm_std  : dataset-level std
        """
        self.augment   = augment and (split == "train")
        self.norm_mean = norm_mean
        self.norm_std  = max(norm_std, 1e-8)
        self.samples   = []   # list of (path, class_idx)

        split_dir = os.path.join(root_dir, split)
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.endswith(".npy"):
                    self.samples.append((os.path.join(class_dir, fname), class_idx))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No .npy files found in {split_dir}. "
                "Run ml/01_preprocess.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        spec = np.load(path).astype(np.float32)  # (64, 64), log-mel

        if self.augment:
            spec = self._time_shift(spec)
            spec = self._spec_augment(spec)
            spec = self._add_noise(spec)

        # Dataset-level normalization (zero-mean unit-var per training stats)
        spec = (spec - self.norm_mean) / self.norm_std

        # Add channel dim → (1, 64, 64)
        spec = torch.from_numpy(spec).unsqueeze(0)
        return spec, label

    @staticmethod
    def _time_shift(spec: np.ndarray) -> np.ndarray:
        """
        TimeShift: random circular shift ±200ms = ±800 samples.
        At 4kHz/128 hop, 800 samples = ~6 frames.
        Circular shift preserves the full spectrogram without zero-padding.
        """
        max_shift = 6   # frames  (800 samples / 128 hop ≈ 6.25)
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(spec, shift, axis=1)

    @staticmethod
    def _spec_augment(spec: np.ndarray) -> np.ndarray:
        """SpecAugment: frequency masking + time masking."""
        spec = spec.copy()
        n_mels, n_frames = spec.shape

        # Frequency masking: mask 2–4 consecutive mel bands
        f_mask  = np.random.randint(2, 5)
        f_start = np.random.randint(0, max(1, n_mels - f_mask))
        spec[f_start:f_start + f_mask, :] = 0.0

        # Time masking: mask 5–10 consecutive frames
        t_mask  = np.random.randint(5, 11)
        t_start = np.random.randint(0, max(1, n_frames - t_mask))
        spec[:, t_start:t_start + t_mask] = 0.0

        return spec

    @staticmethod
    def _add_noise(spec: np.ndarray) -> np.ndarray:
        """AddNoise: Gaussian noise with SNR in [20, 35] dB."""
        snr_db    = np.random.uniform(20, 35)
        signal_pw = np.mean(spec ** 2) + 1e-10
        noise_pw  = signal_pw / (10 ** (snr_db / 10))
        noise     = np.random.randn(*spec.shape).astype(np.float32) * np.sqrt(noise_pw)
        return spec + noise


def get_class_weights(dataset: HeartSoundDataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted cross-entropy loss.
    Handles the Normal class over-representation in PhysioNet 2016.
    """
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for _, label in dataset.samples:
        counts[label] += 1

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(CLASS_NAMES)  # normalize
    return torch.from_numpy(weights)


def make_dataloaders(data_root: str, batch_size: int = 32, num_workers: int = 0,
                     norm_params_path: str = None):
    """Returns (train_loader, val_loader, test_loader, class_weights).

    norm_params_path : path to normalization_params.npy (mean, std).
                       If None, looks for ml/data/normalization_params.npy.
    """
    import os
    if norm_params_path is None:
        # data_root is .../processed/, so params are one level up
        norm_params_path = os.path.join(os.path.dirname(data_root),
                                        "normalization_params.npy")
    if os.path.exists(norm_params_path):
        params = np.load(norm_params_path)
        norm_mean, norm_std = float(params[0]), float(params[1])
        print(f"Normalization: mean={norm_mean:.4f}, std={norm_std:.4f}")
    else:
        norm_mean, norm_std = 0.0, 1.0
        print("WARNING: normalization_params.npy not found — using mean=0, std=1")

    train_ds = HeartSoundDataset(data_root, split="train", augment=True,
                                  norm_mean=norm_mean, norm_std=norm_std)
    val_ds   = HeartSoundDataset(data_root, split="val",   augment=False,
                                  norm_mean=norm_mean, norm_std=norm_std)
    test_ds  = HeartSoundDataset(data_root, split="test",  augment=False,
                                  norm_mean=norm_mean, norm_std=norm_std)

    class_weights = get_class_weights(train_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    print(f"Dataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Class counts (train): {dict(zip(CLASS_NAMES, [sum(1 for _,l in train_ds.samples if l==i) for i in range(4)]))}")
    print(f"Class weights: {class_weights.tolist()}")

    return train_loader, val_loader, test_loader, class_weights
