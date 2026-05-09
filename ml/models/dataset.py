"""
CirCor / PhysioNet 2022 dataloader for murmur classification.

Classes:
  0 = absent
  1 = present
  2 = unknown
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["absent", "present", "unknown"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


class HeartSoundDataset(Dataset):
    """Loads precomputed 64x64 log-mel spectrogram .npy files."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        augment: bool = False,
        norm_mean: float = 0.0,
        norm_std: float = 1.0,
    ):
        self.augment = augment and split == "train"
        self.norm_mean = norm_mean
        self.norm_std = max(norm_std, 1e-8)
        self.samples = []

        split_dir = os.path.join(root_dir, split)
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.endswith(".npy"):
                    self.samples.append((os.path.join(class_dir, fname), class_idx))

        if not self.samples:
            raise FileNotFoundError(
                f"No .npy files found in {split_dir}. Run ml/01_preprocess.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        spec = np.load(path).astype(np.float32)

        if self.augment:
            spec = self._time_shift(spec)
            spec = self._spec_augment(spec)
            spec = self._add_noise(spec)

        spec = (spec - self.norm_mean) / self.norm_std
        spec = torch.from_numpy(spec).unsqueeze(0)
        return spec, label

    @staticmethod
    def _time_shift(spec: np.ndarray) -> np.ndarray:
        shift = np.random.randint(-6, 7)
        return np.roll(spec, shift, axis=1)

    @staticmethod
    def _spec_augment(spec: np.ndarray) -> np.ndarray:
        spec = spec.copy()
        n_mels, n_frames = spec.shape

        f_mask = np.random.randint(2, 5)
        f_start = np.random.randint(0, max(1, n_mels - f_mask))
        spec[f_start:f_start + f_mask, :] = 0.0

        t_mask = np.random.randint(5, 11)
        t_start = np.random.randint(0, max(1, n_frames - t_mask))
        spec[:, t_start:t_start + t_mask] = 0.0
        return spec

    @staticmethod
    def _add_noise(spec: np.ndarray) -> np.ndarray:
        snr_db = np.random.uniform(20, 35)
        signal_pw = np.mean(spec ** 2) + 1e-10
        noise_pw = signal_pw / (10 ** (snr_db / 10))
        noise = np.random.randn(*spec.shape).astype(np.float32) * np.sqrt(noise_pw)
        return spec + noise


def get_class_weights(dataset: HeartSoundDataset) -> torch.Tensor:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for _, label in dataset.samples:
        counts[label] += 1

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(CLASS_NAMES)
    return torch.from_numpy(weights)


def make_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 0,
    norm_params_path: str | None = None,
):
    if norm_params_path is None:
        norm_params_path = os.path.join(os.path.dirname(data_root), "normalization_params.npy")

    if os.path.exists(norm_params_path):
        params = np.load(norm_params_path)
        norm_mean, norm_std = float(params[0]), float(params[1])
        print(f"Normalization: mean={norm_mean:.4f}, std={norm_std:.4f}")
    else:
        norm_mean, norm_std = 0.0, 1.0
        print("WARNING: normalization_params.npy not found; using mean=0, std=1")

    train_ds = HeartSoundDataset(data_root, "train", True, norm_mean, norm_std)
    val_ds = HeartSoundDataset(data_root, "val", False, norm_mean, norm_std)
    test_ds = HeartSoundDataset(data_root, "test", False, norm_mean, norm_std)

    class_weights = get_class_weights(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    counts = [sum(1 for _, label in train_ds.samples if label == i) for i in range(len(CLASS_NAMES))]
    print(f"Dataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Class counts (train): {dict(zip(CLASS_NAMES, counts))}")
    print(f"Class weights: {class_weights.tolist()}")

    return train_loader, val_loader, test_loader, class_weights
