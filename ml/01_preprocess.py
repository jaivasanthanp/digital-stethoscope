"""
01_preprocess.py - CirCor / PhysioNet 2022 preprocessing.

Downloads the CirCor DigiScope PCG dataset used by the George B. Moody
PhysioNet Challenge 2022 and generates 64x64 log-mel spectrograms for
the official murmur labels:

    0 = absent     (no murmur)
    1 = present    (murmur present)
    2 = unknown    (annotator unsure)

Usage:
    py ml/01_preprocess.py --data-dir ml/data_circor
    py ml/01_preprocess.py --data-dir ml/data_circor --skip-download
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


SR_TARGET = 4000
WIN_SEC = 2.0
N_SAMPLES = int(SR_TARGET * WIN_SEC)
N_FFT = 512
HOP_LENGTH = 128
N_MELS = 64
F_MIN = 25
F_MAX = 2000

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

CIRCOR_BASE = "https://physionet.org/files/circor-heart-sound/1.0.3"
CIRCOR_ZIP = (
    "https://www.physionet.org/content/circor-heart-sound/get-zip/1.0.3/"
)

CLASS_NAMES = ["absent", "present", "unknown"]


def download_file(url: str, dest: Path, timeout_sec: int = 60) -> None:
    part_path = dest.with_suffix(dest.suffix + ".part")
    existing = part_path.stat().st_size if part_path.exists() else 0

    headers = {"User-Agent": "digital-stethoscope/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"  Resuming {dest.name} at {existing} bytes")

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        if existing and getattr(response, "status", 200) != 206:
            existing = 0
            part_path.unlink(missing_ok=True)

        content_length = int(response.headers.get("Content-Length", "0") or 0)
        total = existing + content_length if content_length > 0 else 0
        with part_path.open("ab" if existing else "wb") as out:
            with tqdm(
                total=total if total > 0 else None,
                initial=existing,
                unit="B",
                unit_scale=True,
                desc=f"  {dest.name}",
            ) as pbar:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    pbar.update(len(chunk))

    part_path.replace(dest)


def compute_mel_spec(audio: np.ndarray, sr: int = SR_TARGET) -> np.ndarray:
    if sr != SR_TARGET:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR_TARGET)

    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]

    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
        center=False,
    )
    power = np.abs(stft) ** 2
    mel_fb = librosa.filters.mel(
        sr=SR_TARGET,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
    )
    mel_spec = mel_fb @ power
    mel_spec = np.log10(np.maximum(mel_spec, 1e-10))

    if mel_spec.shape[1] < 64:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, 64 - mel_spec.shape[1])))
    else:
        mel_spec = mel_spec[:, :64]

    return mel_spec.astype(np.float32)


def download_circor(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if (raw_dir / "training_data.csv").exists() and (raw_dir / "training_data").exists():
        print("  CirCor raw data already present")
        return

    zip_path = raw_dir / "circor_1.0.3.zip"
    if zip_path.exists() and not zipfile.is_zipfile(zip_path):
        print(f"  Removing corrupt CirCor zip cache: {zip_path}")
        zip_path.unlink()

    if not zip_path.exists():
        print(f"  Downloading CirCor zip -> {zip_path}")
        download_file(CIRCOR_ZIP, zip_path)

    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile(f"{zip_path} is not a valid zip file")

    print("  Extracting CirCor zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    nested = raw_dir / "the-circor-digiscope-phonocardiogram-dataset-1.0.3"
    if nested.exists():
        for child in nested.iterdir():
            dest = raw_dir / child.name
            if dest.exists():
                continue
            if child.is_dir():
                shutil.move(str(child), str(dest))
            else:
                shutil.move(str(child), str(dest))

    if not (raw_dir / "training_data.csv").exists():
        print("  Zip layout did not contain training_data.csv; downloading metadata directly")
        download_file(f"{CIRCOR_BASE}/training_data.csv", raw_dir / "training_data.csv")


def is_present(value: str | None) -> bool:
    if value is None:
        return False
    v = value.strip().lower()
    return bool(v) and v not in {"nan", "none", "not recorded"}


def patient_label(row: dict[str, str]) -> int:
    murmur = row.get("Murmur", "").strip().lower()

    if murmur == "present":
        return 1
    if murmur == "absent":
        return 0
    return 2


def parse_patients(raw_dir: Path) -> dict[str, int]:
    csv_path = raw_dir / "training_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    labels: dict[str, int] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = str(row.get("Patient ID", "")).strip()
            if not patient_id:
                continue
            labels[patient_id] = patient_label(row)

    print(f"  Parsed {len(labels)} patients")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {sum(1 for v in labels.values() if v == i)}")
    return labels


def collect_recordings(raw_dir: Path, labels: dict[str, int]) -> list[tuple[Path, int, str]]:
    training_dir = raw_dir / "training_data"
    if not training_dir.exists():
        training_dir = raw_dir

    recordings: list[tuple[Path, int, str]] = []
    for wav_path in training_dir.rglob("*.wav"):
        patient_id = wav_path.stem.split("_")[0]
        if patient_id in labels:
            recordings.append((wav_path, labels[patient_id], patient_id))

    if not recordings:
        raise FileNotFoundError(f"No CirCor .wav files found under {training_dir}")

    print(f"  Found {len(recordings)} recordings")
    return recordings


def patient_wise_split(recordings: list[tuple[Path, int, str]]):
    patient_to_label: dict[str, int] = {}
    for _, label, patient_id in recordings:
        patient_to_label[patient_id] = label

    by_class = {i: [] for i in range(len(CLASS_NAMES))}
    for patient_id, label in patient_to_label.items():
        by_class[label].append(patient_id)

    rng = random.Random(42)
    split_for_patient = {}
    for label, patient_ids in by_class.items():
        rng.shuffle(patient_ids)
        n_total = len(patient_ids)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        if n_total >= 3:
            n_val = max(1, n_val)
            n_train = min(n_train, n_total - n_val - 1)
        elif n_total == 2:
            n_train = 1
            n_val = 0
        for pid in patient_ids[:n_train]:
            split_for_patient[pid] = "train"
        for pid in patient_ids[n_train:n_train + n_val]:
            split_for_patient[pid] = "val"
        for pid in patient_ids[n_train + n_val:]:
            split_for_patient[pid] = "test"

    splits = {"train": [], "val": [], "test": []}
    for item in recordings:
        splits[split_for_patient[item[2]]].append(item)
    return splits


def windows_from_audio(audio: np.ndarray, max_windows: int) -> list[np.ndarray]:
    if len(audio) <= N_SAMPLES:
        return [audio]

    starts = list(range(0, len(audio) - N_SAMPLES + 1, N_SAMPLES))
    if not starts:
        starts = [0]

    if len(starts) > max_windows:
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in idx]

    return [audio[s:s + N_SAMPLES] for s in starts]


def process_recordings(
    raw_dir: Path,
    processed_dir: Path,
    labels: dict[str, int],
    max_windows: int,
) -> None:
    recordings = collect_recordings(raw_dir, labels)
    splits = patient_wise_split(recordings)

    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    for split_name, split_recs in splits.items():
        print(f"\n  Processing {split_name} ({len(split_recs)} recordings)...")
        counts = {name: 0 for name in CLASS_NAMES}
        for wav_path, label, _patient_id in tqdm(split_recs):
            out_dir = processed_dir / split_name / CLASS_NAMES[label]
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                audio, sr = sf.read(str(wav_path))
                if audio.ndim > 1:
                    audio = audio[:, 0]
                audio = audio.astype(np.float32)
                if np.max(np.abs(audio)) > 0:
                    audio = audio / max(1.0, float(np.max(np.abs(audio))))

                if sr != SR_TARGET:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SR_TARGET)
                    sr = SR_TARGET

                for win_idx, window in enumerate(windows_from_audio(audio, max_windows)):
                    spec = compute_mel_spec(window, sr=sr)
                    out_path = out_dir / f"{wav_path.stem}_w{win_idx:02d}.npy"
                    np.save(out_path, spec)
                    counts[CLASS_NAMES[label]] += 1
            except Exception as exc:
                print(f"    WARNING: failed {wav_path.name}: {exc}")

        print(f"    window counts: {counts}")


def compute_normalization_params(processed_dir: Path) -> tuple[float, float]:
    specs = []
    for npy_file in (processed_dir / "train").rglob("*.npy"):
        specs.append(np.load(npy_file))
    if not specs:
        raise FileNotFoundError(f"No train spectrograms found in {processed_dir}")
    all_data = np.stack(specs, axis=0)
    mean = float(all_data.mean())
    std = float(all_data.std())
    print(f"\n  Training set mean={mean:.4f} std={std:.4f}")
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description="CirCor / PhysioNet 2022 preprocessing")
    parser.add_argument("--data-dir", default="ml/data_circor")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--max-windows", type=int, default=4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    print("=== Step 1: Download CirCor / PhysioNet 2022 ===")
    if args.skip_download:
        print("  Skipping download")
    else:
        download_circor(raw_dir)

    print("\n=== Step 2: Parse labels ===")
    labels = parse_patients(raw_dir)

    print("\n=== Step 3: Compute mel spectrograms ===")
    process_recordings(raw_dir, processed_dir, labels, args.max_windows)

    print("\n=== Step 4: Normalization parameters ===")
    mean, std = compute_normalization_params(processed_dir)
    np.save(data_dir / "normalization_params.npy", np.array([mean, std]))
    print(f"  Saved to {data_dir / 'normalization_params.npy'}")

    print("\n=== Preprocessing complete ===")
    print(f"  Spectrograms: {processed_dir}")
    print("  Next: py ml/03_quantize.py --data-dir ml/data_circor")


if __name__ == "__main__":
    main()
