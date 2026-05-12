"""
generate_mixed_demo.py — Build a long composite heart-sound audio file from
CirCor / PhysioNet 2022 patient recordings.

Picks N patients per Murmur class (Absent / Present / Unknown), takes one
auscultation-location WAV per patient, trims/normalises each to the project's
2-second 4 kHz mono window, concatenates them, and emits:

    ml/data_circor/demo/mixed_demo.wav        16-bit PCM, 4 kHz mono
    ml/data_circor/demo/mixed_demo_labels.json  per-segment ground truth

Each segment is exactly 2 s, so segment K covers samples [K*8000, K*8000+8000)
and time [K*2.0, K*2.0+2.0) seconds. Upload the WAV through the dashboard;
each successive 2-second window will be classified on STM32 against the
labels.json ground truth.

Usage:
    py ml/generate_mixed_demo.py
    py ml/generate_mixed_demo.py --per-class 4 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parent.parent
CIRCOR_DIR = REPO_ROOT / "ml" / "data_circor"
CSV_PATH = CIRCOR_DIR / "raw" / "training_data.csv"
WAV_DIR = CIRCOR_DIR / "raw" / "training_data"
OUT_DIR = CIRCOR_DIR / "demo"

SR_TARGET = 4000
WIN_SEC = 2.0
WIN_SAMPLES = int(SR_TARGET * WIN_SEC)

CLASS_LABELS = ["Absent", "Present", "Unknown"]
CLASS_ID = {"Absent": 0, "Present": 1, "Unknown": 2}
LOCATIONS = ["AV", "MV", "PV", "TV"]


def load_patient_rows():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing {CSV_PATH} — CirCor dataset not present.")
    with CSV_PATH.open() as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader]


def patients_by_class(rows):
    buckets: dict[str, list[dict]] = {label: [] for label in CLASS_LABELS}
    for row in rows:
        label = row.get("Murmur", "").strip()
        if label in buckets:
            buckets[label].append(row)
    return buckets


def first_wav_for_patient(patient_id: str) -> Path | None:
    """Return one auscultation WAV for the patient (prefers the AV/MV/PV/TV order)."""
    for loc in LOCATIONS:
        candidate = WAV_DIR / f"{patient_id}_{loc}.wav"
        if candidate.exists():
            return candidate
    matches = sorted(WAV_DIR.glob(f"{patient_id}_*.wav"))
    return matches[0] if matches else None


def fit_window(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) >= WIN_SAMPLES:
        return audio[:WIN_SAMPLES]
    pad = np.zeros(WIN_SAMPLES - len(audio), dtype=np.float32)
    return np.concatenate([audio, pad])


def load_and_trim(path: Path) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=SR_TARGET, mono=True)
    audio = audio.astype(np.float32)
    # Take the central WIN_SEC for stable cardiac content (skip recording onset/offset).
    if len(audio) > WIN_SAMPLES * 2:
        start = (len(audio) - WIN_SAMPLES) // 2
        audio = audio[start:start + WIN_SAMPLES]
    audio = fit_window(audio)

    # Per-clip peak normalise to 0.7 so all segments sit at a similar level.
    peak = float(np.max(np.abs(audio)) or 1.0)
    if peak > 1e-6:
        audio = audio * (0.7 / peak)
    return audio.astype(np.float32)


def pick_patients(per_class: int, seed: int):
    rng = random.Random(seed)
    rows = load_patient_rows()
    buckets = patients_by_class(rows)

    selections: list[tuple[str, dict, Path]] = []
    for label in CLASS_LABELS:
        candidates = buckets.get(label, [])
        rng.shuffle(candidates)
        kept = 0
        for row in candidates:
            wav = first_wav_for_patient(row["Patient ID"])
            if wav is None:
                continue
            selections.append((label, row, wav))
            kept += 1
            if kept >= per_class:
                break
        if kept < per_class:
            print(f"  WARN: only {kept}/{per_class} usable patients found for class {label}")
    return selections


def build_mixed(selections, out_wav: Path, out_json: Path):
    pieces: list[np.ndarray] = []
    segments: list[dict] = []

    for index, (label, row, wav_path) in enumerate(selections):
        audio = load_and_trim(wav_path)
        pieces.append(audio)
        segments.append({
            "index": index,
            "patient_id": row["Patient ID"],
            "source_file": wav_path.name,
            "class_id": CLASS_ID[label],
            "class_name": label,
            "start_sample": index * WIN_SAMPLES,
            "end_sample": (index + 1) * WIN_SAMPLES,
            "start_seconds": round(index * WIN_SEC, 3),
            "end_seconds": round((index + 1) * WIN_SEC, 3),
        })

    if not pieces:
        raise SystemExit("No usable patients found — aborting.")

    full = np.concatenate(pieces)
    full_int16 = np.clip(full, -1.0, 1.0)
    full_int16 = (full_int16 * 32767.0).astype(np.int16)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), full_int16, SR_TARGET, subtype="PCM_16")

    metadata = {
        "sample_rate": SR_TARGET,
        "window_seconds": WIN_SEC,
        "n_segments": len(segments),
        "class_names": CLASS_LABELS,
        "source": "PhysioNet/CinC 2022 (CirCor DigiScope)",
        "segments": segments,
    }
    out_json.write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=3,
                        help="Number of patients per class (default 3 -> 9 segments / 18 s).")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed used to pick patients.")
    parser.add_argument("--out-wav", type=Path,
                        default=OUT_DIR / "mixed_demo.wav")
    parser.add_argument("--out-json", type=Path,
                        default=OUT_DIR / "mixed_demo_labels.json")
    args = parser.parse_args()

    print("Selecting patients per class:")
    selections = pick_patients(args.per_class, args.seed)
    for label, row, wav in selections:
        print(f"  [{label:7s}] patient {row['Patient ID']:>5s}  -> {wav.name}")

    metadata = build_mixed(selections, args.out_wav, args.out_json)
    duration = metadata["n_segments"] * metadata["window_seconds"]
    print()
    print(f"Wrote {args.out_wav}  ({duration:.1f} s, {metadata['n_segments']} segments)")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
