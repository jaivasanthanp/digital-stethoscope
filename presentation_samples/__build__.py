"""
Build the three presentation samples from the CirCor raw training set.

Outputs (alongside this script):
    01_absent_pid49653.wav   + 01_absent_pid49653_labels.json
    02_present_pid9979.wav   + 02_present_pid9979_labels.json
    03_unknown_pid9983.wav   + 03_unknown_pid9983_labels.json

Each WAV is trimmed to exactly 12 s of the central portion of the source
recording, 4 kHz mono 16-bit PCM, so the dashboard cleanly segments it into
six 2-second classification windows on the STM32.
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "ml" / "data_circor" / "raw" / "training_data"

SR = 4000
WIN_SEC = 2.0
TARGET_SEC = 12.0
TARGET_SAMPLES = int(SR * TARGET_SEC)
N_SEGMENTS = int(TARGET_SEC / WIN_SEC)


SOURCES = [
    {
        "out_stem": "01_absent_pid49653",
        "src_name": "49653_AV.wav",
        "class_id": 0,
        "class_name": "Absent",
        "patient_note": "Adolescent, AV recording. No murmur present. Quiet baseline.",
    },
    {
        "out_stem": "02_present_pid9979",
        "src_name": "9979_TV.wav",
        "class_id": 1,
        "class_name": "Present",
        "patient_note": "Holosystolic murmur, grade III/VI, diamond shape, most audible at TV.",
    },
    {
        "out_stem": "03_unknown_pid9983",
        "src_name": "9983_AV.wav",
        "class_id": 2,
        "class_name": "Unknown",
        "patient_note": "Annotator was unsure whether a murmur is present.",
    },
]


def trim_center(audio: np.ndarray) -> np.ndarray:
    if len(audio) >= TARGET_SAMPLES:
        start = (len(audio) - TARGET_SAMPLES) // 2
        return audio[start:start + TARGET_SAMPLES]
    pad = np.zeros(TARGET_SAMPLES - len(audio), dtype=np.float32)
    return np.concatenate([audio, pad])


def peak_normalise(audio: np.ndarray, peak: float = 0.85) -> np.ndarray:
    p = float(np.max(np.abs(audio)) or 1.0)
    return audio * (peak / p) if p > 1e-6 else audio


def write_sample(spec: dict) -> None:
    src = RAW / spec["src_name"]
    if not src.exists():
        raise SystemExit(f"Missing source WAV: {src}")

    audio, _ = librosa.load(str(src), sr=SR, mono=True)
    audio = trim_center(audio.astype(np.float32))
    audio = peak_normalise(audio)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)

    out_wav = HERE / f"{spec['out_stem']}.wav"
    sf.write(str(out_wav), pcm, SR, subtype="PCM_16")

    segments = []
    for i in range(N_SEGMENTS):
        segments.append({
            "index": i,
            "class_id": spec["class_id"],
            "class_name": spec["class_name"],
            "start_seconds": round(i * WIN_SEC, 3),
            "end_seconds": round((i + 1) * WIN_SEC, 3),
        })

    labels = {
        "sample_rate": SR,
        "window_seconds": WIN_SEC,
        "n_segments": N_SEGMENTS,
        "class_names": ["Absent", "Present", "Unknown"],
        "patient_id": spec["src_name"].split("_")[0],
        "source": "PhysioNet/CinC 2022 (CirCor DigiScope)",
        "patient_note": spec["patient_note"],
        "segments": segments,
    }
    out_json = HERE / f"{spec['out_stem']}_labels.json"
    out_json.write_text(json.dumps(labels, indent=2))
    print(f"  wrote {out_wav.name}  +  {out_json.name}")


def main() -> None:
    print("Building presentation samples ->", HERE)
    for s in SOURCES:
        write_sample(s)
    print("Done.")


if __name__ == "__main__":
    main()
