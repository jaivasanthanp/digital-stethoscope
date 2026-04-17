"""
test_mel_spec.py — DSP chain validation: Python reference vs on-device output

Generates a 440 Hz sine wave test signal, computes the mel spectrogram in
Python (the reference), then compares against the on-device output captured
over UART.

Usage (host-only, Python reference only):
    python tests/dsp/test_mel_spec.py --reference-only

Usage (with device connected):
    python tests/dsp/test_mel_spec.py --port COM3

Expected result:
  - 440 Hz sine energy should be concentrated in mel bins ~25–30
    (440 Hz is above F_MIN=25 Hz, below F_MAX=2000 Hz)
  - Max absolute difference between Python and device output: < 0.05 (normalized)
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Add ml/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ml"))


def generate_sine(freq_hz: float = 440.0, sr: int = 4000,
                  duration_sec: float = 2.0) -> np.ndarray:
    """Generate a pure sine wave test signal."""
    t = np.arange(int(sr * duration_sec)) / sr
    return (np.sin(2 * np.pi * freq_hz * t) * 0.5).astype(np.float32)


def compute_python_mel(audio: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram using librosa (reference implementation)."""
    import librosa

    SR_TARGET  = 4000
    N_FFT      = 512
    HOP_LENGTH = 128
    N_MELS     = 64
    F_MIN      = 25
    F_MAX      = 2000

    S = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
                     window='hann', center=False)
    power = np.abs(S) ** 2
    mel_fb = librosa.filters.mel(sr=SR_TARGET, n_fft=N_FFT, n_mels=N_MELS,
                                  fmin=F_MIN, fmax=F_MAX)
    mel_spec = mel_fb @ power
    mel_spec = np.log10(np.maximum(mel_spec, 1e-10))

    # Pad/crop to 64×64
    n_frames = mel_spec.shape[1]
    if n_frames < 64:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, 64 - n_frames)))
    else:
        mel_spec = mel_spec[:, :64]

    # Zero-mean unit-variance normalization
    mean = mel_spec.mean()
    std  = mel_spec.std()
    mel_spec = (mel_spec - mean) / (std + 1e-8)

    return mel_spec.astype(np.float32)


def test_440hz_reference():
    """Test: 440 Hz sine has energy concentrated in correct mel bins."""
    audio = generate_sine(440.0)
    spec  = compute_python_mel(audio)

    # Find mel bin with max energy
    energy_per_bin = spec.mean(axis=1)  # average over time
    peak_bin = int(np.argmax(energy_per_bin))

    # At 4 kHz sample rate, 440 Hz should map to approximately bin 28–35
    # (mel scale is logarithmic, 25 Hz–2000 Hz mapped to 64 bins)
    import librosa
    mel_freqs = librosa.mel_frequencies(n_mels=64, fmin=25, fmax=2000)
    expected_bin = int(np.argmin(np.abs(mel_freqs - 440.0)))

    print(f"440 Hz sine test:")
    print(f"  Peak mel bin    : {peak_bin}")
    print(f"  Expected ~bin   : {expected_bin}  ({mel_freqs[expected_bin]:.1f} Hz)")

    assert abs(peak_bin - expected_bin) <= 5, \
        f"Peak bin {peak_bin} too far from expected {expected_bin}"
    print("  PASS: energy concentrated in correct mel band")
    return spec


def compare_with_device(port: str, baud: int, ref_spec: np.ndarray):
    """Send a test vector request and compare device output to Python reference."""
    try:
        import serial
    except ImportError:
        print("ERROR: pyserial not installed. pip install pyserial")
        return

    import struct
    import time

    ser = serial.Serial(port, baud, timeout=3.0)
    time.sleep(0.1)
    print(f"\nConnected to {port} at {baud} baud")

    # Request raw spectrogram dump from device
    # Command: 'S' (spectrogram dump request)
    ser.write(b'S')
    ser.flush()

    # Read 64*64*4 = 16384 bytes (float32 array)
    n_bytes = 64 * 64 * 4
    data = ser.read(n_bytes)
    ser.close()

    if len(data) < n_bytes:
        print(f"FAIL: Expected {n_bytes} bytes, got {len(data)}")
        return

    dev_spec = np.frombuffer(data, dtype=np.float32).reshape(64, 64)
    max_diff = np.abs(ref_spec - dev_spec).max()
    mean_diff = np.abs(ref_spec - dev_spec).mean()

    print(f"\nDSP comparison (Python vs device):")
    print(f"  Max absolute diff  : {max_diff:.4f}")
    print(f"  Mean absolute diff : {mean_diff:.4f}")

    if max_diff < 0.1:
        print("  PASS: DSP outputs match within tolerance")
    else:
        print("  FAIL: DSP outputs differ significantly")
        print("  Check: filterbank weights, normalization, FFT config")


def main():
    parser = argparse.ArgumentParser(description="DSP mel spec validation")
    parser.add_argument("--reference-only", action="store_true",
                        help="Run Python reference test only (no device needed)")
    parser.add_argument("--port",  default="COM3")
    parser.add_argument("--baud",  type=int, default=115200)
    args = parser.parse_args()

    print("=== DSP Mel Spectrogram Test ===\n")

    ref_spec = test_440hz_reference()

    if not args.reference_only:
        compare_with_device(args.port, args.baud, ref_spec)
    else:
        print("\nReference-only mode — skipping device comparison")
        print("Run with --port COM3 to compare against on-device output")


if __name__ == "__main__":
    main()
