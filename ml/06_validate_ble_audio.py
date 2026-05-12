"""
06_validate_ble_audio.py - Validate the phone -> BLE -> nRF -> STM32 path.

Reads any ffmpeg-supported audio file, resamples to 4 kHz mono int16 PCM,
chunks it into 2-second (16 KB) windows, streams each window to the nRF52840
DK's AudioIn characteristic via BLE WRITE_WITHOUT_RESPONSE, and prints the
classification notifications that come back on the HSC_Result characteristic.

Usage:
    pip install bleak imageio-ffmpeg soundfile numpy
    python ml/06_validate_ble_audio.py path/to/heartbeat.m4a
    python ml/06_validate_ble_audio.py path/to/heartbeat.wav --device HeartSound
"""

from __future__ import annotations

import argparse
import asyncio
import io
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sys.exit("Install soundfile: pip install soundfile")

try:
    from bleak import BleakScanner, BleakClient
    from bleak.exc import BleakError
except ImportError:
    sys.exit("Install bleak: pip install bleak")

# UUIDs - must match ble_peripheral/src/audio_input_service.c +
# heart_sound_service.c.
SERVICE_UUID      = "12345678-1234-1234-1234-123456789abc"
RESULT_CHAR_UUID  = "12345678-1234-1234-1234-123456789abd"
AUDIO_IN_UUID     = "12345678-1234-1234-1234-123456789abe"

WINDOW_SAMPLES = 8000           # 2 s @ 4 kHz
WINDOW_BYTES   = WINDOW_SAMPLES * 2  # int16 LE -> 16000 B


def find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            sys.exit("ffmpeg not found - pip install imageio-ffmpeg")
        return ffmpeg


def decode_audio(path: Path, sample_rate: int = 4000) -> np.ndarray:
    """Decode any ffmpeg-supported file to mono int16 PCM at sample_rate."""
    ffmpeg = find_ffmpeg()
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path),
         "-f", "wav", "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", str(sample_rate), "-"],
        capture_output=True, check=False,
    )
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[:200]}")
    pcm, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    # Peak-normalise if quiet (matches dashboard auto_boost).
    peak = float(np.max(np.abs(pcm)))
    if 0.0 < peak < 0.5:
        pcm = pcm * (0.95 / peak)
        print(f"  auto-boost: peak {peak:.3f} -> 0.95")
    int16 = np.clip(pcm * 32767.0, -32768, 32767).astype("<i2")
    print(f"  decoded: sr={sr} samples={len(int16)} duration={len(int16)/sr:.2f}s")
    return int16


CLASS_NAMES = ["Absent", "Present", "Unknown"]


async def stream_audio(path: Path, device_name: str, chunk: int):
    print(f"Decoding {path.name} -> 4 kHz mono int16 PCM...")
    audio = decode_audio(path)
    n_windows = len(audio) // WINDOW_SAMPLES
    if n_windows == 0:
        sys.exit("Audio shorter than 2 seconds, nothing to send.")
    print(f"  {n_windows} 2-second window(s) to stream\n")

    print(f"Scanning for BLE device '{device_name}'...")
    target = None
    devices = await BleakScanner.discover(timeout=8.0)
    for d in devices:
        if d.name == device_name:
            target = d
            print(f"  found at {d.address}")
            break
    if target is None:
        sys.exit(f"Device '{device_name}' not found. Make sure nRF is "
                 "advertising and not already connected to another central.")

    notifications: list[tuple[int, str, int]] = []
    notif_event = asyncio.Event()

    def on_notify(_char, data: bytearray):
        # The nRF currently re-uses the legacy ASCII payload from
        # hsc_service_notify("Absent 95%" etc), so we just decode as text.
        try:
            text = bytes(data).decode("ascii", errors="replace").strip()
        except Exception:
            text = repr(bytes(data))
        notifications.append((len(notifications), text, len(data)))
        print(f"  [BLE NOTIFY #{len(notifications)}] {text}  ({len(data)} bytes)")
        notif_event.set()

    async with BleakClient(target.address, timeout=15.0) as client:
        print("Connected.")
        # Negotiate the largest MTU the central supports. On Windows this is
        # automatic; we just request and bleak handles the rest.
        try:
            await client._backend._acquire_mtu()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            mtu = client.mtu_size
            print(f"  ATT MTU: {mtu} (max chunk = {mtu - 3} bytes)")
        except Exception:
            mtu = 23
            print("  ATT MTU: unknown (assuming 23)")

        # Cap chunk by what BLE actually allows.
        chunk = min(chunk, max(20, mtu - 3))
        print(f"  using {chunk}-byte writes\n")

        await client.start_notify(RESULT_CHAR_UUID, on_notify)
        print(f"Subscribed to {RESULT_CHAR_UUID}\n")

        for w in range(n_windows):
            start = w * WINDOW_SAMPLES
            window_bytes = audio[start:start + WINDOW_SAMPLES].tobytes()
            assert len(window_bytes) == WINDOW_BYTES
            print(f"Window {w}: streaming {WINDOW_BYTES} B in {chunk}-byte chunks...")

            sent = 0
            for i in range(0, WINDOW_BYTES, chunk):
                piece = window_bytes[i:i + chunk]
                await client.write_gatt_char(AUDIO_IN_UUID, piece, response=False)
                sent += len(piece)
            print(f"  wrote {sent} B, waiting for notification...")

            # Wait for the next notification or a 10-second timeout.
            try:
                await asyncio.wait_for(notif_event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                print(f"  ! timeout waiting for window {w} result")
            notif_event.clear()

        await client.stop_notify(RESULT_CHAR_UUID)
        print(f"\nReceived {len(notifications)} notification(s) for "
              f"{n_windows} window(s).")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("audio", type=Path, help="input audio file (any ffmpeg format)")
    ap.add_argument("--device", default="HeartSound",
                    help="BLE device name to scan for (default: HeartSound)")
    ap.add_argument("--chunk", type=int, default=200,
                    help="WRITE chunk size in bytes (auto-capped by MTU)")
    args = ap.parse_args()
    if not args.audio.exists():
        sys.exit(f"File not found: {args.audio}")

    try:
        asyncio.run(stream_audio(args.audio, args.device, args.chunk))
    except BleakError as e:
        sys.exit(f"BLE error: {e}")


if __name__ == "__main__":
    main()
