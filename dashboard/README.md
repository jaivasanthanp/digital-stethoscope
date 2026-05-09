# STM32 Inference Dashboard

Local browser dashboard for the STM32U575 synthetic PCG demo.

## Run

```bash
python dashboard/server.py --port COM6 --baud 115200
```

Open:

```text
http://127.0.0.1:8765
```

Use **Start Injection** to send `S` to the STM32. The firmware then renders synthetic
PCG strings, runs DSP + TFLite Micro locally, and logs classifications. Use **Pause**
to send `P`.

The dashboard renders the latest synthetic PCM waveform, the 64-bin log
time-frequency transform computed from that waveform, the exact synthetic script
currently being injected, and a mute/unmute audio control for continuous playback.
The firmware injects 2-second ML windows; the browser repeats the current window
four times per audio buffer so the sound is less choppy while preserving the
actual 2-second STM32 input.

The dashboard parses UART log lines like:

```text
Synthetic input string: SYSTOLIC
Class: Present       Confidence: 96%  Latency: 103ms
```

The synthetic cycle includes `NORMAL`, `SYSTOLIC`, `DIASTOLIC`, `S3`, and
`UNKNOWN`. The deployed ML labels are `Absent`, `Present`, and `Unknown`, so the
old murmur-type script names are demo inputs, not the final classifier labels.

## Dependency

Requires `pyserial`.

```bash
pip install pyserial
```
