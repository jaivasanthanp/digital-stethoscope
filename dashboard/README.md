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

The dashboard also renders the latest synthetic PCM waveform, a mel-style
time-frequency view of the model input, and a **Play Heartbeat** button for the
currently injected 2-second signal.

The dashboard parses UART log lines like:

```text
Synthetic input string: SYSTOLIC
Class: SysMurmur     Confidence: 29%  Latency: 102ms
```

## Dependency

Requires `pyserial`.

```bash
pip install pyserial
```
