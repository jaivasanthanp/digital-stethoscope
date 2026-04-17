# System Architecture — AI Digital Stethoscope

## Overview

```
ICS-43434 MEMS Mic
    │ I2S (256 kHz BCLK, 4 kHz fs)
    ▼
STM32U575 (Cortex-M33, 160 MHz, 786 KB SRAM)
    │
    ├── AudioCaptureThread (prio 2)
    │   └── 2-sec ring buffer → audio_q
    │
    ├── DSPThread (prio 4)
    │   ├── Hann window + arm_rfft_fast_f32
    │   ├── Mel filterbank (64×257, precomputed)
    │   ├── log10 compression
    │   └── 64×64 float32 → spectrogram_q
    │
    ├── InferenceThread (prio 6)
    │   ├── TFLite Micro INT8 ResNet-10
    │   └── argmax → {class_id, confidence} → result_q
    │
    └── CommThread (prio 8)
        └── UART 115200 → 6-byte packet
                │
                ▼
nRF52840 (Cortex-M4, BLE 5.0)
    ├── UART RX → parse 6-byte packet
    └── BLE GATT NOTIFY → Heart Sound Classification characteristic
                │
                ▼
Phone (BLE Central)
    └── nRF Connect app → real-time classification display
```

## Thread Communication

All inter-thread communication uses Zephyr `k_msgq` (message queues):

| Queue | Item size | Depth | Producer → Consumer |
|---|---|---|---|
| `audio_q` | 32 KB (8000 × float32) | 2 | AudioCapture → DSP |
| `spectrogram_q` | 16 KB (64×64 × float32) | 2 | DSP → Inference |
| `result_q` | 6 bytes | 4 | Inference → Comm |

## Memory Budget (STM32U575, 786 KB SRAM)

| Region | Size |
|---|---|
| Tensor arena | 100 KB |
| Audio buffer (×2 in queue) | 64 KB |
| Spectrogram buffer (×2) | 32 KB |
| Thread stacks | 16 KB |
| DSP workspace | 8 KB |
| **Total** | **~220 KB** |

Remaining ~566 KB available for Zephyr kernel + BLE stack overhead.

## Latency Budget (target: < 2.5 sec per classification)

| Stage | Duration |
|---|---|
| Audio acquisition | 2000 ms (fixed window) |
| DSP (62 frames × STFT + mel) | ~30 ms |
| Inference (ResNet-10 INT8) | ~120 ms |
| UART TX (6 bytes @ 115200) | < 1 ms |
| **Total** | **~2150 ms** |

## Development Phases

See CLAUDE.md §Development Phases for the day-by-day plan.

Current phase: **Phase 0** — workspace setup + stub pipeline.
