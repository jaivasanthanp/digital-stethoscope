# Digital Stethoscope — Edge AI Heart Sound Classifier

> Real-time PCG classification on STM32U575 using TFLite Micro + Zephyr RTOS.
> No cloud. No proprietary IDEs. Pure embedded ML at ~102 ms inference per window.

A wearable digital stethoscope prototype. The **laptop dashboard uploads any
WAV / 1D audio NPY** to the STM32U575 over UART (`'A'` command, binary
protocol). The STM32 receives raw 8000-sample 4 kHz PCM, computes the mel
spectrogram on chip, runs a quantized ResNet-10 INT8 CNN to classify **Absent
/ Present / Unknown murmur status** (PhysioNet/CinC 2022 CirCor DigiScope
labels) with a validation-calibrated unknown gate, and returns per-stage
latency, audio statistics, raw 3-class probabilities, and the full 64×64
spectrogram in a single response packet. The dashboard renders a scientific
multi-segment view (probability time-series, segment timeline, per-class
bars, audio RMS, latency breakdown, mel spectrogram heatmap).

Synthetic PCG strings are still rendered on chip as a self-test demo path
(`'S'` / `'P'` UART commands) but are not used by the dashboard. The
ICS-43434 microphone is discontinued for this revision; SAI/I2S microphone
support can be re-introduced later behind the existing audio source API.

## Live demo files

Three single-class CirCor recordings, trimmed to **12 s / 6 STM32 windows**
each, ship under `presentation_samples/` with ground-truth JSON sidecars so
the dashboard auto-fills the accuracy column:

| File | Class | Patient | Note |
|---|---|---|---|
| `01_absent_pid49653.wav` | Absent  | 49653 (Adolescent, AV) | Healthy baseline, quiet S1/S2 only |
| `02_present_pid9979.wav` | Present | 9979 (TV)              | Holosystolic, grade III/VI, diamond shape |
| `03_unknown_pid9983.wav` | Unknown | 9983 (AV)              | Annotator was unsure - should trigger the unknown gate |

There is also a 9-segment mixed demo at
`ml/data_circor/demo/mixed_demo.wav` (3 segments per class) for showing the
timeline view in one upload. Regenerate either set with:

```
py ml\generate_mixed_demo.py            # mixed multi-class demo
py presentation_samples\__build__.py    # single-class samples
```

Built as a final project for the ML course at USST Shanghai (Messtechnik und Sensorik
exchange, Hochschule Coburg)

---

## Hardware

| Component | Role |
|-----------|------|
| STM32U575 NUCLEO-U575ZI-Q | Cortex-M33 @ 160 MHz — audio capture, DSP, CNN inference |
| nRF52840 DK | Zephyr BLE GATT server — broadcasts result to phone |
| Synthetic PCG strings | Current input source, rendered locally to 4 kHz PCM |
| ICS-43434 MEMS microphone | Discontinued in this revision; future optional SAI/I2S input |

---

## System Architecture

```
Laptop dashboard (dashboard/server.py)
    |  HTTP @ 127.0.0.1:8765 - upload .wav / .npy, auto-segment into 2 s windows
    |  serial @ 115200 baud:
    |    'A' + 8000 LE int16 PCM samples (per window, 16 KB)
    v
STM32U575  --  Zephyr RTOS (4 threads + validate thread)
    |
    +-- validate_thread (binary UART command handler)
    |       reads 'A' command, fills s_uploaded_audio[8000]
    |       computes audio stats (RMS, peak, ZCR) on chip
    |
    +-- mel_spec_compute()    (shared, mutex-protected)
    |       Hann window -> arm_rfft_fast_f32 (CMSIS-DSP)
    |       -> 64-band mel filterbank -> log10 -> per-spec z-norm
    |       Output: float32[64x64] spectrogram, ~85 ms / window
    |
    +-- inference_run_probs() (CirCor 2022 INT8 model + unknown gate)
    |       TFLite Micro -- ResNet-10 INT8 (CMSIS-NN kernels)
    |       3 raw probabilities + gate decision, ~102 ms / window
    |
    +-- validate_thread (response packet)
            0xA5 + class + conf + raw probs + gate + dsp_ms + infer_ms
            + rms + peak + zcr + 4096 LE float32 mel values   (16401 B)
                |
                v
Laptop dashboard
    |   - per-segment timeline, probability time-series
    |   - latency breakdown, audio stats, mel heatmap
    |   - optional ground-truth labels JSON -> accuracy column
    v
Browser at http://127.0.0.1:8765
```

Synthetic PCG strings (`'S'`/`'P'` UART commands) feed the legacy
`AudioCaptureThread -> DSPThread -> InferenceThread -> CommThread -> nRF52840`
pipeline; that path still works for self-test and BLE notify, but is not
exercised by the dashboard.

---

## ML Pipeline

### Current Status (2026-05-09)

- Dataset migration from the older PhysioNet 2016 heuristic labels to the CirCor / PhysioNet 2022 DigiScope dataset is complete for the deployed model.
- Current deployed target classes are `Absent`, `Present`, and `Unknown` from the official CirCor `Murmur` label.
- CirCor raw data has been downloaded locally under `ml/data_circor/` and is ignored by git.
- Preprocessing completed successfully with patient-wise splitting and 64x64 log-mel spectrogram generation.
- Generated split counts:

| Split | Absent | Present | Unknown | Total |
|-------|--------|---------|---------|-------|
| Train | 6662 | 1723 | 431 | 8816 |
| Val | 1406 | 343 | 104 | 1853 |
| Test | 1469 | 390 | 88 | 1947 |

- Normalization parameters from the CirCor training split: mean `-3.0214`, std `1.5694`.
- Python dataloader sanity check passes on the new dataset.
- Firmware, dashboard, export, and validation labels have been updated to the deployed three-class mapping.
- Keras ResNet-10 retraining and INT8 post-training quantization completed on CPU.
- Current deployed model results before uncertainty gating: float32 test accuracy `81.2%`, INT8 test accuracy `81.5%`, quantization delta `-0.36%`, INT8 model size `104688` bytes (`102.2 KB`).
- A validation-calibrated Unknown gate is applied after softmax on the STM32. It promotes uncertain Absent/Present outputs to `Unknown` when `unknown_prob >= 0.10` and either `top_prob <= 0.60` or `top2_margin <= 0.41`.
- Held-out test metrics with the gate: accuracy `73.2%`, balanced accuracy `59.6%`, `Unknown` recall `56.8%`. This intentionally trades overall accuracy for safer minority/uncertainty handling.
- Export completed into STM32 source: `app/src/ml/model_data.cc`, `app/src/dsp/normalization_params.h`, and `app/src/ml/test_vectors.h`.
- STM32U575 synthetic firmware build completed successfully in `build_stm32_synth/`.
- Firmware build size with the CirCor model, validation vectors, and Unknown gate: FLASH `561864 B / 2 MB` (`26.79%`), RAM `337100 B / 768 KB` (`42.86%`).
- Flash to STM32U575 completed successfully through ST-LINK/OpenOCD (`build_stm32_synth/zephyr/zephyr.hex`, `561872` bytes written).
- On-device validation over `COM6` passed: STM32U575 matched the Python TFLite reference on `9/9` vectors (`100.0%` reference match).
- Current true-label score on the small generated validation-vector subset is `6/9` (`66.7%`) after the Unknown gate; all three smoke-test Unknown vectors return `Unknown`.
- Live dashboard is running locally at `http://127.0.0.1:8765` with synthetic injection, waveform, mel-style transform, continuous audio mute/unmute, and UART inference logs.
- Handoff for tomorrow: improve Present/Unknown recall with real Unknown audio sampling, stronger augmentation, and possibly a binary `Absent` vs `Present/Unknown` safety gate.

### Model: ResNet-10 with SE blocks

```
Input:      (1, 64, 64)   log-mel spectrogram, zero-mean unit-var

Conv1:      (16, 32, 32)  7x7 conv, stride 2
MaxPool:    (16, 16, 16)  2x2, stride 2

ResBlock1:  (16, 16, 16)  3x3 x2, SE block
ResBlock2:  (32,  8,  8)  3x3 x2, SE block, stride-2 downsample
ResBlock3:  (64,  4,  4)  3x3 x2, SE block, stride-2 downsample

GlobalAvgPool -> FC -> Softmax(3)

Parameters: ~81K float32  ->  ~30K INT8 after quantization
```

### Training

- **Dataset:** CirCor / PhysioNet 2022 DigiScope murmur dataset, patient-wise split
- **Classes:** Absent / Present / Unknown
- **Preprocessing:** resample to 4 kHz, HPF, 2-sec Hann-windowed mel-spectrogram (n_fft=512, hop=128, n_mels=64, f_min=25 Hz, f_max=2 kHz)
- **Augmentation:** TimeShift ±200ms, SpecAugment, Gaussian noise (SNR 20–35 dB)
- **Training:** Keras, Adam, cosine LR decay, class-weighted cross-entropy, 50 epochs

### Quantization

INT8 post-training quantization via TFLite converter with 200-clip representative dataset.

| Metric | Value |
|--------|-------|
| Metric | ResNet-10 (baseline) | ResNet-18 (current deploy, 2026-05-12) |
|---|---|---|
| Parameters | 81 K | **720 K (~9× larger)** |
| INT8 size | 102 KB | **756 KB** |
| Float32 test acc | 81.2 % | 71.3 % |
| INT8 test acc | 81.5 % | 70.9 % |
| INT8 + gate test acc | 73.2 % | n/a (gate threshold unchanged) |
| On-chip inference | 102 ms | **507 ms** |
| Tensor arena | 29 / 40 KB | ~120 / 200 KB |
| FLASH used | 26.3 % | **58.2 %** |
| RAM used | 53.8 % | **74.7 %** |

> **Honest finding (kept in the repo deliberately):** scaling from 81 K to
> 720 K parameters did **not** improve test accuracy on CirCor 2022 — it
> dropped by ~10 percentage points. Validation accuracy went up (63 % → 74 %)
> but test did not follow, the classic symptom of overfitting when model
> capacity exceeds dataset size. Early stopping kicked in at epoch 22.
> Future work: stronger regularization (dropout, mixup, more aggressive
> SpecAugment) or temporal stacking across consecutive 2-second windows
> instead of more spatial capacity.

> **Note on accuracy:** the deployed CirCor model uses the official patient-level
> murmur labels. The post-softmax Unknown gate increases Unknown recall from
> about 19% to 56.8%, at the cost of lower overall accuracy.

---

## Measured Results (on real hardware)

All numbers measured on the physical NUCLEO-U575ZI-Q board via UART.

| Metric | Target | ResNet-10 (initial) | ResNet-18 (current) |
|--------|--------|---------------------|---------------------|
| Inference latency | < 600 ms | 102 ms | **507 ms** |
| Tensor arena allocation | — | 40 KB | **200 KB** |
| FLASH usage | < 2 MB | 561 KB (27 %) | **1.22 MB (58 %)** |
| RAM usage | < 768 KB | 338 KB (44 %) | **587 KB (75 %)** |
| On-device vs Python TFLite | > 95 % match | 9/9 = 100 % | (validated via test vector + M4A round-trip) |

### Boot log (captured from UART on COM6 @ 115200 baud)

```
*** Booting Zephyr OS build v4.4.0-rc1 ***
[00:00:00.000] <inf> main: === Digital Stethoscope v0.1 ===
[00:00:00.000] <inf> main: AudioCaptureThread started
[00:00:00.000] <inf> i2s_capture: Audio: synthetic string source active (ICS-43434 disabled)
[00:00:00.004] <inf> mel_spec: mel_spec: init OK (FFT=512, mels=64, frames=62)
[00:00:00.004] <inf> main: InferenceThread started
[00:00:00.000] <inf> inference: TFLite Micro initialized
[00:00:00.000] <inf> inference:   Model: 104688 bytes
[00:00:00.000] <inf> inference:   Arena used: 29332 / 40960 bytes
[00:00:00.000] <inf> inference:   Input:  [1, 64, 64, 1]  type=9 (INT8)
[00:00:00.000] <inf> inference:   Output: [1, 3]  type=9 (INT8)
[00:00:02.191] <inf> main: Class: Absent        Confidence:  91%  Latency: 102ms
[00:00:04.191] <inf> main: Class: Present       Confidence:  41%  Latency: 102ms
```

### On-device validation (`ml/05_validate_on_device.py`)

```
  #  True             Reference            Device               Match
------------------------------------------------------------------------
  0  Absent           Absent        (91%)  Absent        (91%)  OK
  1  Absent           Absent        (94%)  Absent        (94%)  OK
  2  Absent           Absent        (95%)  Absent        (94%)  OK
  3  Present          Absent        (80%)  Absent        (80%)  OK
  ...
  8  Unknown          Absent        (50%)  Absent        (52%)  OK
------------------------------------------------------------------------
On-device vs Python reference: 9/9 = 100%
```

---

## Repository Structure

```
digital-stethoscope/
|-- CLAUDE.md                        <- AI-assisted dev log (full build notes)
|-- west.yml                         <- Zephyr workspace manifest
|
|-- app/                             <- STM32U575 Zephyr application
|   |-- CMakeLists.txt
|   |-- prj.conf                     <- Kconfig (TFLite Micro, CMSIS-DSP, BLE UART)
|   |-- app.overlay                  <- synthetic build + USART3 (nRF bridge)
|   `-- src/
|       |-- main.c                   <- 4 Zephyr threads + message queues
|       |-- audio/
|       |   `-- i2s_capture.c        <- synthetic PCG string renderer
|       |-- dsp/
|       |   |-- mel_spec.c           <- STFT + mel filterbank + normalization
|       |   |-- mel_filterbank_weights.h  <- precomputed 64x257 float32 weights
|       |   `-- normalization_params.h    <- training set mean/std
|       |-- ml/
|       |   |-- inference.cc         <- TFLite Micro interpreter (C++)
|       |   |-- model_data.cc        <- ResNet-10 INT8 as C array (102 KB)
|       |   |-- test_vectors.h       <- 9 normalized spectrograms for validation
|       |   `-- validate.c           <- UART test harness (05_validate_on_device.py)
|       `-- comms/
|           `-- ble_client.c         <- 6-byte UART packet to nRF52840
|
|-- ble_peripheral/                  <- nRF52840 Zephyr application
|   |-- prj.conf                     <- BT stack config
|   |-- boards/nrf52840dk_nrf52840.overlay
|   `-- src/
|       |-- main.c                   <- UART RX -> BLE GATT notify
|       `-- heart_sound_service.c    <- Custom GATT service (HSC_Result NOTIFY)
|
|-- ml/                              <- Python ML pipeline (runs on laptop)
|   |-- requirements.txt
|   |-- 01_preprocess.py             <- PhysioNet download + mel-spectrogram generation
|   |-- 02_train.py                  <- ResNet-10 PyTorch training
|   |-- 03_quantize.py               <- Keras ResNet-10 + TFLite INT8 PTQ
|   |-- 04_export.py                 <- .tflite -> C arrays + normalization header
|   |-- 05_validate_on_device.py     <- Send test vectors over UART, verify results
|   |-- visualize_spectrograms.py    <- Plot 3-class spectrogram grid
|   |-- generate_mixed_demo.py       <- Build mixed-class WAV from CirCor (NEW)
|   |-- data_circor/                 <- CirCor raw + processed splits (gitignored)
|   |   `-- demo/
|   |       |-- mixed_demo.wav             <- 18 s, 9 segments (3 per class)
|   |       `-- mixed_demo_labels.json
|   `-- models/
|       |-- resnet10.py
|       |-- dataset.py
|       |-- resnet10_int8.tflite
|       |-- unknown_gate.json
|       |-- confusion_matrix.png
|       |-- training_curves.png
|       `-- spectrogram_samples.png
|
|-- dashboard/
|   `-- server.py                    <- Browser dashboard + UART upload client
|
`-- presentation_samples/            <- Live-demo WAVs (one per class, 12 s each)
    |-- 01_absent_pid49653.wav  + _labels.json
    |-- 02_present_pid9979.wav  + _labels.json
    |-- 03_unknown_pid9983.wav  + _labels.json
    |-- __build__.py                 <- Idempotent regenerator
    `-- README.md
```

---

## Build Instructions

### Prerequisites

```bash
# Zephyr workspace at C:/Users/<user>/zephyrproject (v4.4.0)
# Zephyr SDK 1.0.1 at C:/Users/<user>/Desktop/zephyr/zephyr-sdk-1.0.1_...

# Fetch TFLite Micro optional module (one-time)
cd C:/Users/<user>/zephyrproject
west config manifest.group-filter +optional
west update -o=--depth=1 -n tflite-micro
```

### STM32U575 firmware

```bash
export ZEPHYR_BASE="C:/Users/<user>/zephyrproject/zephyr"
export ZEPHYR_SDK_INSTALL_DIR="C:/Users/<user>/Desktop/zephyr/zephyr-sdk-1.0.1_.../zephyr-sdk-1.0.1"
export PATH="/c/ProgramData/chocolatey/bin:$ZEPHYR_SDK_INSTALL_DIR/gnu/arm-zephyr-eabi/bin:$PATH"

cd Digital_Stethoscope
west build --board nucleo_u575zi_q --build-dir build_stm32_synth app \
    -- "-DPython3_EXECUTABLE=C:/Users/<user>/AppData/Local/Programs/Python/Python314/python.exe"

# Flash via OpenOCD (pyocd lacks STM32U5 pack)
west flash --build-dir build_stm32_synth --runner openocd
```

### nRF52840 BLE firmware

```bash
west build --board nrf52840dk/nrf52840 --build-dir build_ble ble_peripheral \
    -- "-DPython3_EXECUTABLE=..."
west flash --build-dir build_ble   # when board arrives
```

### ML pipeline

```bash
pip install -r ml/requirements.txt

python ml/01_preprocess.py          # download CirCor / PhysioNet 2022 + generate spectrograms
python ml/02_train.py               # train ResNet-10 (PyTorch)
python ml/03_quantize.py            # Keras retrain + INT8 PTQ
python ml/04_export.py              # export C arrays to app/src/ml/
python ml/05_validate_on_device.py --port COM6   # on-device validation

# Demo audio (optional - regenerates from CirCor labels)
python ml/generate_mixed_demo.py            # mixed multi-class WAV + labels
python presentation_samples/__build__.py    # 3 single-class demo WAVs + labels
```

### Run the live dashboard

```powershell
C:\Users\<user>\AppData\Local\Programs\Python\Python311\python.exe dashboard\server.py --port COM6 --baud 115200
# Open http://127.0.0.1:8765 in any modern browser.
```

Upload `presentation_samples/02_present_pid9979.wav` for the cleanest live
demo (all 6 windows classify as Present, ground truth comes from the sidecar
JSON, accuracy column reads 100%). Upload `ml/data_circor/demo/mixed_demo.wav`
to demonstrate the full segment timeline and the unknown-gate behaviour on
borderline windows.

Measured on hardware (NUCLEO-U575ZI-Q @ 160 MHz, 115200 baud UART):

| Stage | Per-window time |
|---|---|
| STM32 mel-spectrogram DSP | ~85 ms |
| STM32 INT8 ResNet-10 inference | ~102 ms |
| UART upload (16 KB audio host -> STM32) | ~1.4 s |
| UART return (16 KB spectrogram STM32 -> host) | ~1.4 s |

The UART links dominate total round-trip; on-chip compute is ~190 ms.

---

## BLE GATT Service

**Service UUID:** `12345678-1234-1234-1234-123456789ABC`

**Heart Sound Classification Characteristic** (`...ABD`, NOTIFY, ≤ 20 bytes):

Payload is a short printable ASCII string so any BLE scanner renders it as
readable text in the live notification feed (no manual hex decoding):

| Example payload | Meaning |
|---|---|
| `Absent 91%`  | class 0, 91% confidence |
| `Present 95%` | class 1, 95% confidence |
| `Unknown 14%` | class 2, gate fired |
| `Error 0%`    | inference returned 0xFF |

The characteristic also exposes a **CUD descriptor** (`0x2901`) reading
`"Heart Sound Classification"`, so nRF Connect shows that name instead of
"Unknown Characteristic" in the service browser.

The internal STM32 ↔ nRF UART link still uses the legacy 6-byte binary
frame `[class_id, confidence, reserved, timestamp_ms u24 LE]`; the ASCII
conversion happens inside `hsc_service_notify()` on the nRF, right before
the BLE notify.

### Wiring

| Wire | STM32 NUCLEO-U575ZI-Q | nRF52840 DK |
|---|---|---|
| Data | **PD5** — USART2 TX (silkscreen `USART_B_TX` / D53) | **P0.08** — UART1 RX |
| Ground | any **GND** | any **GND** |

USART3 was originally specced but has no labelled header pin on the
NUCLEO-U575ZI-Q, so the bridge moved to USART2 on 2026-05-12. A common GND
wire is required even when both boards share a PC.

### Demo

Connect with **nRF Connect** (iOS/Android), scan for `HeartSound`
(`CB:1D:2D:72:53:F1` on this build), connect, subscribe to characteristic
`...ABD`, then upload any WAV from the dashboard. Each 2-second classification
window emits one notification on the phone, ~3 s apart.

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Zephyr workspace, 4-thread scaffold, SAI overlay | Done |
| 1 | ML pipeline — preprocess, train, quantize, export (CirCor 2022) | Done |
| 2 | TFLite Micro on STM32U575 — flashed, measured | Done |
| 3 | STFT/Mel DSP chain on U575 — full pipeline running | Done |
| 4 | Synthetic PCG string self-test path | Done |
| 4b | **Laptop audio upload over UART (`'A'` command)** | **Done — measured on hardware** |
| 4c | **Browser dashboard with multi-segment scientific charts** | **Done** |
| 4d | **CirCor demo audio (mixed + presentation samples)** | **Done** |
| 5  | **nRF52840 BLE — flashed, bridge verified, phone notifications confirmed** | **Done (2026-05-12)** |
| 5b | **Dashboard-upload classifications fan out to BLE** | **Done (2026-05-12)** |
| 5c | **Readable ASCII payload + CUD descriptor for live demo** | **Done (2026-05-12)** |
| 6 | Polish, README, demo video | In progress |

---

## Why These Design Choices

**Zephyr over FreeRTOS + CubeIDE:** Single unified build system across both MCUs, deterministic `west build` for regulatory traceability (IEC 62304), no HAL vendor lock-in.

**TFLite Micro over STM32Cube.AI:** Cube.AI links against a proprietary `.a` library that requires CubeIDE project scaffolding — incompatible with Zephyr CMake. TFLite Micro builds cleanly under `west build`. CMSIS-NN backend provides the same Cortex-M33 DSP optimizations. Accepted ~15% latency penalty (102 ms vs ~88 ms estimated for Cube.AI) in exchange for toolchain portability.

**ResNet-10 over VGG:** VGG-16 has 138M parameters — impossible on MCU. ResNet-10 has ~81K parameters with residual connections solving vanishing gradients, enabling deep feature extraction within 786 KB SRAM.

**Mel-spectrogram over raw audio:** 75% of diagnostically relevant cardiac energy lives below 600 Hz. Linear FFT wastes 70% of bins on noise. Mel scale concentrates resolution in the cardiac band. Log power compression equalizes S1 amplitude vs murmur amplitude.

**INT8 quantization:** model size reduction to 102.2 KB, ~4× inference speedup via CMSIS-NN integer MACs, and no observed accuracy loss on the current test split.

---

## Regulatory Framing

IEC 62304 Class B SaMD — advisory output, not autonomous diagnosis. ISO 14971 risk analysis identifies acoustic coupling inconsistency and small dataset as primary hazards. Current prototype has exposed PCB; IEC 60601-1 electrical safety isolation required before patient contact.

---

## References

- [CirCor DigiScope Phonocardiogram Dataset](https://physionet.org/content/circor-heart-sound/1.0.3/) — current PCG murmur dataset
- [Zephyr I2S API](https://docs.zephyrproject.org/latest/hardware/peripherals/i2s.html)
- [TFLite Micro Zephyr sample](https://github.com/zephyrproject-rtos/zephyr/tree/main/samples/modules/tflite-micro)
- [CMSIS-DSP FFT](https://arm-software.github.io/CMSIS-DSP/latest/group__RealFFT.html)
- [NUCLEO-U575ZI-Q board docs](https://docs.zephyrproject.org/latest/boards/st/nucleo_u575zi_q/doc/index.html)
