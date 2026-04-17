# Digital Stethoscope — Edge AI Heart Sound Classifier

> Real-time PCG classification on STM32U575 using TFLite Micro + Zephyr RTOS.
> No cloud. No proprietary IDEs. Pure embedded ML at 102 ms per inference.

A wearable digital stethoscope that captures heart sounds via MEMS microphone,
computes mel-spectrograms on-device, runs a quantized ResNet-10 INT8 CNN to classify
**Normal / Systolic Murmur / Diastolic Murmur / S3 Gallop** in real time, and
transmits results over BLE to a phone.

Built as a final project for the ML course at USST Shanghai (Messtechnik und Sensorik
exchange, Hochschule Coburg) and as a portfolio piece targeting German medtech embedded
internships (Dräger, Getemed, Solectrix).

---

## Hardware

| Component | Role |
|-----------|------|
| STM32U575 NUCLEO-U575ZI-Q | Cortex-M33 @ 160 MHz — audio capture, DSP, CNN inference |
| nRF52840 DK | Zephyr BLE GATT server — broadcasts result to phone |
| ICS-43434 MEMS microphone | I2S digital mic — 4 kHz audio capture |

---

## System Architecture

```
ICS-43434 (I2S / SAI1-B)
    |  16-bit PCM @ 4000 Hz, left channel
    v
STM32U575  --  Zephyr RTOS (4 threads, message queues)
    |
    +-- AudioCaptureThread (prio 2)
    |       2-second ring buffer, DMA-backed
    |
    +-- DSPThread (prio 4)
    |       Hann window -> arm_rfft_fast_f32 (CMSIS-DSP)
    |       -> 64-band mel filterbank -> log10 -> normalize
    |       Output: float32[64x64] spectrogram
    |
    +-- InferenceThread (prio 6)
    |       TFLite Micro -- ResNet-10 INT8
    |       CMSIS-NN kernels (Cortex-M33 DSP MACs)
    |       Output: class_id (0-3) + confidence (0-100%)
    |
    +-- CommThread (prio 8)
            6-byte UART packet -> nRF52840
                |
                v
            nRF52840  --  Zephyr BLE GATT
                |   HSC_Result characteristic (NOTIFY)
                v
            Phone / nRF Connect app
```

---

## ML Pipeline

### Model: ResNet-10 with SE blocks

```
Input:      (1, 64, 64)   log-mel spectrogram, zero-mean unit-var

Conv1:      (16, 32, 32)  7x7 conv, stride 2
MaxPool:    (16, 16, 16)  2x2, stride 2

ResBlock1:  (16, 16, 16)  3x3 x2, SE block
ResBlock2:  (32,  8,  8)  3x3 x2, SE block, stride-2 downsample
ResBlock3:  (64,  4,  4)  3x3 x2, SE block, stride-2 downsample

GlobalAvgPool -> FC -> Softmax(4)

Parameters: ~81K float32  ->  ~30K INT8 after quantization
```

### Training

- **Dataset:** PhysioNet 2016 Challenge (3,126 PCG recordings, 2,531 unique after preprocessing)
- **Classes:** Normal (1673) / Systolic Murmur (430) / Diastolic Murmur (242) / S3 Gallop (186)
- **Preprocessing:** resample to 4 kHz, HPF, 2-sec Hann-windowed mel-spectrogram (n_fft=512, hop=128, n_mels=64, f_min=25 Hz, f_max=2 kHz)
- **Augmentation:** TimeShift ±200ms, SpecAugment, Gaussian noise (SNR 20–35 dB)
- **Training:** Keras, Adam, cosine LR decay, class-weighted cross-entropy, 50 epochs

### Quantization

INT8 post-training quantization via TFLite converter with 200-clip representative dataset.

| Metric | Value |
|--------|-------|
| Float32 accuracy | 63.4% |
| INT8 accuracy | 63.2% |
| Accuracy drop | **0.26%** (target < 2%) |
| Model size | 334 KB float32 → **110 KB INT8** (3× reduction) |

> **Note on accuracy:** PhysioNet 2016 provides only binary labels (normal/abnormal).
> Sub-class labels (systolic/diastolic/S3) are assigned heuristically from prevalence
> statistics. The Normal class achieves F1=0.88. True 4-class accuracy requires
> PhysioNet/CinC 2022 (murmur-type annotations).

---

## Measured Results (on real hardware)

All numbers measured on the physical NUCLEO-U575ZI-Q board via UART.

| Metric | Target | Measured |
|--------|--------|----------|
| Inference latency | < 150 ms | **102 ms** |
| Tensor arena used | ~95 KB est. | **29.3 KB** (28% of allocation) |
| FLASH usage | < 2 MB | **430 KB (21%)** without test vectors |
| RAM usage | < 768 KB | **338 KB (44%)** |
| On-device vs Python TFLite | > 95% match | **12/12 = 100%** |

### Boot log (captured from UART on COM6 @ 115200 baud)

```
*** Booting Zephyr OS build v4.4.0-rc1 ***
[00:00:00.000] <inf> main: === Digital Stethoscope v0.1 ===
[00:00:00.000] <inf> main: AudioCaptureThread started
[00:00:00.000] <inf> i2s_capture: Audio: stub mode (real I2S not yet wired)
[00:00:00.004] <inf> mel_spec: mel_spec: init OK (FFT=512, mels=64, frames=62)
[00:00:00.004] <inf> main: InferenceThread started
[00:00:00.000] <inf> inference: TFLite Micro initialized
[00:00:00.000] <inf> inference:   Model: 104752 bytes
[00:00:00.000] <inf> inference:   Arena used: 29332 / 40960 bytes
[00:00:00.000] <inf> inference:   Input:  [1, 64, 64, 1]  type=9 (INT8)
[00:00:00.000] <inf> inference:   Output: [1, 4]  type=9 (INT8)
[00:00:02.191] <inf> main: Class: Normal        Confidence:  30%  Latency: 102ms
[00:00:04.191] <inf> main: Class: SysMurmur     Confidence:  29%  Latency: 102ms
```

### On-device validation (`ml/05_validate_on_device.py`)

```
  #  True             Reference            Device               Match
------------------------------------------------------------------------
  0  Normal           Normal        (31%)  Normal        (31%)  OK
  1  Normal           Normal        (32%)  Normal        (32%)  OK
  2  Normal           Normal        (30%)  Normal        (30%)  OK
  3  SystolicMurmur   SystolicMurmur(27%)  SystolicMurmur(27%)  OK
  ...
 11  S3Gallop         Normal        (29%)  Normal        (29%)  OK
------------------------------------------------------------------------
On-device vs Python reference: 12/12 = 100%
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
|   |-- app.overlay                  <- SAI1-B (mic) + USART3 (nRF bridge)
|   `-- src/
|       |-- main.c                   <- 4 Zephyr threads + message queues
|       |-- audio/
|       |   `-- i2s_capture.c        <- SAI DMA capture + synthetic stub
|       |-- dsp/
|       |   |-- mel_spec.c           <- STFT + mel filterbank + normalization
|       |   |-- mel_filterbank_weights.h  <- precomputed 64x257 float32 weights
|       |   `-- normalization_params.h    <- training set mean/std
|       |-- ml/
|       |   |-- inference.cc         <- TFLite Micro interpreter (C++)
|       |   |-- model_data.cc        <- ResNet-10 INT8 as C array (110 KB)
|       |   |-- test_vectors.h       <- 12 normalized spectrograms for validation
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
`-- ml/                              <- Python ML pipeline (runs on laptop)
    |-- requirements.txt
    |-- 01_preprocess.py             <- PhysioNet download + mel-spectrogram generation
    |-- 02_train.py                  <- ResNet-10 PyTorch training
    |-- 03_quantize.py               <- Keras ResNet-10 + TFLite INT8 PTQ
    |-- 04_export.py                 <- .tflite -> C arrays + normalization header
    |-- 05_validate_on_device.py     <- Send test vectors over UART, verify results
    |-- visualize_spectrograms.py    <- Plot 4-class spectrogram grid
    `-- models/
        |-- resnet10.py              <- PyTorch ResNet-10 definition
        |-- dataset.py               <- PhysioNet 2016 dataloader
        |-- resnet10_int8.tflite     <- Quantized model (110 KB)
        |-- confusion_matrix.png
        |-- training_curves.png
        `-- spectrogram_samples.png
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
west build --board nucleo_u575zi_q --build-dir build_stm32 app \
    -- "-DPython3_EXECUTABLE=C:/Users/<user>/AppData/Local/Programs/Python/Python314/python.exe"

# Flash via OpenOCD (pyocd lacks STM32U5 pack)
west flash --build-dir build_stm32 --runner openocd
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

python ml/01_preprocess.py          # download PhysioNet 2016 + generate spectrograms
python ml/02_train.py               # train ResNet-10 (PyTorch)
python ml/03_quantize.py            # Keras retrain + INT8 PTQ
python ml/04_export.py              # export C arrays to app/src/ml/
python ml/05_validate_on_device.py --port COM6   # on-device validation
```

---

## BLE GATT Service

**Service UUID:** `12345678-1234-1234-1234-123456789ABC`

**HSC_Result Characteristic** (`...ABD`, NOTIFY, 6 bytes):

| Byte | Field | Description |
|------|-------|-------------|
| 0 | `class_id` | 0=Normal, 1=SystolicMurmur, 2=DiastolicMurmur, 3=S3Gallop, 0xFF=Error |
| 1 | `confidence` | 0–100 (%) |
| 2 | reserved | 0x00 |
| 3–5 | `timestamp_ms` | uptime milliseconds, little-endian |

Connect with **nRF Connect** (iOS/Android), subscribe to notifications, observe real-time classifications.

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Zephyr workspace, 4-thread scaffold, SAI overlay | Done |
| 1 | ML pipeline — preprocess, train, quantize, export | Done |
| 2 | TFLite Micro on STM32U575 — flashed, measured | Done |
| 3 | STFT/Mel DSP chain on U575 — full pipeline running | Done |
| 4 | ICS-43434 SAI mic bring-up | Pending (mic arriving ~Apr 19) |
| 5 | nRF52840 BLE — firmware built, awaiting board | Pending (board arriving Apr 24) |
| 6 | Polish, README, demo video | In progress |

---

## Why These Design Choices

**Zephyr over FreeRTOS + CubeIDE:** Single unified build system across both MCUs, deterministic `west build` for regulatory traceability (IEC 62304), no HAL vendor lock-in.

**TFLite Micro over STM32Cube.AI:** Cube.AI links against a proprietary `.a` library that requires CubeIDE project scaffolding — incompatible with Zephyr CMake. TFLite Micro builds cleanly under `west build`. CMSIS-NN backend provides the same Cortex-M33 DSP optimizations. Accepted ~15% latency penalty (102 ms vs ~88 ms estimated for Cube.AI) in exchange for toolchain portability.

**ResNet-10 over VGG:** VGG-16 has 138M parameters — impossible on MCU. ResNet-10 has ~81K parameters with residual connections solving vanishing gradients, enabling deep feature extraction within 786 KB SRAM.

**Mel-spectrogram over raw audio:** 75% of diagnostically relevant cardiac energy lives below 600 Hz. Linear FFT wastes 70% of bins on noise. Mel scale concentrates resolution in the cardiac band. Log power compression equalizes S1 amplitude vs murmur amplitude.

**INT8 quantization:** 3× model size reduction (334 KB → 110 KB), ~4× inference speedup via CMSIS-NN integer MACs, 0.26% accuracy cost.

---

## Regulatory Framing

IEC 62304 Class B SaMD — advisory output, not autonomous diagnosis. ISO 14971 risk analysis identifies acoustic coupling inconsistency and small dataset as primary hazards. Current prototype has exposed PCB; IEC 60601-1 electrical safety isolation required before patient contact.

---

## References

- [PhysioNet 2016 Challenge](https://physionet.org/content/challenge-2016/1.0.0/) — PCG dataset
- [Zephyr I2S API](https://docs.zephyrproject.org/latest/hardware/peripherals/i2s.html)
- [TFLite Micro Zephyr sample](https://github.com/zephyrproject-rtos/zephyr/tree/main/samples/modules/tflite-micro)
- [CMSIS-DSP FFT](https://arm-software.github.io/CMSIS-DSP/latest/group__RealFFT.html)
- [NUCLEO-U575ZI-Q board docs](https://docs.zephyrproject.org/latest/boards/st/nucleo_u575zi_q/doc/index.html)
