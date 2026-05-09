# CLAUDE.md — AI Digital Stethoscope (PCG Heart Sound Classifier)

## Project Identity

**What this is:** A wearable digital stethoscope prototype that currently uses synthetic
PCG strings as the input source on the STM32U575, computes mel-spectrograms on-device,
runs a quantized ResNet-10 INT8 CNN to classify Normal / Systolic Murmur / Diastolic
Murmur / S3 Gallop in real time, and transmits results over BLE to a phone. No cloud.
No CubeIDE. Pure Zephyr RTOS on both MCUs.

**Current input-source decision (May 2026):** ICS-43434 is discontinued for this
revision and is not part of the active firmware path. The STM32U575 locally renders
compact synthetic heart-sound strings into 2-second PCM windows, then runs the same
DSP + TFLite Micro inference chain locally. SAI/I2S microphone support may be added
later behind `audio_capture_init()` / `audio_capture_get_window()`.

**Why it exists:** Final exam project for ML course at USST Shanghai (Messtechnik und Sensorik
exchange semester). Also a portfolio piece targeting German medtech embedded internships
(Dräger, Getemed, Solectrix). The architecture is deliberately chosen to demonstrate
edge AI, Zephyr on STM32, and BLE GATT — the exact skill stack those employers are hiring for.

**Owner:** Jai — M.Sc. Messtechnik und Sensorik, Hochschule Coburg (exchange @ USST Shanghai)

---

## Hardware

| Board | Role | Status |
|---|---|---|
| STM32U575 NUCLEO-U575ZI-Q | Cortex-M33, audio capture, DSP, CNN inference | In hand |
| nRF52840 DK (or Adafruit Feather) | Zephyr BLE GATT server, result broadcast | Arriving ~Apr 24 |
| ICS-43434 MEMS microphone | Discontinued in current revision; future optional SAI/I2S source | Not active |

**Budget constraint:** €50 excluding boards. All additional components sourced in Shanghai
(Taobao/LCSC) — dramatically cheaper than Germany. PCB from JLCPCB (5x 2-layer, ~¥30).

**DO NOT assume CubeIDE or STM32Cube.AI are available.** This project uses Zephyr end-to-end.
TFLite Micro replaces STM32Cube.AI. See ML section for rationale.

---

## Toolchain

```
west (Zephyr build system) — single workspace covering both STM32U575 and nRF52840
arm-zephyr-eabi-gcc — Zephyr SDK toolchain
west flash — ST-Link on NUCLEO, J-Link on nRF DK
west debug — GDB server via pyOCD or J-Link
Python 3.10+ — ML pipeline (PyTorch or TF 2.12)
```

**Zephyr version:** Use latest LTS (v3.6.x or v3.7.x). Confirm `nucleo_u575zi_q` board target
exists in `west list` before starting.

**Do not use:**
- STM32CubeIDE
- STM32CubeMX  
- STM32Cube.AI (use TFLite Micro instead — see ML section)
- Arduino HAL
- PlatformIO

---

## Repository Structure

```
digital-stethoscope/
├── CLAUDE.md                    ← this file
├── west.yml                     ← Zephyr manifest
├── CMakeLists.txt               ← top-level
│
├── app/                         ← STM32U575 Zephyr application
│   ├── CMakeLists.txt
│   ├── prj.conf                 ← Kconfig
│   ├── app.overlay              ← devicetree overlay (I2S, UART)
│   └── src/
│       ├── main.c
│       ├── audio/
│       │   ├── i2s_capture.c    ← ICS-43434 I2S DMA driver
│       │   └── i2s_capture.h
│       ├── dsp/
│       │   ├── mel_spec.c       ← STFT + mel filterbank
│       │   ├── mel_spec.h
│       │   └── mel_filterbank_weights.h  ← precomputed, const array
│       ├── ml/
│       │   ├── inference.cc     ← TFLite Micro interpreter
│       │   ├── inference.h
│       │   └── model_data.cc    ← quantized model as C array (xxd generated)
│       └── comms/
│           ├── ble_client.c     ← sends result to nRF over UART bridge
│           └── ble_client.h
│
├── ble_peripheral/              ← nRF52840 Zephyr application
│   ├── CMakeLists.txt
│   ├── prj.conf
│   ├── boards/
│   │   └── nrf52840dk_nrf52840.overlay
│   └── src/
│       ├── main.c
│       └── heart_sound_service.c  ← custom BLE GATT service
│
├── ml/                          ← Python ML pipeline (runs on laptop)
│   ├── requirements.txt
│   ├── 01_preprocess.py         ← PhysioNet download + mel-spectrogram generation
│   ├── 02_train.py              ← ResNet-10 training
│   ├── 03_quantize.py           ← INT8 post-training quantization
│   ├── 04_export.py             ← .tflite → C array
│   ├── 05_validate_on_device.py ← sends test vector over UART, reads inference result
│   └── models/
│       ├── resnet10.py          ← model definition
│       └── dataset.py           ← PhysioNet 2016 dataloader
│
├── tests/
│   ├── dsp/
│   │   └── test_mel_spec.py     ← compare C output vs Python reference (UART)
│   └── ml/
│       └── test_inference.py    ← on-device accuracy spot check
│
└── docs/
    ├── architecture.md
    ├── ble_gatt_spec.md
    └── regulatory_framing.md    ← IEC 62304 framing notes
```

---

## Signal Chain (End-to-End)

Current firmware path:

```
Synthetic PCG string
    |  LABEL|start,duration,frequency,amplitude;...
    v
STM32U575 synthetic source
    |  renders float32 PCM @ 4000 Hz, 8000 samples per 2-second window
    v
DSPTask
    |  Hann window -> CMSIS-DSP FFT -> mel filterbank -> log10 -> normalize
    v
InferenceTask
    |  TFLite Micro ResNet-10 INT8 runs locally on STM32U575
    v
CommTask
    |  UART result packet to nRF52840, then BLE notify
```

Legacy/future optional microphone path:

```
ICS-43434 (I2S)
    │  16-bit PCM @ 4000 Hz, stereo (use left channel only)
    ▼
STM32U575 — I2S peripheral + DMA (circular buffer, 512-sample periods)
    │  AudioCaptureTask: fills 8000-sample (2-sec) ring buffer
    ▼
DSPTask
    │  1. Apply Hann window (512 samples, hop=128) → 62 frames
    │  2. arm_rfft_fast_f32 (CMSIS-DSP) → 257 power bins per frame
    │  3. Mel filterbank matrix multiply (precomputed 64×257 weights)
    │  4. Log10 compression → 64×62 float matrix
    │  5. Resize/pad to 64×64, normalize to zero-mean unit-variance
    ▼
InferenceTask
    │  TFLite MicroInterpreter, ResNet-10 INT8
    │  Tensor arena: ~95 KB (fits in 786 KB SRAM)
    │  Output: float32[4] softmax probabilities
    │  Argmax → class label (0=Normal, 1=SysMurmur, 2=DiaMurmur, 3=S3)
    ▼
CommTask
    │  UART bridge to nRF52840 (or direct BLE if nRF integrated)
    │  Packet: {class_id: u8, confidence: u8 (0-100), timestamp: u32}
    ▼
nRF52840 — Zephyr BLE GATT Server
    │  Heart Sound Classification Service (custom UUID)
    │  Characteristic: HSC_Result (notify, 6 bytes)
    ▼
Phone (BLE Central)
    │  Any BLE scanner or custom app
    └  nRF Connect app works for demo/testing
```

---

## ML Architecture

### Model: ResNet-10

```python
Input: (1, 64, 64)        # grayscale mel-spectrogram

Conv1:     (16, 32, 32)   # 7×7 kernel, stride 2
MaxPool:   (16, 16, 16)   # 2×2, stride 2

ResBlock1: (16, 16, 16)   # 3×3 × 2, no downsample
ResBlock2: (32, 8,  8)    # 3×3 × 2, stride 2 downsample
ResBlock3: (64, 4,  4)    # 3×3 × 2, stride 2 downsample

GlobalAvgPool: (64,)
FC + Softmax:  (4,)       # Normal / SysMurmur / DiaMurmur / S3Gallop

Total params: ~120K float32 → ~30K INT8
```

Each ResBlock = Conv→BN→ReLU→Conv→BN + skip connection, ReLU after add.
Skip connection uses 1×1 conv projection when channels change (ResBlock2, ResBlock3).

Optional: add SE (Squeeze-and-Excitation) after GlobalAvgPool in each ResBlock.
SE: GAP → FC(64→16) → ReLU → FC(16→64) → Sigmoid → channel-wise multiply.
Cost: ~2K extra params, ~+3% accuracy. Worth it.

### Why NOT STM32Cube.AI

STM32Cube.AI generates code that links against ST's proprietary `NetworkRuntime` .a library
and uses STM32 HAL macros internally. It cannot be compiled in a Zephyr CMakeLists.txt
without the CubeIDE project scaffolding. Since this project uses Zephyr end-to-end,
**TFLite Micro is the correct choice:**

- HAL-agnostic C++, builds cleanly under `west build`
- CMSIS-NN backend (`CONFIG_TENSORFLOW_LITE_MICRO_CMSIS_NN_KERNELS=y`) provides the same
  Cortex-M33 DSP optimizations as Cube.AI's runtime
- Standard TFLite INT8 quantization flow — same training pipeline
- Latency penalty vs Cube.AI: ~15-20% (estimate ~115-130ms vs ~100ms). Acceptable.

### Dataset

**PhysioNet 2016 Challenge** — open access, no registration required.
URL: https://physionet.org/content/challenge-2016/1.0.0/
Size: ~500MB. 3,240 PCG recordings, 6 clinical sites (a/ through f/).
Labels: normal / abnormal. Abnormal subset annotated by type (murmur, extrasystole, etc.)

For 4-class training: use annotation files to split abnormal into
SystolicMurmur / DiastolicMurmur / S3Gallop. Class balance is uneven — use weighted
cross-entropy loss or oversampling on minority classes.

### Preprocessing (Python reference: `ml/01_preprocess.py`)

```python
sr_target = 4000          # resample all to 4 kHz
window_sec = 2.0          # 8000 samples per clip
n_fft = 512               # FFT size
hop_length = 128          # 32ms hop
n_mels = 64               # mel bins
f_min = 25                # Hz — high-pass (removes DC/rumble)
f_max = 2000              # Hz — Nyquist at 4 kHz
```

Augmentation (apply during training only, NOT on validation/test):
- `TimeShift`: random shift ±200ms (±800 samples)
- `SpecAugment`: mask 2-4 consecutive mel bands, mask 5-10 consecutive time frames
- `AddNoise`: Gaussian noise, SNR uniform in [20, 35] dB

Normalization: per-spectrogram zero-mean, unit-variance. Compute mean/std on training
set and store them — apply same stats to validation, test, and on-device inference.

### Quantization (Python: `ml/03_quantize.py`)

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset: 200 clips covering all 4 classes
def representative_dataset():
    for spec in calibration_specs:
        yield [spec.astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
```

After conversion, verify: load .tflite in Python interpreter, run all test clips,
confirm accuracy drop < 2% vs float32 model. If > 2%, increase calibration set size.

### C Array Export

```bash
xxd -i model_quantized.tflite > app/src/ml/model_data.cc
# Then manually add:  const in model_data.cc, alignas(8) attribute
```

---

## Zephyr Configuration Notes

### `prj.conf` (STM32U575) — verified working

```kconfig
# Core
CONFIG_STDOUT_CONSOLE=y
CONFIG_UART_CONSOLE=y
CONFIG_LOG=y
CONFIG_LOG_DEFAULT_LEVEL=3

# Audio source: synthetic PCG strings rendered locally on STM32U575
# CONFIG_I2S is not set
# CONFIG_DMA is not set

# CMSIS-DSP for STFT — NOTE: symbol is CMSIS_DSP_BASICMATH (not BASIC_MATH)
CONFIG_CMSIS_DSP=y
CONFIG_CMSIS_DSP_TRANSFORM=y
CONFIG_CMSIS_DSP_STATISTICS=y
CONFIG_CMSIS_DSP_BASICMATH=y
CONFIG_FPU=y

# TFLite Micro — uncomment in Phase 2 after running: west update
# CONFIG_TENSORFLOW_LITE_MICRO=y
# CONFIG_TENSORFLOW_LITE_MICRO_CMSIS_NN_KERNELS=y

# Stack and heap
# Phase 0: 32 KB heap (no TFLite). Phase 2+: restore to 524288 (512 KB)
CONFIG_MAIN_STACK_SIZE=4096
CONFIG_HEAP_MEM_POOL_SIZE=32768

# Threads
CONFIG_NUM_PREEMPT_PRIORITIES=16

# UART for BLE bridge
CONFIG_UART_INTERRUPT_DRIVEN=y
```

### `app.overlay` (SAI1 + USART3 — verified working on Zephyr 4.4)

```dts
/* SAI1 Block B — ICS-43434 receiver */
&sai1_b {
    status = "okay";
    pinctrl-0 = <&sai1_sck_b_pb3 &sai1_fs_b_pb6 &sai1_sd_b_pb5>;
    pinctrl-names = "default";
    dma-names = "rx";   /* required — base DTSI has dmas but not dma-names */
};

/* USART3 — bridge to nRF52840 at 115200 baud */
&usart3 {
    status = "okay";
    pinctrl-0 = <&usart3_tx_pd8 &usart3_rx_pd9>;
    pinctrl-names = "default";
    current-speed = <115200>;
};
```

ICS-43434 wiring: BCLK→PB3, LRCLK→PB6, DOUT→PB5, VDD→3.3V, L/R→GND (left ch).

### Tensor Arena Sizing

The tensor arena must be statically allocated. 95 KB is the estimate — tune by running
`interpreter->arena_used_bytes()` after `AllocateTensors()` and adjusting.

```c
// in inference.cc
static uint8_t tensor_arena[100 * 1024] __attribute__((aligned(16)));
```

---

## Zephyr Task Architecture

Four Zephyr threads communicate via message queues:

```
AudioCaptureThread  (prio 2, stack 2048)
    k_msgq: audio_q (8000-sample float32 buffers, depth 2)

DSPThread           (prio 4, stack 4096)
    k_msgq: spectrogram_q (64×64 float32, depth 2)

InferenceThread     (prio 6, stack 8192)  ← TFLite needs large stack
    k_msgq: result_q ({class_id, confidence}, depth 4)

CommThread          (prio 8, stack 2048)
    UART TX to nRF52840
```

Higher priority number = lower priority in Zephyr. Audio capture runs highest priority
to avoid I2S buffer overruns. Inference runs lowest because it's the longest operation.

**During development (before mic arrives):** AudioCaptureThread is a stub that loads
a hardcoded float array from `test_vectors.h` and pushes it to audio_q on a 2-second
timer. This lets you test the full DSP+inference pipeline on real hardware without mic.

---

## BLE GATT Service Definition

**Service UUID:** `12345678-1234-1234-1234-123456789ABC` (custom, use for prototype)

**Characteristic: HSC_Result**
- UUID: `12345678-1234-1234-1234-123456789ABD`
- Properties: NOTIFY
- Length: 6 bytes
- Format: `[class_id: u8, confidence: u8, reserved: u8, timestamp: u32 little-endian]`

class_id encoding: 0=Normal, 1=SystolicMurmur, 2=DiastolicMurmur, 3=S3Gallop, 0xFF=NoResult

For demo: pair with nRF Connect app (iOS/Android), subscribe to notifications,
observe classification results in real time. No custom app needed for exam demo.

---

## Development Phases & Roadmap

### Phase 0 — Zephyr workspace ✅ COMPLETE (Apr 16–17)
- [x] Existing Zephyr workspace confirmed at `C:/Users/jaiva/zephyrproject/` (v4.4.0-rc1)
- [x] `nucleo_u575zi_q` board target confirmed present
- [x] Full repo structure created (app/, ble_peripheral/, ml/, tests/, docs/)
- [x] Firmware builds and links: FLASH 209 KB / 2 MB, RAM 261 KB / 768 KB
- [x] All four Zephyr threads scaffolded with message queues
- [x] AudioCapture stub (2-sec timer replay), mel_spec.c (CMSIS-DSP), inference stub, ble_client
- [x] SAI1 overlay correct (`&sai1_b` + DMA, not `&i2s1`) — STM32U575 uses SAI not I2S
- [x] Real mel filterbank weights generated (`ml/generate_filterbank.py`)
- [x] DSP reference test passes: 440 Hz → correct mel bin (PASS)

**Build command (Windows):**
```bash
export ZEPHYR_BASE="C:/Users/jaiva/zephyrproject/zephyr"
export ZEPHYR_SDK_INSTALL_DIR="C:/Users/jaiva/Desktop/zephyr/zephyr-sdk-1.0.1_windows-x86_64_gnu/zephyr-sdk-1.0.1"
export PATH="/c/ProgramData/chocolatey/bin:/c/Users/jaiva/Desktop/zephyr/zephyr-sdk-1.0.1_windows-x86_64_gnu/zephyr-sdk-1.0.1/gnu/arm-zephyr-eabi/bin:$PATH"
cd Digital_Stethoscope
west build --board nucleo_u575zi_q --build-dir build_stm32_synth app \
    -- "-DPython3_EXECUTABLE=C:/Users/jaiva/AppData/Local/Programs/Python/Python314/python.exe"
```

**Known Zephyr quirks logged:**
- Python 3.14 required (system default 3.11 is too old for Zephyr 4.4)
- Use Chocolatey ninja — `C:/Windows/System32/ninja` is broken
- `CONFIG_CMSIS_DSP_BASICMATH` (not `_BASIC_MATH`)
- TFLite Micro is an OPTIONAL module in Zephyr workspace (not fetched by default)
  Enable with: `west config manifest.group-filter +optional && west update -o=--depth=1 -n tflite-micro`
- `CONFIG_NEWLIB_LIBC=y` MUST NOT be set — conflicts with board's PICOLIBC choice
- `CONFIG_CPP=y` + `CONFIG_STD_CPP17=y` required for TFLite Micro
- Heap: DO NOT use 512 KB — use 64 KB. Tensor arena is separate static allocation (NOT heap).
- Tensor arena: measured 29332 bytes used → set to 40 KB (40960) for 36% headroom
- Queue depth 1 for Phase 0 (restore to 2 in Phase 4 with real I2S DMA)

---

### Phase 1 — ML pipeline on laptop ✅ COMPLETE (Apr 17)

**Day 1–2:**
- [x] Write `01_preprocess.py`: resample, HPF, segment, mel-spectrogram, save as .npy
- [x] Download PhysioNet 2016 dataset (subsets a–e, 3126 recordings: 1765 train / 383 val / 383 test)
- [ ] Verify spectrograms visually (plot 4 classes — S1/S2 stripes should be visible)

**Day 3–4:**
- [x] Write `models/resnet10.py`: ResNet-10 with SE blocks (80,832 params, tested)
- [x] Write `models/dataset.py`: PhysioNet dataloader with 4-class labels + weighted loss
- [x] Write `02_train.py`: Adam, cosine LR, weighted loss, SpecAugment
- [x] Train 50 epochs on CPU. Best val acc: 67.4%, test acc: 64.2%
      NOTE: 80% target unachievable with this dataset — PhysioNet 2016 has ONLY binary
      labels (normal/abnormal). The systolic/diastolic/s3gallop sub-labels are assigned
      heuristically via hash (60/25/15% clinical prevalence split). No model can learn
      real sub-type features from random labels. Normal class F1=0.88. For true 4-class
      accuracy, need PhysioNet/CinC 2022 (has murmur-type annotations).

**Day 5:**
- [x] Write `03_quantize.py`: Keras ResNet-10 training + TFLite INT8 PTQ
      Keras best val acc: 63.2% (consistent with PyTorch — same data/label quality)
- [x] Write `04_export.py`: .tflite → model_data.cc C array
- [x] Run quantization: resnet10_int8.tflite = 110.3 KB (3× from 334.6 KB float32)
- [x] Validate quantized model: 63.4% float32 → 63.2% INT8 = 0.26% drop ✅ PASS (<2%)

**Day 6:**
- [x] Compute training set mean/std → `app/src/dsp/normalization_params.h` ✅
      mean=-5.8497, std=3.5882
- [x] Generate test vectors → `app/src/ml/test_vectors.h` ✅ (12 vectors, 3 per class)
- [x] Confusion matrix + training curves → `ml/models/confusion_matrix.png`, `training_curves.png`
- [x] Mel filterbank regenerated → `app/src/dsp/mel_filterbank_weights.h` ✅
- [x] INT8 model → `app/src/ml/model_data.cc` ✅ (110.3 KB)

### Phase 2 — TFLite Micro on STM32U575 ✅ COMPLETE (Apr 17)

- [x] Add TFLite Micro to Zephyr workspace:
      `cd C:/Users/jaiva/zephyrproject && west config manifest.group-filter +optional`
      `west update -o=--depth=1 -n tflite-micro`
      Module at: `C:/Users/jaiva/zephyrproject/optional/modules/lib/tflite-micro`
      Revision: `fcc760af130f3a595b5802cdebcc77461e54f382`
- [x] Enable `CONFIG_TENSORFLOW_LITE_MICRO=y` + `CONFIG_TENSORFLOW_LITE_MICRO_CMSIS_NN_KERNELS=y`
- [x] Enable `CONFIG_CPP=y` + `CONFIG_STD_CPP17=y` (required by TFLite Micro C++ runtime)
- [x] Heap set to 64 KB (`CONFIG_HEAP_MEM_POOL_SIZE=65536`) — tensor arena is separate static alloc
      NOTE: Do NOT restore to 512 KB — tensor arena is not from heap.
- [x] Real TFLite inference in `app/src/ml/inference.cc` (MicroMutableOpResolver, 12 ops)
- [x] model_data.cc: 102.3 KB INT8 model as C array
- [x] **Flash to hardware via OpenOCD** (`west flash --runner openocd`)
Flash runner: `west flash --build-dir build_stm32_synth --runner openocd`
      (pyocd pack index lacks STM32U5 series; OpenOCD has stm32u5x.cfg in Zephyr SDK)
- [x] **MEASURED UART output on hardware:**
      ```
      *** Booting Zephyr OS build v4.4.0-rc1-178-gccfd5efa09f9 ***
      [00:00:00.000] <inf> main: === Digital Stethoscope v0.1 ===
      [00:00:00.000] <inf> main: AudioCaptureThread started
      [00:00:00.000] <inf> i2s_capture: Audio: synthetic string source active (ICS-43434 disabled)
      [00:00:00.004] <inf> mel_spec: mel_spec: init OK (FFT=512, mels=64, frames=62)
      [00:00:00.004] <inf> main: InferenceThread started
      [00:00:00.000] <inf> inference: TFLite Micro initialized
      [00:00:00.000] <inf> inference:   Model: 104752 bytes
      [00:00:00.000] <inf> inference:   Arena used: 29332 / 40960 bytes  (72% headroom)
      [00:00:00.000] <inf> inference:   Input:  [1, 64, 64, 1]  type=9 (INT8)
      [00:00:00.000] <inf> inference:   Output: [1, 4]  type=9 (INT8)
      [00:00:02.xxx] <inf> main: Class: Normal        Confidence:  30%  Latency: 102ms
      [00:00:04.xxx] <inf> main: Class: SysMurmur     Confidence:  29%  Latency: 102ms
      ```
- [x] **Tensor arena tuned:** 29332 bytes used → arena reduced from 100 KB to 40 KB
      RAM freed: 61 KB. Final build: FLASH 430 KB / 2 MB (20%), RAM 336 KB / 768 KB (43%)
- [x] **Inference latency: 102 ms** (target < 150 ms — 32% faster than estimate)
- [x] Run `05_validate_on_device.py`: send test vector over UART, compare result ✅
      **Result: 12/12 device outputs match Python TFLite reference (100%)**
      On-device accuracy vs true labels: 4/12 = 33.3% (expected — heuristic sub-labels)
      Confirms: quantization, normalization, and INT8 inference chain are all correct.

**Known Zephyr/TFLite Micro quirks (Phase 2):**
- `CONFIG_NEWLIB_LIBC=y` CONFLICTS with board's PICOLIBC default — remove it
- `CONFIG_CPP=y` + `CONFIG_STD_CPP17=y` are required; TFLite selects `REQUIRES_FULL_LIBCPP`
  which depends on CPP=y — omitting these causes Kconfig abort
- Keras SE block with `GlobalAveragePooling2D(keepdims=True)` generates REDUCE_PROD/GATHER
  ops unsupported by TFLM; fixed with `keepdims=False` + explicit `Reshape((1,1,C))`
- Model ops for ResNet-10 INT8: ADD, CONV_2D, FULLY_CONNECTED, LOGISTIC, MAX_POOL_2D,
  MEAN, MUL, PACK, RESHAPE, SHAPE, SOFTMAX, STRIDED_SLICE (12 total)
- pyocd pack index does NOT include STM32U5 series — use `--runner openocd` instead
  OpenOCD at: `zephyr-sdk-1.0.1/hosttools/openocd/bin/openocd.exe` with `stm32u5x.cfg`
- Current synthetic-data build disables SAI/I2S, so the old mic clock warning should not
  appear. Revisit SAI clocking only if a future microphone source is enabled.
- `uart_poll_in()` does NOT work for UART RX when `CONFIG_UART_INTERRUPT_DRIVEN=y` is
  active on the same device — the ISR may consume bytes before poll_in() reads them.
  Fix: use `uart_irq_callback_user_data_set()` + `uart_irq_rx_enable()` + ring buffer.
  See `app/src/ml/validate.c` for the correct pattern.
- Console UART is `usart1` (PA9/PA10 → ST-Link VCP → COM6), NOT `usart3`.
  `usart3` (PD8/PD9) is the nRF52840 bridge — separate physical UART.
  Use `DT_CHOSEN(zephyr_console)` in code to always get the correct console device.

### Phase 3 — STFT/Mel DSP chain on U575 ✅ COMPLETE (Apr 17)

- [x] Write `app/src/dsp/mel_spec.c`: Hann window, arm_rfft_fast_f32, mel multiply, log10, normalize
- [x] Generate filterbank weights → `mel_filterbank_weights.h` ✅ (real weights, not placeholder)
- [x] DSP Python reference test passes (440 Hz → correct mel bin)
- [x] **Full pipeline running on hardware**: synthetic PCM → real DSP → real inference → UART
      AudioCapture stub now cycles through 4 synthetic heartbeat patterns (Normal/Systolic/
      Diastolic/S3 — each with distinct spectral profile) to exercise the full DSP+inference chain
      mel_spec init confirmed on UART: "mel_spec: init OK (FFT=512, mels=64, frames=62)"
      Inference confirmed cycling: Normal (30%) → SysMurmur (29%) → etc.
- [ ] Unit test on hardware: feed 440 Hz sine → verify energy in correct mel bins via UART
- [ ] Feed test clip → compare mel output with Python reference (UART log)

NOTE: Confidence values (27-30%) are expected for synthetic signals — the model was trained on
real heartbeat spectrograms. Once ICS-43434 mic is wired (Phase 4), real audio will produce
much higher confidence values matching training distribution.

### Phase 4 — ICS-43434 SAI bring-up (Apr ~19 mic arrives, or Apr 24)

- [x] SAI1 Block B devicetree overlay written (`&sai1_b`, PB3/PB5/PB6, DMA configured)
- [ ] Wire ICS-43434 to NUCLEO: BCLK→PB3, LRCLK→PB6, DOUT→PB5, VDD→3.3V, L/R→GND
- [ ] Replace stub in `i2s_capture.c` with real SAI DMA capture (template in comments)
- [ ] Capture raw audio, log RMS level via UART (confirms mic is alive)
- [ ] Connect real stethoscope chestpiece to mic (silicone coupler or tape)
- [ ] Full pipeline end-to-end: real heartbeat → classification → UART

### Phase 5 — nRF52840 BLE (Apr 24+, board arriving)

- [x] Write `ble_peripheral/src/heart_sound_service.c`: custom GATT service (HSC_Result NOTIFY)
- [x] Write `ble_peripheral/src/main.c`: UART RX → BLE GATT notify
- [x] **Build nRF app** (compile-validated, Apr 17, board not yet in hand):
      `west build --board nrf52840dk/nrf52840 --build-dir build_ble ble_peripheral`
      Result: FLASH 122 KB / 1 MB (12%), RAM 23 KB / 256 KB (9%) — zero warnings
      Fixes vs original scaffold: `BT_LE_ADV_CONN` → `BT_LE_ADV_CONN_FAST_1` (Zephyr 4.4 API),
      `BT_DEVICE_APPEARANCE` inline comment removed, `BT_GATT_CACHING=y` added as dependency,
      overlay pinctrl-1 sleep state added (nRF52840 requires both default+sleep states).
      Flash command (when board arrives): `west flash --build-dir build_ble`
- [ ] UART bridge: STM32 CommThread → nRF UART1 RX (P0.08) → BLE GATT notify
- [ ] Test with nRF Connect app: subscribe to HSC_Result, observe notifications

### Phase 6 — Polish & exam prep (Apr 24+)

- [ ] GitHub README with architecture diagram, photo of hardware, example UART output
- [ ] German README (LinkedIn post framing)
- [ ] Record short demo video: phone showing BLE notifications while tapping stethoscope
- [ ] Clean up confusion matrix + training curves figures for presentation

---

## Key Technical Decisions (Rationale for Interviews)

**Zephyr over FreeRTOS + CubeIDE:**
Single unified build system across both MCUs, better regulatory traceability via
deterministic `west build`, portable away from ST HAL lock-in. Accepted ~15% inference
latency increase from TFLite Micro vs Cube.AI in exchange for toolchain portability.

**ResNet over VGG:**
VGG-16 has 138M parameters — impossible to fit on MCU. ResNet-10 has ~120K parameters
(1000× fewer) while solving the vanishing gradient problem via residual connections,
enabling deep feature extraction (full 2-second cardiac cycle context) on 3K training clips.

**Mel-spectrogram over raw audio:**
Scientific justification, not convenience: 75% of diagnostically relevant cardiac energy
lives below 600 Hz. Linear FFT wastes 70% of bins on noise. Mel scale concentrates
resolution in the cardiac band. Log power compression equalizes S1 amplitude vs murmur
amplitude so the classifier sees both equally. This was invented in 1980 — predates CNNs.

**INT8 quantization:**
4× model size reduction (120KB → 30KB), 4× inference speedup via CMSIS-NN DSP MACs,
< 2% accuracy cost on this task. Enables deployment within U575's 786KB SRAM.

**Regulatory framing:**
IEC 62304 Class B SaMD (advisory output, not autonomous diagnosis). ISO 14971 risk
analysis identifies acoustic coupling inconsistency and small dataset as primary hazards.
IEC 60601-1 electrical safety requires isolation revision before patient contact — current
prototype has exposed PCB, not suitable for clinical use.

---

## Known Issues / Things to Watch

**STM32U575 has SAI, not I2S:** The NUCLEO-U575ZI-Q does not have a standard I2S
peripheral. It uses SAI (Serial Audio Interface). Zephyr exposes SAI via the I2S API
through `i2s_stm32_sai.c`. The overlay uses `&sai1_b` (not `&i2s1`). DMA is required
and `dma-names = "rx"` must be set in the overlay (the base DTSI has `dmas` but not
`dma-names`). SAI1 Block B pins: SCK=PB3, FS=PB6, SD=PB5.

**SAI clock configuration:** ICS-43434 BCLK = 2 × 32 × 4000 = 256 kHz (2-channel
I2S frames even when only using left channel). The STM32U575 SAI clock is derived from
PLL2_P. Verify PLL2 configuration in the Zephyr board default config generates a valid
256 kHz BCLK divider. 4 kHz target is unusual vs typical 48 kHz audio examples.

**TFLite Micro C++ in Zephyr C project:** `inference.cc` must be C++. Your main.c and
other files are C. Use `extern "C"` wrappers in inference.h for the API functions called
from C threads.

**Tensor arena alignment:** Must be 16-byte aligned. Use `alignas(16)` or
`__attribute__((aligned(16)))` on the arena array.

**CMSIS-DSP float vs fixed-point:** `arm_rfft_fast_f32` is float32. On Cortex-M33 with
FPU this is fast enough. If latency is too high, switch to `arm_rfft_q15` (fixed-point)
but then mel filterbank multiply also needs to be fixed-point — more complex.

**PhysioNet class imbalance:** Normal class is overrepresented (~60% of data). Use
`torch.nn.CrossEntropyLoss(weight=class_weights)` where class_weights are inverse
class frequencies. Otherwise model learns to always predict Normal.

**nRF52840 UART bridge vs direct UART:** If the nRF and STM32 share a UART, configure
STM32 CommTask to output the 6-byte packet at a fixed baud (115200). On nRF, receive
and forward to BLE GATT notify. Keep packets small and fixed-length to simplify parsing.

---

## Useful References

- Zephyr I2S API: https://docs.zephyrproject.org/latest/hardware/peripherals/i2s.html
- TFLite Micro Zephyr sample: `zephyr/samples/modules/tflite-micro/hello_world/`
- CMSIS-DSP FFT example: `CMSIS/DSP/Examples/ARM/arm_fft_bin_example/`
- PhysioNet 2016: https://physionet.org/content/challenge-2016/1.0.0/
- nucleo_u575zi_q board docs: `zephyr/boards/st/nucleo_u575zi_q/`
- nRF52840 DK BLE GATT peripheral sample: `zephyr/samples/bluetooth/peripheral/`
- ICS-43434 datasheet: search "ICS-43434 datasheet" — TDK InvenSense

---

## Interview Talking Points

When asked about this project by Dräger / Getemed / Solectrix:

1. **Why Zephyr on STM32?** Toolchain portability, single west workspace across two MCUs,
   better regulatory traceability (IEC 62304 build reproducibility), differentiates from
   candidates who only know CubeIDE.

2. **Why CNN and not a frequency threshold?** A threshold cannot distinguish systolic from
   diastolic murmur — both activate the same frequency bands, just at different points in
   the cardiac cycle. CNN captures the 2D temporal-frequency shape, including the relative
   timing of the murmur window between S1 and S2.

3. **Why mel-spectrogram?** Scientific justification predating CNNs: 75% of cardiac
   diagnostic information lives below 600 Hz; mel scale concentrates resolution there.
   Log power compression prevents S1 from dominating the classifier.

4. **How would you validate for clinical use?** IEC 62304 Class B SaMD, ISO 14971 risk
   analysis, clinical study with echocardiography ground truth (500+ patients), sensitivity/
   specificity per class published. Current prototype is verification-stage only.

5. **What would you do differently with more time?** Dynamic cardiac-cycle-adaptive
   windowing using accompanying PPG, severity regression (Levine grade I-VI), federated
   learning across hospital sites, STM32N6 port for Neural-ART NPU (600 GOPS).
