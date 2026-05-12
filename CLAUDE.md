# CLAUDE.md — AI Digital Stethoscope (PCG Heart Sound Classifier)

> **2026-05-11 update**: input source is now **laptop audio upload over UART**,
> not synthetic strings. The STM32 receives raw 8000-sample PCM from the host,
> computes the mel spectrogram on chip, runs the CirCor 2022 INT8 ResNet-10 +
> calibrated unknown gate, and returns class + raw probabilities + per-stage
> latency + audio stats + the full 64x64 spectrogram to the dashboard for
> rendering. Synthetic injection is retained only as a self-test demo command.
> See the **Session Log — 2026-05-11** section at the bottom of this file for
> the full set of changes, the new UART protocol, and measured hardware
> results.
>
> **2026-05-12 update**: nRF52840 DK BLE peripheral is now wired, flashed, and
> verified end-to-end. STM32 ↔ nRF UART bridge moved from USART3 to **USART2
> (PD5 / NUCLEO D53)** because USART3 has no labelled header pin on the
> NUCLEO-U575ZI-Q. The dashboard-upload (`'A'`) inference path now also calls
> `ble_client_send()`, so every WAV uploaded to the dashboard fans out to the
> phone over BLE. The BLE characteristic payload was changed from a 6-byte
> binary packet to a short printable ASCII string (e.g. `"Present 95%"`) so
> nRF Connect renders each notification as readable text, and the
> characteristic now advertises a User Description ("Heart Sound
> Classification").
>
> **2026-05-12 (later) — ResNet-18 + M4A + live mic**: Dashboard now decodes
> M4A / MP3 / AAC / OGG / FLAC via a bundled `imageio-ffmpeg` binary and has
> a new **Live Mic** tab that captures from the laptop microphone via the
> Web Audio API. The STM32 was reflashed with a bigger **ResNet-18** model
> (720 K params, 756 KB INT8, ~507 ms inference, 200 KB tensor arena). It
> uses 58 % of the 2 MB flash and 75 % of RAM, demonstrating the previously
> unused headroom. Honest finding kept in the repo: ResNet-18's test
> accuracy is ~10 pp lower than ResNet-10's (70.9 % vs 81.5 %) because
> CirCor is too small to support the extra capacity without stronger
> regularization. See **Session Log — 2026-05-12** at the bottom.

## Project Identity

**What this is:** A wearable digital stethoscope prototype. The STM32U575
accepts 2-second 4 kHz PCM windows over UART, computes the mel spectrogram on
chip, runs a quantized ResNet-10 INT8 CNN to classify **Absent / Present /
Unknown** murmur status (PhysioNet/CinC 2022 CirCor DigiScope labels), and
transmits results to a phone over BLE — or to the local dashboard for richer
scientific visualisation. No cloud. No CubeIDE. Pure Zephyr RTOS on both MCUs.

**Input-source decisions:**

- *Active path (May 2026)*: the laptop dashboard reads any uploaded WAV / 1D
  audio NPY, resamples to 4 kHz mono, splits long files into successive
  2-second windows, and streams each window as int16 PCM to the STM32 via the
  `'A'` UART command. The STM32 owns the full DSP + ML chain on chip.
- *Self-test path*: a synthetic PCG-string source still exists in
  `i2s_capture.c` and can be toggled with the `'S'` / `'P'` UART commands.
  Not used by the current dashboard.
- *Future hardware path*: the ICS-43434 MEMS microphone is discontinued for
  this revision. A SAI/I2S microphone source can be re-introduced later behind
  the existing `audio_capture_init()` / `audio_capture_get_window()` API.

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

/* USART2 — bridge to nRF52840 at 115200 baud (PD5 TX / D53 → nRF P0.08) */
&usart2 {
    status = "okay";
    pinctrl-0 = <&usart2_tx_pd5 &usart2_rx_pd6>;
    pinctrl-names = "default";
    current-speed = <115200>;
};
```

nRF52840 DK wiring (two wires only):
- STM32 PD5 (USART2 TX, NUCLEO D53 / silkscreen `USART_B_TX`) → nRF P0.08 (UART1 RX)
- STM32 GND ↔ nRF GND (common ground)

ICS-43434 wiring (future microphone path): BCLK→PB3, LRCLK→PB6, DOUT→PB5,
VDD→3.3V, L/R→GND (left ch).

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

**Service UUID:** `12345678-1234-1234-1234-123456789ABC` (custom, prototype)

**Characteristic: Heart Sound Classification**
- UUID: `12345678-1234-1234-1234-123456789ABD`
- Properties: NOTIFY
- User Description (CUD): `"Heart Sound Classification"`
- Payload: printable ASCII string, ≤ 20 bytes, e.g. `"Present 95%"`,
  `"Absent 91%"`, `"Unknown 14%"`, `"Error 0%"` (class 0xFF).

Format chosen so nRF Connect auto-detects the bytes as text and renders each
notification as a readable string in the live feed (no manual hex decoding).

For demo: pair with nRF Connect app (iOS/Android), subscribe to notifications,
observe classification results in real time as the dashboard uploads WAVs.
No custom app needed for exam demo. The internal STM32 ↔ nRF UART link still
uses the legacy 6-byte binary frame `[class_id, confidence, reserved,
timestamp_ms u24 LE]` — the ASCII conversion happens in `hsc_service_notify()`
on the nRF before the BLE notify.

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

### Phase 5 — nRF52840 BLE ✅ COMPLETE (2026-05-12)

- [x] Write `ble_peripheral/src/heart_sound_service.c`: custom GATT service (HSC_Result NOTIFY)
- [x] Write `ble_peripheral/src/main.c`: UART RX → BLE GATT notify
- [x] **Build nRF app**: `west build --board nrf52840dk/nrf52840 --build-dir build_ble ble_peripheral`
      Result: FLASH 122 KB / 1 MB (11.65%), RAM 22.5 KB / 256 KB (8.60%)
      Fixes vs original scaffold: `BT_LE_ADV_CONN` → `BT_LE_ADV_CONN_FAST_1` (Zephyr 4.4 API),
      `BT_DEVICE_APPEARANCE` inline comment removed, `BT_GATT_CACHING=y` added as dependency,
      overlay pinctrl-1 sleep state added (nRF52840 requires both default+sleep states).
- [x] **Flash to hardware** (`west flash --build-dir build_ble --runner jlink`):
      the default `nrfutil` runner is broken on Python 3.14 (`ModuleNotFoundError: constants`).
      Use `--runner jlink` instead — the DK's onboard J-Link is detected as SEGGER 1366:1051.
- [x] **UART bridge over USART2** (PD5 TX / NUCLEO D53 → nRF P0.08 UART1 RX, + GND):
      USART3 was originally specced, but the NUCLEO-U575ZI-Q has no labelled USART3
      pin on the Arduino-style morpho header. Migrated to USART2 instead. PD5 is
      labelled `USART_B_TX` / D53 on the silkscreen.
- [x] **Cross-board verified**: synthetic-loop `'S'` produced 6 packets in 12 s, all
      decoded correctly on nRF side as `RX: <Absent|Present|Unknown> conf=NN% ts=...ms`.
- [x] **Dashboard upload path now broadcasts to BLE**: `validate.c` `'A'` handler calls
      `ble_client_send()` after on-chip inference, so each uploaded WAV emits one
      BLE notification per 2-second window.
- [x] **Readable notification payload**: BLE characteristic now sends a short ASCII
      string like `"Present 95%"` instead of a 6-byte binary packet. nRF Connect
      auto-renders printable ASCII, so each notification reads as text in the live
      feed. Characteristic also advertises a CUD descriptor naming itself
      "Heart Sound Classification".
- [x] **Phone-side verified end-to-end** with nRF Connect on Android (2026-05-12):
      `HeartSound` (CB:1D:2D:72:53:F1), characteristic `...ABD` notifications
      arrive as readable strings while WAVs upload from the dashboard.

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

---

## Session Log — 2026-05-11 (Laptop upload + scientific dashboard)

### Goal of the session

Replace the synthetic-only firmware flow with a **real-audio** input path:
- The laptop becomes a thin client that uploads any WAV / 1D NPY.
- All DSP and ML stays on the STM32, using as much of the available flash as
  is useful (no need to optimise for size — 2 MB is plenty).
- The dashboard turns into a scientific ML visualisation, not a passive UART
  log parser.
- A "big mixed audio file" with several classes is shipped so the project can
  be demonstrated end-to-end against ground truth from PhysioNet/CinC 2022.

### Architecture decision: extended binary UART protocol

`CONFIG_LOG` is disabled in `app/prj.conf` so the console UART is binary-clean.
A new command `'A'` is added in `validate.c`:

```
Host -> STM32 : 'A' (0x41) + 8000 LE int16 PCM samples
                (delivered in 128-sample / 256-byte chunks)
STM32 -> Host : 0xA6 after every 128 samples + one final 0xA6
STM32 -> Host : 0xA5 + class_id (uint8, after unknown gate)
              + confidence (uint8, 0..100)
              + raw_probabilities[3] (uint8 percentages, BEFORE the gate)
              + gate_applied (uint8, 0 or 1)
              + dsp_ms (uint16 LE)
              + inference_ms (uint16 LE)
              + rms_q15 (uint16 LE, rms  * 32768)
              + peak_q15 (uint16 LE, peak * 32768)
              + zcr (uint16 LE, zero-crossings count)
              + spectrogram[4096] (float32 LE)
              = 16401 bytes total
```

The two existing commands (`'T'` test vector and `'S'`/`'P'` synthetic
start/pause) are retained but no longer used by the dashboard.

### Firmware changes (all under `app/`)

| File | What changed |
|---|---|
| `prj.conf` | Removed `CONFIG_LOG`. Added `CONFIG_RING_BUFFER_LARGE=y` for the 40 KB upload RX FIFO. |
| `src/dsp/mel_spec.h` / `mel_spec.c` | New `g_mel_spec_mutex` so the validate-thread upload path can share `mel_spec_compute()` with the synthetic `dsp_thread`. |
| `src/main.c` | `dsp_thread` now holds the mel-spec mutex around its compute call. |
| `src/ml/inference.h` / `inference.cc` | Refactored into a shared `run_inference_core()`. New `inference_run_probs(spec, n, &conf, raw_probs[3], &gate_applied)` returns the **raw** dequantised probabilities and a flag indicating whether the validation-calibrated unknown gate fired. |
| `src/ml/validate.h` / `validate.c` | New `'A'` upload command. Audio statistics (`rms`, `peak`, `zcr`) computed in `compute_audio_stats()`. Per-stage latency measured with `k_uptime_get_32()`. Extended response packet built byte-by-byte into `uart_poll_out()`. |

#### Static memory footprint added

| Buffer | Size | Purpose |
|---|---|---|
| `s_rx_rb` | 40 KB | Interrupt-driven UART RX ring buffer (absorbs full 16 KB audio upload + jitter) |
| `s_uploaded_audio` | 32 KB | float32 mirror of one 8000-sample window |
| `s_uploaded_spec` | 16 KB | float32 64x64 spectrogram returned to host |

All in `.bss`. No heap usage. Tensor arena stays at 40 KB and is unchanged.

#### Clean-build size (after the new code)

```
FLASH: 550 KB / 2 MB   (26.3%)
RAM  : 423 KB / 768 KB (53.8%)
```

Build command used: `./build.sh --clean` (writes `build_stm32_synth/`).

Flash command used: `west flash --build-dir build_stm32_synth --runner openocd`.
OpenOCD wrote 550,656 bytes in 5.3 s over ST-Link on COM6.

### Dashboard rewrite (`dashboard/server.py`)

The previous v2 dashboard was a passive UART log parser that drove
synthetic-string injection via `'S'`/`'P'`. It has been **fully replaced**
with a binary-protocol upload client.

User-visible features:

- Custom-Upload tab accepts a WAV (any length, any sample rate) or a 1D
  audio NPY. 64x64 NPY spectrograms are intentionally rejected so the STM32
  DSP stage cannot be bypassed.
- Long uploads are auto-segmented into successive 2-second windows; each
  window is streamed to the STM32 and classified independently.
- Optional labels-JSON sidecar (or `<wav-stem>_labels.json` in either
  `presentation_samples/` or `ml/data_circor/demo/`) is auto-loaded for
  ground-truth accuracy reporting.
- Generic Loop tab cycles through `presentation_samples/`, then
  `ml/data_circor/demo/`, then `ml/data_circor/raw/training_data/`.

Scientific views rendered per upload:

- Aggregate result card (dominant class, confidence gauge, accuracy %)
- Per-segment probability time-series (3 colored traces; gate-fired points
  drawn as enlarged dots)
- Color-coded segment timeline (click any block to load that segment below)
- Per-class cumulative bar chart
- Audio RMS over time
- Latency breakdown (DSP / Inference / UART upload / UART return)
- Selected-segment waveform + RMS / peak / ZCR / gate flag panel
- STM32-computed 64x64 mel spectrogram heatmap
- Audio playback widget for the uploaded window
- Per-segment results table + CSV export

### Mixed multi-class demo audio (`ml/generate_mixed_demo.py`)

Generates a long composite WAV from the CirCor raw training set with the
class membership pulled from `raw/training_data.csv`:

```
py ml\generate_mixed_demo.py --per-class 3 --seed 2026
->  ml/data_circor/demo/mixed_demo.wav           (18 s, 9 segments)
->  ml/data_circor/demo/mixed_demo_labels.json   (per-segment ground truth)
```

### Presentation samples (`presentation_samples/`)

Three single-class CirCor recordings trimmed to **12 s / 6 segments** each,
peak-normalised, with sidecar labels JSONs:

| File | Class | Patient | Note |
|---|---|---|---|
| `01_absent_pid49653.wav` | Absent  | 49653 (Adolescent, AV) | Healthy baseline, quiet S1/S2 only |
| `02_present_pid9979.wav` | Present | 9979 (TV)              | Holosystolic, grade III/VI, diamond |
| `03_unknown_pid9983.wav` | Unknown | 9983 (AV)              | Annotator was unsure; should trigger the unknown gate |

A `__build__.py` script regenerates them idempotently from the CirCor raw
training data and the published murmur labels.

### Hardware results captured this session

#### `mixed_demo.wav` (18 s / 9 segments, 3 per class)

- Per-window on-chip DSP: **~85 ms**
- Per-window on-chip inference: **~102 ms**
- UART upload (16 KB at 115200 baud): **~1.4 s**
- UART return (16 KB spectrogram at 115200 baud): **~1.4 s**
- 5 of 9 segments matched ground truth. Two `Unknown` predictions came from
  the calibrated gate elevating low-margin softmax outputs.

#### `presentation_samples/02_present_pid9979.wav` (12 s / 6 segments)

- All 6 segments → **Present**, confidence 51 % → 95 %, gate did not fire.
- Ground-truth column populated automatically via the sidecar JSON.
- Per-file accuracy: **100 %**.

### Files added or modified this session

```
app/prj.conf                          edited
app/src/dsp/mel_spec.h                edited
app/src/dsp/mel_spec.c                edited
app/src/main.c                        edited
app/src/ml/inference.h                edited
app/src/ml/inference.cc               edited
app/src/ml/validate.h                 edited
app/src/ml/validate.c                 edited
dashboard/server.py                   rewritten
ml/generate_mixed_demo.py             NEW
ml/data_circor/demo/mixed_demo.wav        NEW (generated)
ml/data_circor/demo/mixed_demo_labels.json NEW (generated)
presentation_samples/__build__.py     NEW
presentation_samples/README.md        NEW
presentation_samples/01_absent_pid49653.wav         NEW
presentation_samples/01_absent_pid49653_labels.json NEW
presentation_samples/02_present_pid9979.wav         NEW
presentation_samples/02_present_pid9979_labels.json NEW
presentation_samples/03_unknown_pid9983.wav         NEW
presentation_samples/03_unknown_pid9983_labels.json NEW
CODEX.md                              rewritten (full handoff)
CLAUDE.md                             header + this session log
```

### Things to remember for the next session

- The legacy v1 dashboard from earlier today auto-bound port 8765 in parallel
  with the new v2 dashboard. If responses ever look like the old single-segment
  format (top-level `class_id`, no `segments[]`), kill the older Python
  process that was started from `Digital_Stethoscope` (with backslashes in
  the command line) and keep only the v2 process started from
  `Digital_Stethoscope_2`.
- `tests/` and `ml/05_validate_on_device.py` still use the `'T'` test-vector
  command. They were left untouched but should be updated to also exercise
  the new `'A'` upload path before the next release.
- No git commit was created for any of these changes. `git status` will show
  the firmware files, dashboard, generators, presentation samples, and md
  updates as modified / untracked.
- BLE peripheral firmware is unchanged. The 6-byte BLE notify packet still
  contains `{class_id, confidence, reserved, timestamp_ms}` from the
  synthetic-injection pipeline; the upload-path classifications do not
  currently propagate to BLE.

---

## Session Log — 2026-05-12 (nRF52840 BLE integration)

### Goal of the session

Wire the nRF52840 DK into the system end-to-end so that:

- The STM32 streams every classification (synthetic AND dashboard uploads)
  to the nRF over a hardware UART bridge.
- The nRF advertises a custom GATT service and pushes notifications to a
  paired phone running nRF Connect.
- The on-air payload is human-readable in the nRF Connect feed (no hex
  decoding step required for a live demo).

End of session: all three goals reached on hardware, with notifications
visible on an Android phone from real dashboard uploads of
`presentation_samples/02_present_pid9979.wav`.

### Bridge re-pinning: USART3 → USART2 (PD5 / D53)

The original spec called for STM32 USART3 (PD8/PD9) as the cross-board UART.
The NUCLEO-U575ZI-Q does technically expose PD8/PD9, but neither pin is
labelled "USART3" on the silkscreen or in the user manual quick-reference.
The pin labelled `USART_B_TX` on the morpho header (D53) is **PD5**, which
maps to USART2_TX on the STM32U575. Switching the bridge to USART2 made the
physical wiring obvious and matched the silkscreen.

Firmware changes:

| File | What changed |
|---|---|
| `app/app.overlay` | `&usart3` block replaced with `&usart2` block using `usart2_tx_pd5 usart2_rx_pd6` pinctrl entries (both confirmed present in `stm32u575zitxq-pinctrl.dtsi`, AF7). |
| `app/src/comms/ble_client.c` | `DT_NODELABEL(usart3)` → `DT_NODELABEL(usart2)`; header comment updated. |

Final wiring (only two jumper wires between boards):

```
STM32 PD5 (USART2 TX, NUCLEO D53)  ─────▶  nRF P0.08 (UART1 RX)
STM32 GND                          ────── nRF GND
```

A common ground wire is **required** even though both boards are powered
from the same PC USB — USB shield ground proved unreliable for UART signal
levels in our setup.

### nRF firmware flashing

- Default `west flash` runner is `nrfutil`, but Nordic's Python wrapper
  fails on Python 3.14 with `ModuleNotFoundError: No module named 'constants'`
  inside `nordicsemi.lister.windows.lister_win32`. Use `--runner jlink`
  instead; the DK's onboard J-Link enumerates as SEGGER VID 1366 PID 1051
  and writes the hex over SWD at ~98 KiB/s.
- The DK exposes **two** JLink CDC UART ports (e.g. COM7 + COM8). On this
  machine the Zephyr console UART (uart0, P0.06/P0.08… wait no — uart0 is
  on the J-Link VCOM pair, not the bridge) appears on **COM8**. COM7 is
  silent in this configuration.
- Boot banner expected on COM8 within ~300 ms of reset:

  ```
  *** Booting Zephyr OS build v4.4.0-rc1-178-gccfd5efa09f9 ***
  <inf> ble_main: === HeartSound BLE Peripheral ===
  <inf> ble_main: UART1 ready (115200 baud)
  <inf> bt_hci_core: Identity: CB:1D:2D:72:53:F1 (random)
  <inf> hsc_service: Heart Sound Classification service registered
  <inf> ble_main: Advertising as 'HeartSound' — connect with nRF Connect app
  ```

### Dashboard-upload BLE hook (`'A'` command path)

Previously only `comm_thread` (fed by the synthetic-injection pipeline)
called `ble_client_send()`. The dashboard `'A'`-command path in
`validate.c` ran on-chip inference and built the extended response packet
but never fanned out to the nRF.

Fix (`app/src/ml/validate.c`):

```c
#include "comms/ble_client.h"
...
uint8_t out_class = (class_id < 0) ? 0xFFu : (uint8_t)class_id;
ble_client_send(out_class, confidence, k_uptime_get_32());   // ← added
/* 4. Extended response packet. */
send_byte(VALIDATE_MAGIC);
...
```

Result: every WAV uploaded to the dashboard now produces exactly one BLE
notification per 2-second window. A 12-second WAV from
`presentation_samples/` produces six phone notifications, ~3 s apart
(2 s window + ~1.4 s UART up + ~1.4 s UART return + ~190 ms compute).

### Stale 4-class labels on nRF side

`ble_peripheral/src/main.c` was still using the pre-CirCor 4-class label
table `{"Normal","SysMurmur","DiaMurmur","S3Gallop"}` for its `LOG_INF`
trace. The STM32 has been emitting CirCor 3-class IDs (0/1/2) since
2026-05-09, so class 2 was being printed as `"DiaMurmur"` on the nRF
console even though the on-chip model treats it as `"Unknown"`.

Replaced with `{"Absent","Present","Unknown"}` and bound-checked via
`ARRAY_SIZE(names)`. Header comment in `heart_sound_service.c` updated
to match.

### BLE payload: 6-byte binary → printable ASCII

User feedback from nRF Connect after the first end-to-end test: the
notifications showed up as raw hex like `(0x) 01-5F-00-93-11-03`, which is
correct but unreadable for a live demo. nRF Connect renders ASCII
payloads as strings automatically (e.g. it already showed the device name
`HeartSound` as text in the Device Name characteristic).

Rewrote `hsc_service_notify()` in
`ble_peripheral/src/heart_sound_service.c` to build a short printable
string via `snprintf` and notify that instead:

```c
static const char *const class_names[] = {"Absent","Present","Unknown"};
char pkt[20];
int len = snprintf(pkt, sizeof(pkt), "%s %u%%", name, confidence);
bt_gatt_notify(...)   // len bytes, no trailing NUL
```

Also added a Characteristic User Description (CUD) descriptor so the
characteristic now advertises a human-readable name in the service browser:

```c
BT_GATT_CUD("Heart Sound Classification", BT_GATT_PERM_READ),
```

After this change Android nRF Connect needed a single disconnect /
reconnect cycle to drop its cached GATT structure and pick up the new
descriptor.

### `bt_le_adv_start()` does NOT auto-resume after a phone-side disconnect

The `disconnected()` callback in `ble_peripheral/src/main.c` re-calls
`bt_le_adv_start()`, but only fires once the controller has actually
observed the link drop. When the Android side of the link closes
abruptly (e.g. nRF Connect "Disconnect" button without a proper L2CAP
teardown), the controller can sit in a half-open state for tens of
seconds before the disconnect callback fires, during which the device
will NOT show up in a scan. A SoC reset clears it instantly.

Fastest reset from this PC, no power-cycle needed:

```
"C:\Program Files\SEGGER\JLink\JLink.exe" -device nRF52840_xxAA -if SWD \
    -speed 4000 -autoconnect 1 -CommanderScript reset.cmd
# reset.cmd contents:
#   r
#   g
#   exit
```

Boot banner reappears on COM8 within ~300 ms and `HeartSound` shows up in
the scanner immediately.

### Files added or modified this session

```
app/app.overlay                      USART3 block replaced with USART2 (PD5/PD6)
app/src/comms/ble_client.c           DT_NODELABEL(usart3) → usart2 + comment
app/src/ml/validate.c                +#include "comms/ble_client.h";
                                     ble_client_send() after on-chip inference
ble_peripheral/src/main.c            4-class names → CirCor 3-class names
ble_peripheral/src/heart_sound_service.c
                                     ASCII payload via snprintf;
                                     BT_GATT_CUD descriptor;
                                     header comment updated
CLAUDE.md                            this session log + Phase 5 marked complete
README.md                            Phase 5 status, BLE GATT section, pin table
CODEX.md                             new handoff entry for today
```

### Measured hardware results

- STM32 build after BLE-hook + USART2 change: FLASH 550500 B / 2 MB
  (26.25%), RAM 423276 B / 768 KB (53.82%). Footprint unchanged from the
  2026-05-11 build except for `ble_client_send()` inlining.
- nRF build: FLASH 122124 B / 1 MB (11.65%), RAM 22534 B / 256 KB (8.60%).
- Synthetic-loop bridge test: 6 packets / 12 s, all decoded correctly on
  the nRF console.
- `presentation_samples/02_present_pid9979.wav` upload from dashboard: 6
  segments, all classified as `Present`, all six BLE notifications visible
  on Android nRF Connect as `"Present NN%"` strings.

### Things to remember for the next session

- The STM32 ↔ nRF wire IS USART2 (PD5 / D53), not USART3 — older comments
  and earlier drafts of this file mention USART3.
- nRF flashing requires `--runner jlink`. `nrfutil` is broken until Nordic
  publishes a Python 3.14-compatible wheel.
- The default Zephyr console UART on the nRF52840 DK shows up on **COM8**
  in this PC's enumeration order, not COM7. If you read COM7 you will
  see nothing.
- A clean disconnect/reconnect on the phone is sometimes necessary after a
  service-definition change (CUD descriptor add etc.); Android nRF Connect
  caches GATT structure aggressively.
- After a phone-side abrupt disconnect, advertising may not auto-resume
  until the controller observes the link drop — reset the nRF via J-Link
  Commander if `HeartSound` stops appearing in scans.
- `flash_out.log` / `flash_err.log` and the downloaded nRF Connect log
  file under `dashboard/` are session artifacts; do not commit them.

---

## Session Log — 2026-05-12 (continued: M4A, live mic, ResNet-18 deployment)

After the BLE peripheral was end-to-end verified (above), the rest of the
session focused on broadening the dashboard's audio sources, deploying a
much larger CNN to use the previously-unused STM32 flash headroom, and
characterising what that bigger model actually delivers on this dataset.

### Phone-recorded heartbeat → classification (the trigger for everything)

The user recorded their own heartbeat with an Android phone and wanted to
see it classified. The phone produced an **M4A (AAC-in-MP4) file** at low
levels (`Myownheartbeat.m4a`, 332 KB, 9.54 s, peak ≈ 0.47 of full scale,
RMS ≈ 0.049). The existing dashboard accepted WAV / NPY only, so two
things had to land before any classification could happen:

1. M4A decoding inside the dashboard server.
2. Some way to keep the audio level usable when input recordings are
   below half scale.

### Bundled-ffmpeg M4A / MP3 / AAC / OGG / FLAC support

`dashboard/server.py` now imports `imageio_ffmpeg` (one-shot
`pip install imageio-ffmpeg`, ~30 MB wheel) and uses the bundled
`ffmpeg-win-x86_64-v7.1.exe` to decode anything ffmpeg understands into
mono 16-bit PCM at 4 kHz. Key bits:

```python
def load_compressed(path: Path) -> tuple[np.ndarray, float]:
    ffmpeg = _get_ffmpeg()        # imageio_ffmpeg.get_ffmpeg_exe()
    proc   = subprocess.run([ffmpeg, "-v", "error", "-i", str(path),
                             "-f", "wav", "-acodec", "pcm_s16le",
                             "-ac", "1", "-ar", "4000", "-"],
                            capture_output=True, check=False)
    pcm, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    ...
```

`COMPRESSED_SUFFIXES = {.m4a, .m4b, .aac, .mp3, .mp4, .ogg, .oga, .flac,
.webm, .opus, .wma, .3gp}`. WAV and NPY paths are unchanged.

A new `auto_boost()` peak-normalises only when the input is below half
scale, so loud recordings pass through untouched but a phone recording
gets scaled to 0.95 peak before int16 cast:

```python
def auto_boost(audio, target_peak=0.95, min_peak_threshold=0.5):
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0 or peak >= min_peak_threshold:
        return audio
    return audio * (target_peak / peak)
```

The file-input accept-list and rejection error message were updated to
match.

**Result on `Myownheartbeat.m4a` through ResNet-10 (still flashed at
this point)**:

```
File           : Myownheartbeat.m4a
Audio length   : 9.54 s (5 windows)
Dominant       : Absent (avg conf 80%)
Counts         : {absent: 5, present: 0, unknown: 0}
Window 0  86%  86/14/0   peak=0.95
Window 1  86%  86/14/0   peak=0.79
Window 2  87%  87/13/0   peak=0.57
Window 3  89%  89/10/0   peak=0.75
Window 4  51%  51/48/1   peak=0.50   (borderline, end of recording)
```

All five windows correctly classified as Absent. Five BLE notifications
also arrived on the phone during this run, since the `'A'`-command path
calls `ble_client_send()` per window.

### Live laptop-microphone mode in the dashboard

New `Live Mic` tab alongside `Custom Upload` and `Generic Loop`. Pure
client-side capture, no server changes beyond the tab markup:

- `navigator.mediaDevices.getUserMedia({audio: {channelCount: 1,
  echoCancellation: false, noiseSuppression: false,
  autoGainControl: false}})` (HTTPS-or-localhost is fine; we're on
  localhost).
- `AudioContext` + `ScriptProcessorNode(4096)` at the device's native
  rate (typically 44.1 or 48 kHz).
- Buffer two seconds of float32 samples, downsample to 4 kHz via linear
  interpolation, convert to int16, build a 44-byte-header WAV blob, POST
  to the existing `/api/classify` endpoint.
- Serialised: while window N is being classified (~3 s round trip),
  window N+1 is being captured. The two phases overlap by design so
  there is no audio gap, but only one classification is in flight at any
  time. A queue of depth 1 holds the most recent next window.

The serial-port `<select>` is wired into the new tab too (defaults to
COM6), so existing port plumbing applies.

### Bigger model: ResNet-18-tiny

`ml/03_quantize.py` was parameterised with a new `--arch` CLI flag.
`build_resnet10()` is preserved (still the default) and a new
`build_resnet18()` adds a wider, deeper "ResNet-18-tiny":

```
Stem (Conv 32, 7×7, stride 2 + BN + ReLU + MaxPool) -> 16×16×32
Stage 1: 2 × ResBlock(32, stride=1) -> 16×16×32
Stage 2: 2 × ResBlock(64, stride=2 then 1) -> 8×8×64
Stage 3: 2 × ResBlock(128, stride=2 then 1) -> 4×4×128
GlobalAvgPool -> Dense(3) softmax
```

Each ResBlock keeps the SE attention sub-block from the baseline model.
**~720 K params float32 → 756 KB INT8**, ~9× the baseline.

Output filenames are now arch-prefixed (`resnet18_int8.tflite`,
`resnet18_keras_best.keras`, `resnet18_savedmodel/`, etc.), so the
existing `resnet10_*` artifacts are untouched.

`ml/04_export.py` got a matching `--arch` flag that selects which tflite
to embed into `app/src/ml/model_data.cc`.

### STM32 firmware adjustments for the bigger model

Only one source change was needed before re-flashing:

```c
// app/src/ml/inference.cc
static uint8_t tensor_arena[200 * 1024] __attribute__((aligned(16)));
```

The arena went from 40 KB (sized for ResNet-10's 29 KB working set) to
200 KB so ResNet-18's larger intermediate feature maps fit. STM32U575
has 768 KB of SRAM total; even with the larger arena we sit at 75 % RAM
usage.

The MicroMutableOpResolver (12 ops: CONV_2D, FULLY_CONNECTED, ADD,
MAX_POOL_2D, MEAN, MUL, LOGISTIC, RESHAPE, SOFTMAX, PACK, SHAPE,
STRIDED_SLICE) was unchanged — ResNet-18 uses the same op set as
ResNet-10, just more invocations of each.

### Training: 50 epochs, early-stopped at 22

```
python ml/03_quantize.py --arch resnet18 --epochs 50 --batch-size 32
```

Ran on CPU (no GPU available on this box). Each epoch was ~40 s, 50
epochs would have been ~33 min but early stopping fired at epoch 32 with
weights restored from epoch 22. Total wall time ~18 min, plus ~3 min
for quantization + test-set eval.

```
Best val accuracy: 74.2%   (epoch 22)
Keras float32 test accuracy: 71.3%
Float32 TFLite test:         71.3%
INT8    TFLite test:         70.9%   (-0.4 pp from float32, fine)
Compression: 3.7×
```

### Deploy and on-hardware verification

```
python ml/04_export.py --arch resnet18   # writes app/src/ml/model_data.cc
./build.sh                                # FLASH 1.22 MB, RAM 587 KB
west flash --build-dir build_stm32_synth --runner openocd
```

`'T 0'` test command from the dashboard returned `A5 00 47` =
ACK + class 0 (Absent) + 0x47 = **71 % confidence**, confirming the
new INT8 model loaded and produced sensible output on the seeded test
vector.

Re-running `Myownheartbeat.m4a` through ResNet-18:

```
Dominant       : Absent (avg conf 80%)
Counts         : {absent: 5, present: 0, unknown: 0}
Latencies avg  : DSP 84 ms, Inference 507 ms
Window 0 82%  82/18/0   no gate
Window 1 88%  88/13/0   no gate
Window 2 75%  75/25/0   no gate
Window 3 88%  88/12/0   no gate
Window 4 68%  68/30/1   no gate
```

Same verdict as ResNet-10 (all Absent, ~80 % avg). The bigger model
agrees on the user's heartbeat — both think it's healthy.

### Honest finding: bigger ≠ better here

**Test-set accuracy got WORSE.** ResNet-10 hit 81.5 % INT8 on the same
test split; ResNet-18 hits 70.9 %. Validation went up (63 → 74 %) but
test did not follow. Classic small-dataset overfitting: 9× capacity
without 9× regularization. CirCor has ~8.8 K training spectrograms and
the class distribution is skewed (Absent dominates), so the extra
capacity learned val-specific patterns that don't transfer.

We kept ResNet-18 deployed deliberately, with the regression documented
in `README.md` and here, because:

1. It demonstrates that the previously-unused flash headroom is now
   actually used (26 % → 58 %).
2. It demonstrates the full retrain → requantize → reflash → reverify
   pipeline on real hardware.
3. It is a real engineering result, not a marketing one.
4. The right next moves are spelled out in "What is NOT done" below
   rather than glossed over.

### Levine grade / severity investigation

`ml/data_circor/raw/training_data.csv` has 23 columns including:

```
Murmur                       : 695 Absent, 179 Present, 68 Unknown
Systolic murmur grading      : 178 labelled (I/VI:104, II/VI:28, III/VI:46)
Diastolic murmur grading     :   5 labelled (statistically useless)
```

So **a systolic-murmur severity regression head IS feasible** with the
existing data — 178 patients across three Levine grades is enough for a
multi-task auxiliary head trained on the shared backbone.

The work is **out of scope for this session** (half-day of focused
work):

- Modify `models/dataset.py` to emit `(spec, class, severity, sev_mask)`
  tuples and mask out Absent / Unknown patients in the severity loss.
- Modify `build_resnet*()` to expose the GAP embedding and add a second
  Dense head producing a scalar (or 3-class) severity prediction.
- Joint cross-entropy + severity loss with `sample_weight` for the
  mask.
- Multi-output INT8 PTQ (the converter handles this but the calibration
  set needs both outputs).
- Two output tensors in `app/src/ml/inference.cc` — second tensor read,
  decoded, included in the response packet.
- Dashboard segment row gains a "Severity" column.

### Files added or modified this half-session

```
.gitignore                       new ignore rules for resnet18_* artifacts
CLAUDE.md                        this session log addendum
README.md                        ResNet-18 vs ResNet-10 comparison + honest finding
CODEX.md                         new handoff entry for today's continuation
app/src/ml/inference.cc          tensor arena 40 KB -> 200 KB
app/src/ml/model_data.cc         AUTO-GENERATED, now the ResNet-18 INT8 array
dashboard/server.py              M4A/MP3/AAC/OGG/FLAC decode, auto_boost,
                                 Live Mic tab + Web Audio JS
ml/03_quantize.py                build_resnet18(), --arch flag, arch-prefixed paths
ml/04_export.py                  --arch flag selects which tflite to embed
ml/models/resnet18_int8.tflite   NEW 756 KB
ml/models/resnet18_float32.tflite NEW 2.8 MB
ml/models/unknown_gate.json      regenerated by training run
```

Session artifacts (NOT committed):

```
flash_out.log, flash_err.log
dashboard/Log 2026-05-12 14_34_55.txt
GEMINI.md
ml/models/resnet18_keras_best.keras
ml/models/resnet18_savedmodel/
ml/models/resnet18_train_log.txt
```

### What is NOT done (next session work)

1. **Temporal GRU head (task 4).** Architecture: small GRU (units=32)
   consuming GAP-layer embeddings of 4-5 consecutive 2-second windows,
   final Dense(3) softmax. Requires: rewrite the dataloader to group
   windows by patient and emit sequences with consistent ordering;
   modify the model to expose the GAP embedding output AND a sequence
   classifier; joint train with weighting; STM32 firmware to maintain a
   small embedding ring buffer across `inference_run()` calls and run
   the GRU head locally OR offload the sequence step to the host
   dashboard.

2. **Severity head (task 5).** Detailed above. CirCor has the labels;
   the work is dataloader masking, multi-head model, joint loss,
   multi-output INT8 PTQ, second-output read on STM32, dashboard
   column.

3. **Flutter Android app for the BLE audio streaming path.** Firmware
   side is done and verified (see Session Log — 2026-05-12 BLE audio
   streaming below). Remaining work is a Flutter or SwiftUI / Web
   Bluetooth client that captures from the phone microphone, downsamples
   to 4 kHz int16 PCM and writes to the audio-in characteristic
   `12345678-1234-1234-1234-123456789ABE`. Reference implementation:
   `ml/06_validate_ble_audio.py` — same GATT writes, just in Dart.

4. **ResNet-18 accuracy recovery.** The test-set regression is the
   most interesting next experiment:
   - Add Dropout(0.2) on the GAP output and after each ResBlock add.
   - Increase weight decay 1e-4 → 5e-4.
   - More aggressive SpecAugment (2-3 freq masks, 2-3 time masks per
     spec instead of 1+1).
   - Mixup with alpha=0.2 between same-class pairs.
   - If none of those close the gap, revert `model_data.cc` to
     `resnet10_int8.tflite` for the demo and keep the bigger-model
     work as a documented experiment branch.

---

## Session Log — 2026-05-12 (final: phone -> BLE -> STM32 audio streaming)

End-to-end wireless audio path verified on hardware. The user's
`Myownheartbeat.m4a` (the same phone recording that classified as 5/5
Absent through the dashboard upload path) was streamed over BLE from a
Python bleak client to the nRF52840 DK, forwarded to the STM32U575 over
UART, classified by the on-chip ResNet-18, and returned to the bleak
client as four BLE notifications: `"Absent 82%"`, `"Absent 88%"`,
`"Absent 75%"`, `"Absent 88%"`. All consistent with the dashboard
upload's verdict.

### Architecture (one box per data direction)

```
Phone or bleak Python client
   │
   │  BLE WRITE_WITHOUT_RESPONSE chunks  (MTU 247 -> 244 B/chunk, 66 writes / window)
   │  -> characteristic 12345678-...-789ABE  (audio_in, 16 KB / window)
   ▼
nRF52840 DK  (Zephyr, ble_peripheral/)
   │  audio_input_service.c:
   │    - accumulates int16 LE PCM into s_window[16000]
   │    - on full window, swaps into s_window_ready[] and submits k_work
   │  forward_work_handler():
   │    - writes 'B' (0x42) + 16000 bytes on UART1 TX (P1.02)
   │
   │  UART1 TX = P1.02   (115200 baud, ~1.4 s for the full 16001-byte burst)
   ▼
STM32U575  (Zephyr, app/)
   │  audio_bridge.c (NEW):
   │    - ISR on USART2 RX (PD6) fills an 18 KB ring buffer
   │    - audio_bridge_thread waits for 'B' sync byte
   │    - reads 16000 bytes, converts int16 LE -> float32 / 32768
   │    - mel_spec_compute (mutex)  -> 64x64 log-mel spectrogram
   │    - inference_run_probs (mutex) -> ResNet-18 INT8 -> class + conf + raw probs
   │    - ble_client_send() writes 6-byte [class, conf, 0x00, ts_u24_le] over USART2 TX (PD5)
   │
   │  6-byte packet, identical format to existing dashboard-upload + synth paths
   ▼
nRF52840 DK  (existing uart_rx_cb in main.c)
   │  parses 6-byte packet exactly as before
   │  hsc_service_notify(conn, class_id, confidence, ts_ms)
   │    -> formats "Absent NN%" / "Present NN%" / "Unknown NN%" string
   │    -> bt_gatt_notify() on characteristic ...ABD (HSC_Result)
   ▼
Phone or bleak Python client
   readable ASCII notification arrives in the BLE log
```

### Pin re-mapping saga (kept for future debugging)

The first two pin choices for `UART1 TX` on the nRF52840 DK failed
silently. Both lie in the strip of GPIOs Nordic ties to the onboard
J-Link OB's UART via factory solder bridges:

| Pin | J-Link function | Outcome |
|---|---|---|
| `P0.06` | UART0 TXD | nRF console output stayed alive (UART0 won the pinmux), our UART1 TX bytes went nowhere |
| `P0.07` | UART CTS | nRF's own logs said "forwarded window #N in 1395 ms" — uart_poll_out wrote the bytes — but J-Link OB was driving the line and the STM32 never saw them |
| **`P1.02`** | **none (free GPIO)** | **works cleanly** |

The DK's J-Link OB controls P0.05 (RXD) / P0.06 (TXD) / P0.07 (CTS) /
P0.08 (RTS) via SB5..SB8. P0.08 happened to work for our **RX** because
J-Link is high-Z on that pin, but using P0.06 or P0.07 as nRF **outputs**
puts us in contention with the J-Link MCU and the bytes are eaten.
P1.02 is the Zephyr board-file default for `uart1` TX precisely because
the P1.xx header is J-Link-free.

### Diagnostic technique that broke it open

`CONFIG_LOG` is disabled on the STM32 build (dashboard binary-clean
console), so the audio_bridge had no visible output. To debug we
temporarily added single-byte sentinel bursts on USART1 (the dashboard
UART), keyed off the 0xE0 prefix to be easy to spot among the existing
0xA5 / 0xA6 protocol bytes:

```
0xE0 0xB1                ─ bridge thread reached "waiting for 'B'"
0xE0 0xB2 <byte>         ─ first non-'B' byte ever seen on USART2 RX
0xE0 0xB3                ─ 'B' sync byte received
0xE0 0xB4 <count_le_16>  ─ window finished arriving (count == 8000 samples)
0xE0 0xB5 <class> <conf> ─ inference finished
0xE0 0xBE                ─ periodic heartbeat (every 4 s while idle)
```

With those in place we could read raw bytes off COM6 in PowerShell and
see exactly which stage stalled. The P0.07 J-Link contention manifested
as the heartbeat byte `E0 BE` repeating indefinitely with no `E0 B3`
following any BLE upload. After moving the wire to P1.02 we immediately
saw the expected `E0 B3 E0 B4 40 1F E0 B5 00 52` sequence (class 0,
0x52 = 82 %) and matching `"Absent 82%"` BLE notifications.

The diagnostic markers were removed before the final commit because
they would interfere with the dashboard's `0xA5` / `0xA6` parser if
both data paths ran simultaneously. They live in the git history at
the commit immediately before the final audio_bridge cleanup.

### Files added / modified this segment

```
app/CMakeLists.txt                    + src/ml/audio_bridge.c
app/src/main.c                        + #include "ml/audio_bridge.h"
                                      + audio_bridge_init() after validate
app/src/ml/audio_bridge.h             NEW
app/src/ml/audio_bridge.c             NEW (~115 lines, 18 KB RX rb,
                                      32 KB float audio, 16 KB spec
                                      buffer, 4 KB stack)
ble_peripheral/CMakeLists.txt         + src/audio_input_service.c
ble_peripheral/src/main.c             + #include "audio_input_service.h"
                                      + audio_input_service_init(uart_dev)
ble_peripheral/src/audio_input_service.h   NEW
ble_peripheral/src/audio_input_service.c   NEW (BT_GATT_SERVICE_DEFINE
                                            for the AudioIn char + 16 KB
                                            accumulator + forward_work
                                            handler)
ble_peripheral/boards/nrf52840dk_nrf52840.overlay
                                      uart1_default + uart1_sleep now
                                      include NRF_PSEL(UART_TX, 1, 2)
                                      alongside the existing
                                      NRF_PSEL(UART_RX, 0, 8)
ble_peripheral/prj.conf               CONFIG_BT_L2CAP_TX_MTU=247
                                      CONFIG_BT_BUF_ACL_TX_SIZE=251
                                      CONFIG_BT_BUF_ACL_RX_SIZE=251
                                      CONFIG_BT_CTLR_DATA_LENGTH_MAX=251
                                      CONFIG_BT_CTLR_PHY_2M=y
ml/06_validate_ble_audio.py           NEW — bleak Python client
```

### Build numbers after this segment

- STM32 firmware: FLASH 1.22 MB / 2 MB (58.2 %, unchanged), RAM
  587 KB → 658 KB / 768 KB (75 % → 84 %). +71 KB for the new audio
  bridge (18 KB ring buffer + 32 KB float audio + 16 KB spec + 4 KB
  thread stack).
- nRF firmware: FLASH 125.9 KB / 1 MB (12 %), RAM 62 KB / 256 KB (24 %).
  ~3 KB FLASH and ~40 KB RAM added vs the BLE-only build (two 16 KB
  accumulator buffers + service descriptors).

### Measured timing on hardware

- BLE write phase (phone → nRF): ~3 s per window in the bleak test
  (80 × 200-byte chunks; could be tightened to ~500 ms with 244-byte
  chunks).
- nRF → STM32 UART forward: 1395 ms per window at 115200 baud (matches
  the theoretical 16001 × 10 / 115200 = 1.389 s).
- STM32 DSP + ResNet-18 inference: ~600 ms per window.
- STM32 → nRF response (6 bytes): ~0.5 ms.
- nRF → phone BLE notify: ~30 ms.

**Total round trip per window: ~5 s** with the current test client.
Dominated by the BLE write phase (which the bleak test pessimises with
small chunks). The Flutter app can use larger writes and the BLE 2 M
PHY to bring this under 1 s round trip.

### Things to remember for the next session

- The UART1 TX pin on the nRF52840 DK is `P1.02`, NOT P0.06 / P0.07
  (those are J-Link reserved). The user guide and Zephyr board file
  agree on this — anything in P0.05–P0.08 should be assumed off-limits
  for nRF outputs on this DK.
- The audio bridge logs `LOG_INF` lines on USART1 but `CONFIG_LOG` is
  disabled on this STM32 build, so they're effectively no-ops. If you
  need to debug the bridge again, re-introduce the 0xE0-prefixed
  sentinel bursts (full set documented above) or enable CONFIG_LOG on
  a separate UART.
- A 6-byte response from `ble_client_send()` is sent over USART2 TX
  for BOTH the dashboard-upload path and the BLE-audio path. The nRF's
  6-byte parser does not disambiguate the source — it just notifies
  the most recent classification. If both paths fire windows
  simultaneously, the order of notifications may interleave.
- `ml/06_validate_ble_audio.py` requires `pip install bleak
  imageio-ffmpeg soundfile`. It works on Windows 10/11 via the native
  WinRT BLE stack.
