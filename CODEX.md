# Digital Stethoscope (v2 / CirCor 2022) — Session Handoff

Last updated: 2026-05-12 (Claude session, Web Bluetooth phone client live + BLE auto-recover)

## End-of-day (2026-05-12, fourth handoff) — Web Bluetooth phone client + BLE fixes

After the bleak validator proved the firmware path works end-to-end, the
project shipped a phone-side client and hardened the BLE stack against
the common failure modes that came up in real-world use.

### What landed since the third handoff (the BLE-audio-streaming one)

1. **Web Bluetooth phone client (`docs/index.html`)**: single 16 KB
   HTML page that, opened in Chrome on Android, captures phone mic →
   downsamples to 4 kHz → connects to nRF52840 via Web Bluetooth →
   streams int16 PCM windows to the audio-in characteristic →
   subscribes to the classification characteristic and renders each
   inbound ASCII notification as a big "Absent NN%" card. Deployed via
   GitHub Pages from `docs/` (HTTPS required for Web Bluetooth, GH
   Pages provides it). Local desktop testing is `python -m http.server
   -d docs 8080`; phones need the GH Pages URL.

2. **nRF auto-restarts advertising on disconnect**. Previous firmware
   called `bt_le_adv_start()` synchronously in the disconnect callback,
   which sometimes lost the race with the controller's connection
   teardown and left the peripheral silent until a manual J-Link reset.
   New behaviour: a `k_work_delayable adv_work` retries every 500 ms on
   `-ENOMEM` / `-EINVAL`, no-ops on `-EALREADY`, and is also used for
   the initial boot-time advertising start.

3. **Merged BLE characteristics into one GATT service.** The firmware
   used to have TWO `BT_GATT_SERVICE_DEFINE` blocks (one per
   characteristic) both declaring the same service UUID `...ABC`. BLE
   treats these as two distinct primary services. Web Bluetooth's
   `getPrimaryService(uuid)` returns only the FIRST, so the audio-in
   characteristic in the second one was unreachable. bleak worked
   because its characteristic lookup walks all services. Fix: single
   `BT_GATT_SERVICE_DEFINE` in `heart_sound_service.c` declares both
   characteristics. `audio_input_service.c` keeps the accumulator +
   forward worker logic and exposes the write handler via its header.

### Verification done after the fixes

- `bleak` introspection on the live nRF reports BOTH `...ABD` and
  `...ABE` characteristics under the same primary service `...ABC`
  instance.
- bleak end-to-end audio streaming on the user's `Myownheartbeat.m4a`
  still returns the same four BLE notifications: `Absent 82%`,
  `Absent 88%`, `Absent 75%`, `Absent 88%`.
- bleak disconnect → 2 s wait → scan → `HeartSound` rediscovered with no
  J-Link reset, confirming the auto-restart loop.

### nRF / STM32 footprint after this segment

- nRF: FLASH 126.0 KB / 1 MB (12.0 %), RAM 62 KB / 256 KB (24 %).
- STM32 unchanged from the BLE-audio segment: FLASH 1.22 MB / 2 MB
  (58.2 %), RAM 658 KB / 768 KB (83.7 %).

### Two operational caveats that users (and future-me) need to know

1. **BLE peripherals are 1-to-1.** If nRF Connect (or any other BLE
   central) is connected to `HeartSound`, the Web Bluetooth chooser
   cannot find it — the device is not advertising while connected. The
   page now displays a callout under the Connect button.

2. **Android caches GATT service definitions across reconnects.** After
   the merged-service reflash, the cached service list on the phone
   still shows the broken split version, and Web Bluetooth's strict
   per-service characteristic search fails. Fix: in Android *Settings →
   Bluetooth*, tap the gear icon next to `HeartSound` and choose
   *Forget device*. Reload the web page and reconnect — Android does a
   fresh service discovery.

## End-of-day (2026-05-12, third handoff) — BLE audio streaming end-to-end

The wireless audio path is now fully verified on hardware:

```
Phone / bleak client  --BLE WRITE_WITHOUT_RESPONSE-->  nRF52840 DK
nRF52840 DK           --UART 'B' + 16 KB PCM ------>  STM32U575
STM32U575             --on-chip DSP + INT8 ResNet -->  classification
STM32U575             --UART 6-byte response ------>  nRF52840 DK
nRF52840 DK           --BLE notify "Absent NN%" --->  Phone / bleak client
```

The user's `Myownheartbeat.m4a` was streamed through this path and the
four BLE notifications matched the dashboard upload's verdict exactly:
`Absent 82%`, `Absent 88%`, `Absent 75%`, `Absent 88%`.

### Wiring (final, 3 wires)

| STM32 NUCLEO-U575ZI-Q | nRF52840 DK | Direction |
|---|---|---|
| PD5 (USART2 TX, D53) | P0.08 (UART1 RX) | STM32 -> nRF, classifications |
| PD6 (USART2 RX, D52) | P1.02 (UART1 TX) | nRF -> STM32, BLE-streamed audio |
| GND | GND | common ground (required) |

### Why nRF UART1 TX must be on P1.02

The nRF52840 DK ties P0.05/P0.06/P0.07/P0.08 to the onboard J-Link OB's
UART (RXD/TXD/CTS/RTS) via factory-intact solder bridges SB5..SB8. The
JLink MCU actively drives those lines, so using any of them as nRF
outputs results in contention that silently swallows bytes. P0.08 happens
to work as an INPUT because J-Link is high-Z on its RXD-line, but using
P0.06 or P0.07 as an OUTPUT does not. P1.02 is the Zephyr board-file
default for `uart1` TX precisely because the P1.xx header is J-Link-free.

### Firmware additions

```
app/src/ml/audio_bridge.c, .h            NEW — STM32 USART2 RX listener
                                         that runs the 'B' + 16 KB
                                         protocol, runs DSP + inference
                                         under existing mutexes, replies
                                         via ble_client_send()
app/src/main.c                           + audio_bridge_init() after
                                         validate_thread create
app/CMakeLists.txt                       + src/ml/audio_bridge.c
ble_peripheral/src/audio_input_service.c, .h
                                         NEW — adds AudioIn GATT char
                                         at UUID ...ABE on the existing
                                         HSC service. 16 KB accumulator,
                                         double-buffered handoff to a
                                         k_work that drains over UART
ble_peripheral/src/main.c                + audio_input_service_init(uart_dev)
ble_peripheral/CMakeLists.txt            + src/audio_input_service.c
ble_peripheral/boards/nrf52840dk_nrf52840.overlay
                                         uart1_default and uart1_sleep
                                         now include NRF_PSEL(UART_TX, 1, 2)
                                         alongside NRF_PSEL(UART_RX, 0, 8)
ble_peripheral/prj.conf                  CONFIG_BT_L2CAP_TX_MTU=247,
                                         CONFIG_BT_BUF_ACL_TX_SIZE=251,
                                         CONFIG_BT_BUF_ACL_RX_SIZE=251,
                                         CONFIG_BT_CTLR_DATA_LENGTH_MAX=251,
                                         CONFIG_BT_CTLR_PHY_2M=y
                                         (Data Length Extension + 2M PHY
                                         so the phone can write big
                                         chunks fast)
ml/06_validate_ble_audio.py              NEW — bleak-based Python BLE
                                         audio sender. Decodes any
                                         ffmpeg-supported file, scans
                                         for "HeartSound", negotiates
                                         MTU, streams in MTU-sized
                                         WRITE_WITHOUT_RESPONSE chunks,
                                         prints inbound notifications
```

### STM32 footprint after this segment

FLASH 1.22 MB / 2 MB (58.2 %, unchanged from the ResNet-18 deploy).
RAM   658 KB / 768 KB (83.7 %, +71 KB for the bridge thread).

### nRF footprint

FLASH 125.9 KB / 1 MB (12.0 %, +3 KB).
RAM   62 KB / 256 KB (24 %, +40 KB for the two 16 KB accumulators).

### Round-trip timing measured today

- BLE write (phone -> nRF): ~3 s per window (bleak test uses 200-byte
  chunks; Flutter / Web Bluetooth could halve this with full MTU 244
  chunks)
- UART forward (nRF -> STM32): 1395 ms per window at 115200 baud
  (matches the theoretical 16001 * 10 / 115200)
- STM32 DSP + ResNet-18 inference: ~600 ms / window
- STM32 -> nRF response (6 bytes): ~0.5 ms
- nRF BLE notify -> phone: ~30 ms

Total ~5 s round trip per 2-second window with this test client.

### Diagnostic technique (kept here for the next time we debug a silent
firmware path)

`CONFIG_LOG` is off on the STM32 to keep the dashboard UART
binary-clean. To debug the audio bridge we temporarily emitted
0xE0-prefixed sentinel bursts on USART1:

```
0xE0 0xB1                — bridge thread reached "waiting for 'B'"
0xE0 0xB2 <byte>         — first non-'B' byte seen on USART2 RX
0xE0 0xB3                — 'B' sync byte received
0xE0 0xB4 <count_le_16>  — window finished arriving (count == 8000)
0xE0 0xB5 <class> <conf> — inference finished
0xE0 0xBE                — periodic heartbeat (every 4 s while idle)
```

This unblocked the P0.07-vs-P1.02 pin diagnosis (only `E0 BE`
heartbeats kept arriving — bridge alive, no bytes flowing in). The
markers were removed before the final commit because they would
interfere with the dashboard's 0xA5 / 0xA6 parser if both data paths
ran simultaneously. They live in commit history right before the
final audio_bridge cleanup.

## What is NOT done

1. **Flutter Android app for the audio streaming path.** Firmware is
   verified working with the Python bleak validator; the remaining
   piece is a phone app that:
   - Captures from the device microphone via `record` or `flutter_sound`
   - Downsamples to 4 kHz mono int16 PCM
   - Connects to `HeartSound` and discovers service
     `12345678-1234-1234-1234-123456789ABC`
   - Subscribes to characteristic `...ABD` for classification
     notifications
   - Writes PCM windows to characteristic `...ABE` in MTU-sized chunks
     via WRITE_WITHOUT_RESPONSE
   - Reference Dart-equivalent of `ml/06_validate_ble_audio.py`

2. **Temporal GRU head (task 4).** Multi-day rewrite of dataloader +
   model + STM32 inference loop.

3. **Severity (Levine grade) head (task 5).** Half-day; data confirmed
   present in `training_data.csv`.

4. **ResNet-18 accuracy recovery.** Dropout, mixup, stronger
   SpecAugment.

---

## Earlier today (third-to-last handoff) — M4A, live mic, ResNet-18 deploy

(Original entry; trimmed to keep this file readable. See CLAUDE.md
"Session Log — 2026-05-12 (continued)" for full detail.)

## Project state at end of session (2026-05-12, second handoff)

Three big things landed on top of the morning's BLE integration:

1. **Phone-recorded M4A → classification.** The dashboard now decodes
   any ffmpeg-supported audio format (`.m4a`, `.mp3`, `.aac`, `.ogg`,
   `.flac`, `.webm`, `.opus`, `.mp4`, `.m4b`, …) via a bundled
   `imageio-ffmpeg` binary. Quiet phone recordings get a soft
   peak-normalisation (`auto_boost`, kicks in only when peak < 0.5).
2. **Live laptop microphone mode.** New tab next to Custom Upload /
   Generic Loop. Browser Web Audio API captures from the default mic,
   downsamples to 4 kHz, ships each 2-second window to the STM32 in
   real time. No server changes beyond the tab markup.
3. **ResNet-18-tiny deployed to STM32U575.** Bigger CNN (720 K params,
   756 KB INT8, ~507 ms inference) replacing the 81 K-param ResNet-10
   baseline. Demonstrates that the previously-unused 1.45 MB flash and
   345 KB RAM headroom are now being used (FLASH 58 %, RAM 75 %).

Plus a real engineering finding kept visible in the docs:
**ResNet-18's test accuracy is 10 pp LOWER than ResNet-10's** on the same
CirCor test split (70.9 % vs 81.5 %). Classic small-dataset overfitting.
The deploy is intentionally kept as-is, with the accuracy-recovery work
(dropout, mixup, stronger SpecAugment) listed as a next-session
experiment rather than papered over.

Also: Levine grade severity-head data was investigated. CirCor 2022 has
178 patients with systolic Levine grades (I/VI / II/VI / III/VI) in
`training_data.csv`. Severity regression is feasible; implementation is
half-day of focused work, deferred to next session.

## What changed this half-session

### Dashboard: M4A + auto-boost

`dashboard/server.py` (committed `b8b844e`):

- `_get_ffmpeg()` resolves the bundled `imageio_ffmpeg` binary
  (`ffmpeg-win-x86_64-v7.1.exe`), with `shutil.which("ffmpeg")` as
  fallback.
- `load_compressed(path)` shells out to ffmpeg for `-f wav -acodec
  pcm_s16le -ac 1 -ar 4000 -` and reads the resulting WAV from
  `proc.stdout` via `soundfile.read(io.BytesIO(...))`.
- `auto_boost(audio, target_peak=0.95, min_peak_threshold=0.5)` scales
  to 0.95 peak only when the input is below half scale, so loud
  recordings pass through unchanged.
- `COMPRESSED_SUFFIXES` set drives the `preprocess_input_file()`
  dispatch.
- File-input `accept` attribute and the "choose a file" error message
  updated to advertise the broader format list.

### Dashboard: Live Mic tab

`dashboard/server.py` (committed `344a0bb`):

- Third tab `Live Mic` next to Custom Upload and Generic Loop.
- `setMode(mode)` handles three modes instead of two.
- `startLiveMic()` requests mic permission via
  `navigator.mediaDevices.getUserMedia({audio: {channelCount: 1,
  echoCancellation: false, noiseSuppression: false,
  autoGainControl: false}})`.
- `AudioContext` + deprecated-but-stable `ScriptProcessorNode(4096)`
  feeds 4096-sample chunks at the device's native rate (44.1 or 48 kHz).
- Buffer 2 seconds, downsample to 4 kHz via `downsampleLinear`, convert
  to int16 via `floatToInt16`, build a 44-byte-header WAV in
  `buildWavBlob`, POST as `FormData` to existing `/api/classify`.
- Serialised: classification is in flight while next window is being
  captured. Queue depth 1 holds the most recent next window.
- `liveSerialPort` `<select>` shares the same port-list loader as the
  other two tabs.

### ML pipeline: `--arch` flag

`ml/03_quantize.py` (committed `344a0bb`):

- `build_resnet18(use_se=True)` adds the bigger variant: 3 stages with
  2 ResBlocks each, widths 32/64/128, ~720 K params float32 → ~700 KB
  INT8 raw.
- `build_model(arch, use_se)` dispatches on `arch in {resnet10,
  resnet18}`.
- Output files arch-prefixed: `resnet18_int8.tflite`,
  `resnet18_keras_best.keras`, `resnet18_savedmodel/`, etc.
- `--arch` CLI flag defaults to `resnet10` for backwards compatibility.

`ml/04_export.py` (committed `344a0bb`):

- Matching `--arch` flag picks which `.tflite` to embed into
  `app/src/ml/model_data.cc`.

`.gitignore` (committed `a7aa831`):

- Added `ml/models/*_keras_best.keras`,
  `ml/models/resnet18_savedmodel/`, `ml/models/*_train_log.txt`.

### STM32 firmware: tensor arena bump

`app/src/ml/inference.cc` (committed `344a0bb`):

- `tensor_arena[40 * 1024]` → `tensor_arena[200 * 1024]` so the bigger
  intermediate feature maps fit. Op resolver unchanged (same 12 ops).

### Training command and outcome

```
python ml/03_quantize.py --arch resnet18 --epochs 50 --batch-size 32
```

CPU-only training, ~40 s per epoch. Early-stopped at epoch 32 with
weights restored from epoch 22.

```
Best val_acc           : 74.2%   (vs ResNet-10's 63.2%)
Keras float32 test_acc : 71.3%
Float32 TFLite test    : 71.3%
INT8 TFLite test       : 70.9%   (-0.4 pp from float32)
Compression            : 3.7x
INT8 model size        : 756 KB
```

### Deploy and verification

```
python ml/04_export.py --arch resnet18   # -> app/src/ml/model_data.cc (756 KB array)
./build.sh                               # FLASH 1.22 MB / 2 MB (58.2 %), RAM 587 KB / 768 KB (74.7 %)
west flash --build-dir build_stm32_synth --runner openocd
```

`'T 0'` test from the dashboard → `A5 00 47` = ACK + Absent + 71 %.

Re-running the user's `Myownheartbeat.m4a` (9.54 s, M4A) end-to-end
through dashboard → STM32 ResNet-18 → BLE:

```
Dominant      : Absent (avg conf 80%)
Counts        : {absent: 5, present: 0, unknown: 0}
Latencies avg : DSP 84 ms, Inference 507 ms
Window 0 82%  82/18/0
Window 1 88%  88/13/0
Window 2 75%  75/25/0
Window 3 88%  88/12/0
Window 4 68%  68/30/1
```

Same verdict as ResNet-10 — both models agree the user's heartbeat is
healthy.

### Levine grade investigation (not committed; documented only)

`ml/data_circor/raw/training_data.csv` columns include `Murmur`,
`Systolic murmur grading` (178 labelled), `Diastolic murmur grading` (5
labelled). Systolic distribution: I/VI 104 patients, II/VI 28, III/VI
46. Enough for a multi-task auxiliary head trained jointly on the
shared backbone. Full implementation is half-day:
dataloader masking, multi-head model, joint loss, multi-output INT8
PTQ, second-output read on STM32, dashboard column.

## Final commit timeline

```
b8b844e Accept M4A/MP3/AAC/OGG/FLAC uploads via bundled ffmpeg
344a0bb Scaffold ResNet-18, live-mic dashboard tab, prep STM32 for bigger model
a7aa831 Deploy ResNet-18 to STM32: 720 K params, 756 KB INT8, 507 ms inference
(this) Update CLAUDE.md / README.md / CODEX.md with end-of-day handoff
```

All on `origin/master`. Branch is currently `ahead 0` of remote (or 1
if you count this doc commit).

## Quick start for the next session

1. `git status --short --branch` — verify clean / aligned with remote.
2. `cd C:/Users/jaiva/Desktop/Internship_Files/STM_Project/Digital_Stethoscope_2`.
3. `./build.sh` — verify STM32 builds (should land at ~1.22 MB FLASH).
4. `west flash --build-dir build_stm32_synth --runner openocd`.
5. nRF: `west flash --build-dir build_ble --runner jlink` (skip if not
   touching BLE).
6. `py dashboard\server.py --port COM6 --baud 115200`, open
   `http://127.0.0.1:8765`.
7. Pick one of the deferred tasks (recommend ResNet-18 accuracy
   recovery first, then BLE audio streaming).

## What is NOT done (handoff)

1. **ResNet-18 accuracy recovery.** Add Dropout(0.2) on GAP output and
   after each ResBlock add. Increase weight decay 1e-4 → 5e-4. More
   aggressive SpecAugment (2-3 freq masks, 2-3 time masks per spec
   instead of 1+1). Optional mixup. If gap doesn't close, revert
   `model_data.cc` to `resnet10_int8.tflite`.
2. **Phone → BLE → STM32 audio streaming (task 6 from upgrade list).**
   Add wire nRF P0.06 (UART1 TX) → STM32 PD6 (USART2 RX). nRF needs
   new WRITE-without-response GATT characteristic with 16 KB
   accumulator + forward to STM32 over UART using the existing
   `'A'`-protocol. STM32 returns extended response over existing
   PD5→nRF P0.08 wire; nRF parses class + confidence; BLE-notifies
   back. Phone-side sender needs to be a bleak Python script or
   Flutter / SwiftUI app.
3. **Severity head (task 5).** Detailed above; implementation
   deferred.
4. **Temporal GRU head (task 4).** Sequence dataloader, GRU over GAP
   embeddings of 4-5 consecutive windows, STM32 embedding ring buffer.
   Multi-day rewrite.

---

## Earlier this session (morning) — nRF52840 BLE integration

End-to-end demo path completed in the morning slot of this session:

```
WAV  ─dashboard upload──▶  STM32U575  ─USART2 PD5─▶  nRF52840 DK  ─BLE─▶  Phone
        (browser, HTTP)    DSP + INT8 ML            ASCII notify          nRF Connect
```

Every WAV uploaded to the dashboard now fans out to the phone over BLE in
addition to the dashboard's scientific multi-segment view. A 12-second
presentation sample produces six BLE notifications on the phone, each
rendered as readable text (`"Present 95%"`, `"Absent 91%"`, `"Unknown 14%"`,
or `"Error 0%"`).

Verified end-to-end on hardware in the morning: STM32U575 NUCLEO-U575ZI-Q +
nRF52840 DK + Android phone running nRF Connect, uploading
`presentation_samples/02_present_pid9979.wav` from the local dashboard.

## What changed in the morning slot

### 1. UART bridge re-pinned: USART3 → USART2 (PD5 / NUCLEO D53)

The original spec used STM32 USART3 (PD8/PD9). On the NUCLEO-U575ZI-Q those
pins exist on the morpho header but are NOT labelled "USART3" anywhere on
the silkscreen or quick-reference. The pin labelled `USART_B_TX` (Arduino
header position D53) is **PD5**, which is USART2_TX on the STM32U575.
Switching to USART2 made the wiring obvious.

Two wires only between the boards:

```
STM32 PD5 (USART2 TX, D53)  ──▶  nRF P0.08 (UART1 RX)
STM32 GND                   ──── nRF GND
```

A common GND wire is **required** even when both boards share a PC USB
host — USB shield grounding proved unreliable for UART signal levels.

Firmware changes:
- `app/app.overlay` — `&usart3` block replaced with `&usart2` (using
  `usart2_tx_pd5 usart2_rx_pd6` pinctrl entries, both AF7).
- `app/src/comms/ble_client.c` — `DT_NODELABEL(usart3)` → `usart2`.

### 2. Dashboard-upload (`'A'`) path now broadcasts to BLE

Previously only the synthetic-injection pipeline reached
`ble_client_send()`. The `'A'`-command handler in `validate.c` ran on-chip
inference and returned the extended response packet to the dashboard but
did not fan out to the nRF.

```c
// app/src/ml/validate.c, after the on-chip inference:
uint8_t out_class = (class_id < 0) ? 0xFFu : (uint8_t)class_id;
ble_client_send(out_class, confidence, k_uptime_get_32());   // NEW
```

### 3. nRF stale 4-class labels replaced

`ble_peripheral/src/main.c` was still using the pre-CirCor
`{"Normal","SysMurmur","DiaMurmur","S3Gallop"}` table for its UART-RX
trace, so class 2 was being printed as `"DiaMurmur"` on the nRF console
even though the on-chip model treats it as `"Unknown"`. Replaced with
`{"Absent","Present","Unknown"}` and bound-checked via `ARRAY_SIZE`.

### 4. BLE payload changed to printable ASCII + CUD descriptor

First end-to-end test produced notifications like `(0x) 01-5F-00-93-11-03`
in nRF Connect — correct but unreadable for a live demo. Rewrote
`hsc_service_notify()` in
`ble_peripheral/src/heart_sound_service.c` to send a short printable
string (up to 20 bytes) instead:

```c
char pkt[20];
int len = snprintf(pkt, sizeof(pkt), "%s %u%%", name, confidence);
bt_gatt_notify(...)   // ASCII, no NUL
```

Also added a Characteristic User Description (CUD) descriptor so the
characteristic advertises a human-readable name:

```c
BT_GATT_CUD("Heart Sound Classification", BT_GATT_PERM_READ),
```

After this change Android nRF Connect needs a single disconnect / reconnect
cycle to drop its cached GATT structure and pick up the new descriptor.

### 5. nRF flashing notes

- Default `west flash` runner is `nrfutil`, but Nordic's wrapper is broken
  on Python 3.14 (`ModuleNotFoundError: No module named 'constants'`). Use
  `west flash --build-dir build_ble --runner jlink` instead. The DK's
  onboard J-Link is detected as SEGGER 1366:1051.
- DK exposes **two** JLink CDC UART ports. On this machine the Zephyr
  console UART appears on **COM8**, not COM7.
- Fastest reset (no power-cycle) for clearing a stuck "connected" state:
  J-Link Commander script `r; g; exit` against `nRF52840_xxAA`.

### 6. Sizes after this session

- STM32 firmware: FLASH 550500 B / 2 MB (26.25%), RAM 423276 B / 768 KB
  (53.82%). No measurable change from yesterday — the new
  `ble_client_send()` call inlines into existing code path.
- nRF firmware: FLASH 122124 B / 1 MB (11.65%), RAM 22534 B / 256 KB
  (8.60%).

## Files modified this session

```
app/app.overlay
app/src/comms/ble_client.c
app/src/ml/validate.c
ble_peripheral/src/main.c
ble_peripheral/src/heart_sound_service.c
CLAUDE.md
README.md
CODEX.md     (this file)
```

Session artifacts NOT to commit (kept in working tree only):
- `flash_out.log`, `flash_err.log` (PowerShell capture from west flash)
- `dashboard/Log 2026-05-12 14_34_55.txt` (downloaded nRF Connect log)

## What still isn't done

- `tests/` and `ml/05_validate_on_device.py` still use the `'T'`
  test-vector command (unchanged since 2026-05-11). They do not exercise
  the new `'A'` upload path or the BLE fan-out.
- No retraining or model changes this session.
- No BLE security configured (bond / pairing / passkey). Anyone in range
  can subscribe. Acceptable for prototype demo; revisit before any
  clinical handling.

---

## Previous handoff — 2026-05-11 (UART upload + scientific dashboard)

The STM32U575 firmware was rewritten so the laptop can stream **real audio**
through the dashboard. The STM32 receives raw PCM over UART, computes the
mel spectrogram on chip, runs the CirCor 2022 INT8 ResNet-10 + calibrated
unknown gate, and returns a single extended response packet containing the
class, raw 3-class probabilities, per-stage latency, audio statistics, and
the full 64x64 spectrogram. The dashboard is a thin web client that renders
a multi-segment scientific view.

End-to-end verified on hardware this session.

## Repository

- Branch: `master`
- Last upstream commit: `53fd335 Document final ML handoff`
- Working tree has uncommitted changes for the new upload protocol,
  dashboard rewrite, mixed-demo audio generator, presentation samples,
  and CODEX.md / CLAUDE.md / README.md updates.

## UART protocol (current)

```
Host -> STM32 : 'T' (0x54) + uint16 LE index
STM32 -> Host : 0xA5 + class_id + confidence                      (3 B)

Host -> STM32 : 'A' (0x41) + 8000 LE int16 PCM samples
                (delivered in 128-sample / 256-byte chunks)
STM32 -> Host : 0xA6 after every 128 samples received + one final 0xA6
STM32 -> Host : 0xA5 + class_id + confidence
              + raw probabilities[3]  (Absent, Present, Unknown - percent)
              + gate_applied          (uint8)
              + dsp_ms                (uint16 LE)
              + inference_ms          (uint16 LE)
              + rms_q15               (uint16 LE,  rms  * 32768)
              + peak_q15              (uint16 LE,  peak * 32768)
              + zcr                   (uint16 LE,  zero-crossing count)
              + 4096 LE float32 mel values                      (16401 B total)

Host -> STM32 : 'S' / 'P' to start/pause the synthetic PCG demo loop
STM32 -> Host : 0xA6 + command + state                            (3 B)
```

`CONFIG_LOG` is **disabled** in `app/prj.conf` so no log text can interleave
with the binary payload on the same UART.

The 128-sample host chunking matches the STM32 inline-ACK cadence: with 8000
samples, Python's `range(0, 8000, 128)` yields 63 iterations, which lines up
with 62 inline ACKs + 1 post-loop ACK = 63 ACKs total. One ACK per host
chunk write.

## Files changed this session

```
app/prj.conf                       CONFIG_LOG removed, CONFIG_RING_BUFFER_LARGE=y added
app/src/dsp/mel_spec.h             new g_mel_spec_mutex prototype
app/src/dsp/mel_spec.c             mutex definition
app/src/main.c                     dsp_thread now holds g_mel_spec_mutex
app/src/ml/inference.h             new inference_run_probs() prototype
app/src/ml/inference.cc            shared run_inference_core(); raw probs + gate flag
app/src/ml/validate.h              extended response format doc
app/src/ml/validate.c              new 'A' audio-upload command, audio stats, ext. response

dashboard/server.py                full rewrite: file upload + segmentation + ML charts
ml/generate_mixed_demo.py          NEW - builds mixed_demo.wav from CirCor labels
ml/data_circor/demo/
  mixed_demo.wav                   NEW - 18 s, 9 segments (3 per class)
  mixed_demo_labels.json           NEW - per-segment ground truth

presentation_samples/              NEW - 3 single-class WAVs for live demos
  01_absent_pid49653.wav
  02_present_pid9979.wav
  03_unknown_pid9983.wav
  *_labels.json                    sidecar ground truth (auto-loaded by dashboard)
  __build__.py                     idempotent regenerator
  README.md
```

## Build / flash status

Clean firmware build (`./build.sh --clean`):

```
FLASH: 550 KB / 2 MB   (26.3%)
RAM  : 423 KB / 768 KB (53.8%)
```

Growth vs the previous CirCor build (~430 KB / ~336 KB) is dominated by:

- 40 KB UART RX ring buffer (`s_rx_rb`) needed to absorb the 16 KB audio upload
- 32 KB `s_uploaded_audio` (float32 mirror of the 8000-sample window)
- 16 KB `s_uploaded_spec` (float32 64x64 spectrogram)

All in `.bss`; no heap usage. Tensor arena is still 40 KB and independent.

Flash command (used today):

```
west flash --build-dir build_stm32_synth --runner openocd
```

OpenOCD wrote `550656 B from build_stm32_synth/zephyr/zephyr.hex in 5.3 s`.

## End-to-end hardware measurements (today)

### mixed_demo.wav (18 s, 9 segments)

```
counts             = {'absent': 6, 'present': 1, 'unknown': 2}
accuracy_pct       = 55.6     (5 / 9, expected for a tiny mixed set)
avg DSP / Infer    = 84.8 ms / 102.0 ms per window
avg UART up/return = 1416 ms / 1423 ms     (limited by 115200 baud; 16 KB each way)
```

Per-segment trace (`+` correct, `-` incorrect):

```
+  0   0.0 s  Absent (68%)  raw=[68,11,21]               gt=Absent
+  1   2.0 s  Absent (86%)  raw=[86, 7, 7]               gt=Absent
+  2   4.0 s  Absent (80%)  raw=[80,14, 5]               gt=Absent
-  3   6.0 s  Absent (75%)  raw=[75,11,14]               gt=Present
-  4   8.0 s  Unknown(24%)  raw=[57,18,24]   gate fired  gt=Present
+  5  10.0 s  Present(89%)  raw=[ 8,89, 2]               gt=Present
-  6  12.0 s  Absent (78%)  raw=[78, 8,14]               gt=Unknown
-  7  14.0 s  Absent (75%)  raw=[75, 9,16]               gt=Unknown
+  8  16.0 s  Unknown(13%)  raw=[56,31,13]   gate fired  gt=Unknown
```

### 02_present_pid9979.wav (12 s, 6 segments)

```
accuracy_pct = 100.0     (6 / 6 correct)
predictions  = Present at 59 / 51 / 72 / 95 / 88 / 73 %
```

Strong classification on a high-grade (III/VI) holosystolic murmur recording.

## Dashboard

Run command (already happens on the user's box):

```powershell
C:\Users\jaiva\AppData\Local\Programs\Python\Python311\python.exe dashboard\server.py --port COM6 --baud 115200
```

URL: `http://127.0.0.1:8765`

Note: the legacy v1 dashboard from earlier today also auto-binds 8765 if the
old process is still alive. Kill it (it owns the `dashboard\server.py` path
with backslashes; the new one uses forward slashes if started from Bash). The
new dashboard's response shape has top-level `n_segments`, `segments[]`, and
`aggregate{}`; the v1 response uses single-segment `class_id` / `spectrogram`
/ `latency` keys.

Tabs:

- **Custom Upload** - WAV (any length) or 1D NPY waveform.
  - Optional **labels JSON** upload, or sidecar `<wav-stem>_labels.json`
    in `presentation_samples/` or `ml/data_circor/demo/` is auto-loaded.
  - 64x64 `.npy` spectrograms are intentionally rejected (would bypass the
    on-chip DSP).
- **Generic Loop** - cycles through `presentation_samples/*.wav`, then
  `ml/data_circor/demo/*.wav`, then a slice of `ml/data_circor/raw/training_data/*.wav`.

Scientific charts rendered for each upload:

- Aggregate result card (dominant class, confidence gauge, accuracy)
- Per-segment probability time-series (3 colored traces, gate-fired points marked)
- Color-coded segment timeline (click to select)
- Per-class cumulative bars
- Audio RMS over time
- Latency breakdown (DSP / Inference / UART up / UART return)
- Selected-segment waveform + audio stats (RMS, peak, ZCR, gate flag)
- STM32-computed 64x64 mel spectrogram
- Audio player + per-segment CSV export

## Presentation samples (live-demo set)

`presentation_samples/` holds three CirCor recordings, **trimmed to exactly
12 s** (six classification windows each) for quick live demos:

| File | Class | Patient | Note |
|---|---|---|---|
| `01_absent_pid49653.wav` | Absent  | 49653 (Adolescent, AV) | Healthy baseline, quiet S1/S2 only |
| `02_present_pid9979.wav` | Present | 9979 (TV)              | Holosystolic, grade III/VI, diamond |
| `03_unknown_pid9983.wav` | Unknown | 9983 (AV)              | Annotator was unsure - should trigger the gate |

Each ships with a `*_labels.json` sidecar so the dashboard fills the
"Ground Truth" column automatically and computes per-file accuracy.

Regenerate idempotently with `py presentation_samples\__build__.py`.

## Mixed multi-class demo

`ml/data_circor/demo/mixed_demo.wav` (default 18 s / 9 segments, 3 per class)
plus `mixed_demo_labels.json` for ground-truth accuracy. Generator:

```
py ml\generate_mixed_demo.py --per-class 3 --seed 2026
```

## What is NOT done

- The `tests/` directory was not refreshed to call the new `'A'` protocol.
- `05_validate_on_device.py` still uses the `'T'` test-vector path. Works,
  but does not exercise the new DSP-on-chip path.
- nRF52840 BLE peripheral was not touched.
- No git commit was created for any of these changes.

## Quick start for the next session

1. `git status --short --branch`
2. Build firmware: `./build.sh` (writes `build_stm32_synth/zephyr/zephyr.elf`)
3. Flash: `west flash --build-dir build_stm32_synth --runner openocd`
4. Start dashboard: `py dashboard\server.py --port COM6 --baud 115200`
5. Open `http://127.0.0.1:8765`
6. Upload `presentation_samples\02_present_pid9979.wav` - expect 6/6 Present.
