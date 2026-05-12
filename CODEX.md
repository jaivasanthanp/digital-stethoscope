# Digital Stethoscope (v2 / CirCor 2022) — Session Handoff

Last updated: 2026-05-12 (Claude session, BLE end-to-end verified)

## Project state at end of session (2026-05-12)

End-to-end demo path is now complete:

```
WAV  ─dashboard upload──▶  STM32U575  ─USART2 PD5─▶  nRF52840 DK  ─BLE─▶  Phone
        (browser, HTTP)    DSP + INT8 ML            ASCII notify          nRF Connect
```

Every WAV uploaded to the dashboard now fans out to the phone over BLE in
addition to the dashboard's scientific multi-segment view. A 12-second
presentation sample produces six BLE notifications on the phone, each
rendered as readable text (`"Present 95%"`, `"Absent 91%"`, `"Unknown 14%"`,
or `"Error 0%"`).

Verified end-to-end on hardware this session: STM32U575 NUCLEO-U575ZI-Q +
nRF52840 DK + Android phone running nRF Connect, uploading
`presentation_samples/02_present_pid9979.wav` from the local dashboard.

## What changed this session

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
