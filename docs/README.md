# Digital Stethoscope — Web Bluetooth Phone Client

A single-page HTML/JS app that turns any Android phone running Chrome into a
wireless stethoscope frontend for the Digital Stethoscope project. Captures
from the phone microphone, streams 4 kHz int16 PCM windows over BLE to the
nRF52840 DK paired with the STM32U575, and shows the classification
notifications coming back.

No native app install. No backend server needed for the live demo — the
phone talks directly to the nRF over Bluetooth.

## Live URL (once GitHub Pages is enabled)

```
https://jaivasanthanp.github.io/digital-stethoscope/
```

Open that on **Chrome on Android** (Chrome ≥ 56 with Web Bluetooth flag on
by default). Tap **Connect to HeartSound**, pick the device in the system
chooser, tap **Start streaming mic**, and you should see classification
text update every ~2-5 seconds.

## Enabling GitHub Pages (one-time, ~60 seconds)

1. Open the repo on github.com → **Settings**.
2. Left sidebar → **Pages**.
3. **Source:** *Deploy from a branch*.
4. **Branch:** `master` (or `main`), folder: `/docs`.
5. **Save**. GitHub deploys within ~30-60 seconds.
6. The page is live at `https://<username>.github.io/<repo>/`.

If you'd rather host elsewhere, the page is fully static — drop `index.html`
on any HTTPS host (Web Bluetooth requires HTTPS or `localhost`).

## Why Web Bluetooth and not Flutter

The original Stage B of task 6 specced a Flutter Android app. Web
Bluetooth + Web Audio gives the same end-to-end demo for ~200 lines of
code and zero installs on the user's phone. Pros / cons:

| | Web Bluetooth | Flutter Android |
|---|---|---|
| Install on phone | none, just open URL | sideload APK |
| Toolchain setup | none | Flutter SDK + Android SDK (~3-5 GB) |
| iOS support | only via Bluefy | with Mac toolchain, app signing |
| GATT API parity | identical (writes / notifications) | identical |
| Throughput | ~10-30 KB/s on Android Chrome | similar |
| Demo "look" | website on phone | native app |

The firmware (STM32 + nRF) is the same in both cases — see
`ble_peripheral/src/audio_input_service.{c,h}` and
`app/src/ml/audio_bridge.{c,h}`.

## Local testing without GH Pages

Web Bluetooth requires HTTPS or `localhost`. To test on the laptop's
Chrome (desktop has a Bluetooth radio):

```
python -m http.server -d docs 8080
# open http://localhost:8080
```

That works on `localhost`. For the phone, the page must be served over
HTTPS — GH Pages is the simplest path, but `ngrok` / `cloudflared` /
`mkcert + a TLS-enabled http.server` all work too.

## How the data flows

```
Phone mic (44.1 / 48 kHz)
    │  Web Audio: ScriptProcessor 4096-sample chunks
    │  Downsample (linear interp) → 4 kHz, peak-normalize quiet windows
    ▼
Web Bluetooth: writeValueWithoutResponse() in 180-byte chunks
    │  → characteristic 12345678-1234-1234-1234-123456789ABE
    │      (audio_in, accumulator on nRF, 16 KB per window)
    ▼
nRF52840 DK forwards 'B' + 16 KB on UART1 TX (P1.02)
    │
    ▼
STM32U575 audio_bridge: DSP + ResNet-18 INT8 inference
    │
    ▼
ble_client_send() 6-byte packet back over UART1 RX (P0.08)
    │
    ▼
nRF BLE-notifies "Absent NN%" / "Present NN%" / "Unknown NN%" on
characteristic 12345678-1234-1234-1234-123456789ABD
    │
    ▼
Phone Chrome receives notification → updates the result card on screen
```

## Browser compatibility

| Browser / OS | Status |
|---|---|
| Chrome / Edge on Android | Supported (Web Bluetooth + Web Audio) |
| Chrome / Edge on Windows / Mac / Linux | Supported (laptop mic + laptop BT radio) |
| Safari on iOS | **Not supported** — Apple does not implement Web Bluetooth. Use the *Bluefy* browser as a fallback on iOS. |
| Firefox | Not supported (Mozilla declines to ship Web Bluetooth). |

## Troubleshooting

- **"BLE connect failed: User cancelled the requestDevice() chooser"** —
  the system device picker was dismissed. Tap *Connect* again.
- **"Mic / streaming failed: Permission denied"** — Chrome needs mic
  permission. Pull down notification shade, allow microphone.
- **No notifications arrive after streaming starts** — first verify the
  Python `ml/06_validate_ble_audio.py` works from the laptop. If that
  works but the phone doesn't, check both jumper wires on the nRF / STM32
  are still in place and that no other BLE central (e.g. the bleak
  script) is currently connected to the nRF.
- **Throughput feels slow** — Android negotiates ATT MTU per-connection.
  If your phone falls back to MTU 23, each chunk drops to 20 bytes and a
  16 KB window takes ~8 seconds. Newer phones and Chrome versions
  negotiate up to 247 automatically.

## License / credits

Part of the
[Digital Stethoscope](https://github.com/jaivasanthanp/digital-stethoscope)
project. Same license as the parent repo.
