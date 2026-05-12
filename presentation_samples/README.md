# Presentation Samples

Three CirCor / PhysioNet 2022 heart-sound recordings — one per class — trimmed
to exactly **12 s** (six 2-second STM32 classification windows) for a quick
live demo.

Upload any of these WAVs through the Custom Upload tab at
`http://127.0.0.1:8765` — the dashboard automatically picks up the matching
`*_labels.json` sidecar and shows per-segment ground-truth accuracy in the
results table. They also appear in the Generic Loop dropdown.

| File | Class | Patient | Clinical note |
|---|---|---|---|
| `01_absent_pid49653.wav` | **Absent**  | 49653 (Adolescent, AV) | Healthy baseline. No murmur. Quiet S1/S2 only. |
| `02_present_pid9979.wav` | **Present** | 9979 (TV)              | Holosystolic murmur, **grade III/VI**, diamond shape, most audible at the tricuspid position. |
| `03_unknown_pid9983.wav` | **Unknown** | 9983 (AV)              | CirCor annotator was unsure whether a murmur is present. Should ideally trigger the calibrated unknown gate. |

Each file is 16-bit PCM, 4 kHz mono, peak-normalised to 0.85 so all three play
back at a comparable level. Source recordings live under
`ml/data_circor/raw/training_data/` and ground-truth class labels come from
`ml/data_circor/raw/training_data.csv`.

Regenerate the samples (e.g., after pulling new CirCor data) with:

```
py presentation_samples/__build__.py
```
