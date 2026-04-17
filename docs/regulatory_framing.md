# Regulatory Framing — IEC 62304 / ISO 14971 Notes

> **Disclaimer:** This is a student portfolio project. It is NOT intended for
> clinical use. These notes frame the project within medical device software
> standards for interview/exam purposes only.

## Software Classification (IEC 62304)

**Classification: Class B SaMD**

- Output is advisory (displays classification to a clinician)
- Not directly controlling a device or treatment
- Incorrect output could contribute to a missed diagnosis (harm pathway exists)
- Class C would require output to autonomously administer treatment

**Implications for Class B:**
- Software development lifecycle documentation required
- Unit + integration testing required
- Anomaly resolution process required
- No formal verification/validation (that's Class C)

## Risk Analysis Summary (ISO 14971)

| Hazard | Severity | Probability | Risk Level | Mitigation |
|---|---|---|---|---|
| Acoustic coupling inconsistency (poor mic placement) | Moderate | High | Medium | User instruction: reposition if confidence < 60% |
| Small training dataset (~3,240 recordings, 4 sites) | High | Medium | High | Clinical study required before deployment |
| INT8 quantization accuracy loss | Low | Low | Low | Validated < 2% drop during development |
| Battery/power failure mid-measurement | Low | Low | Low | Result invalidated, repeat measurement |
| BLE interference causing missed notification | Low | Medium | Low | GATT retry, host-side packet loss detection |

**Primary residual risk:** Dataset size and diversity insufficient for population-level
use. Mitigation path: IRB-approved clinical study with 500+ patients, echocardiography
ground truth.

## IEC 60601-1 Electrical Safety

Current prototype has exposed PCB and direct connection to NUCLEO development board.

**Not suitable for patient contact.** Minimum changes for clinical prototype:
- Isolated power supply (no USB connection during patient contact)
- Enclosure rated for patient environment (IP2X minimum)
- Creepage/clearance analysis for applied parts

## Build Reproducibility (IEC 62304 §8.1)

The Zephyr `west build` system provides a reproducible build via:
- `west.yml` pins exact Zephyr commit hash
- `prj.conf` pins all Kconfig options
- `build_info.yml` records compiler, SDK version, and git hash

This is superior to STM32CubeIDE for regulatory traceability because the
entire build is driven by text files in version control.
