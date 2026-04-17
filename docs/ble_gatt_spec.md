# BLE GATT Specification — Heart Sound Classification Service

## Service

| Field | Value |
|---|---|
| Service UUID | `12345678-1234-1234-1234-123456789ABC` |
| Type | Custom (128-bit UUID) |
| Device name | `HeartSound` |

## Characteristic: HSC_Result

| Field | Value |
|---|---|
| UUID | `12345678-1234-1234-1234-123456789ABD` |
| Properties | NOTIFY |
| Length | 6 bytes |

### Packet Format

```
Byte 0 : class_id     uint8   Classification result
Byte 1 : confidence   uint8   Confidence 0–100 (%)
Byte 2 : reserved     uint8   0x00
Byte 3 : timestamp    uint8   ms bits  0– 7
Byte 4 : timestamp    uint8   ms bits  8–15
Byte 5 : timestamp    uint8   ms bits 16–23
```

### class_id Encoding

| Value | Meaning |
|---|---|
| 0x00 | Normal |
| 0x01 | Systolic Murmur |
| 0x02 | Diastolic Murmur |
| 0x03 | S3 Gallop |
| 0xFF | No Result / Error |

## Testing with nRF Connect

1. Install nRF Connect (iOS/Android)
2. Scan for device named `HeartSound`
3. Connect
4. Navigate to the custom service UUID
5. Subscribe to notifications on the `HSC_Result` characteristic
6. Observe classification packets every ~2 seconds

## BLE Parameters

| Parameter | Value |
|---|---|
| Advertising interval | 100 ms (fast) |
| Connection interval | 7.5–15 ms |
| TX power | 0 dBm |
| PHY | LE 1M |
