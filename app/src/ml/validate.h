#ifndef VALIDATE_H
#define VALIDATE_H

/*
 * validate.h — On-device inference validation via UART (Phase 2 test harness)
 *
 * validate_thread_fn() runs as a low-priority Zephyr thread and listens on
 * the console UART for test commands from ml/05_validate_on_device.py.
 *
 * Protocol:
 *   Host -> STM32 : 'T' (0x54) + 2-byte vector index (uint16 little-endian)
 *   STM32 -> Host : 0xA5 (magic) + class_id (uint8) + confidence (uint8)
 *
 *   Host -> STM32 : 'A' (0x41) + 8000 little-endian int16 PCM samples,
 *                   delivered in 128-sample (256-byte) chunks.
 *                   The STM32 ACKs each chunk with 0xA6 plus a final 0xA6
 *                   once all samples are received.
 *   STM32 -> Host : 0xA5 + class_id (post unknown-gate) + confidence
 *                 + 3 raw probability bytes (Absent, Present, Unknown)
 *                 + gate_applied (uint8, 0 or 1)
 *                 + dsp_ms          (uint16 little-endian)
 *                 + inference_ms    (uint16 little-endian)
 *                 + rms_q15         (uint16 little-endian, rms * 32768)
 *                 + peak_q15        (uint16 little-endian, peak * 32768)
 *                 + zcr             (uint16 little-endian, zero-crossing count)
 *                 + spectrogram     (4096 little-endian float32 values)
 *                 (Total: 18 header bytes + 16384 spec = 16402 bytes)
 *
 *   Host -> STM32 : 'S' to start the synthetic PCG demo loop
 *   Host -> STM32 : 'P' to pause the synthetic PCG demo loop
 *   STM32 -> Host : 0xA6 + command byte + state (kept for legacy demo mode)
 *
 * The 0xA5 magic prefix unambiguously frames a binary response on the UART.
 * CONFIG_LOG is disabled in prj.conf so no log text can interleave with the
 * binary payload.
 */

void validate_thread_fn(void *p1, void *p2, void *p3);

#endif /* VALIDATE_H */
