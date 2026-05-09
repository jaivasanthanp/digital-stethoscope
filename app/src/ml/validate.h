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
 *   Host -> STM32 : 'S' to start synthetic PCG injection
 *   Host -> STM32 : 'P' to pause synthetic PCG injection
 *   STM32 -> Host : 0xA6 + command + state
 *
 * The 0xA5 magic prefix lets the Python script distinguish the binary
 * response from any LOG text appearing on the same UART.
 */

void validate_thread_fn(void *p1, void *p2, void *p3);

#endif /* VALIDATE_H */
