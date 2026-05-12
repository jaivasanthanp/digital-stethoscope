#ifndef AUDIO_INPUT_SERVICE_H
#define AUDIO_INPUT_SERVICE_H

#include <stdint.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/device.h>

/* Audio Input characteristic UUID (declared inside the single
 * HeartSound primary service in heart_sound_service.c):
 *   12345678-1234-1234-1234-123456789ABE
 *
 * Phone writes chunks of int16 LE PCM (4 kHz mono) into this characteristic
 * using WRITE_WITHOUT_RESPONSE. Each window is 16000 bytes = 8000 samples.
 *
 * When a complete window is accumulated, the service forwards 'B' + the
 * 16 KB payload to the STM32U575 over UART1 TX (P1.02 -> STM32 PD6).
 * The STM32 runs DSP + INT8 inference and sends the classification back
 * over UART1 RX (P0.08) as a 6-byte packet, which the main UART parser
 * already turns into a BLE notification on the classification
 * characteristic.
 *
 * To keep all characteristics inside a SINGLE primary GATT service
 * (required by Web Bluetooth's getPrimaryService model), this module
 * does NOT declare its own BT_GATT_SERVICE_DEFINE block. Instead it
 * exposes the write handler so heart_sound_service.c can include the
 * audio-in characteristic alongside the classification one.
 */

/* GATT write handler for the AudioIn characteristic. */
ssize_t audio_input_write(struct bt_conn *conn,
                          const struct bt_gatt_attr *attr,
                          const void *buf, uint16_t len,
                          uint16_t offset, uint8_t flags);

/* Initialise the accumulator + the UART forward worker. Must be called
 * once from main() before BLE advertising starts. */
void audio_input_service_init(const struct device *uart_to_stm32);

#endif /* AUDIO_INPUT_SERVICE_H */
