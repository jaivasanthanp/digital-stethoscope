#ifndef AUDIO_INPUT_SERVICE_H
#define AUDIO_INPUT_SERVICE_H

#include <stdint.h>
#include <zephyr/bluetooth/conn.h>

/* Audio Input characteristic UUID (next slot in our private service):
 *   12345678-1234-1234-1234-123456789ABE
 *
 * Phone writes chunks of int16 LE PCM (4 kHz mono) into this characteristic
 * using WRITE_WITHOUT_RESPONSE. Each window is 16000 bytes = 8000 samples.
 *
 * When a complete window is accumulated, the service forwards 'B' + the
 * 16 KB payload to the STM32U575 over UART1 TX (P0.06 -> STM32 PD6).
 * The STM32 runs DSP + INT8 inference and sends the classification back
 * over UART1 RX (existing wire) as a 6-byte packet, which the main UART
 * parser already turns into a BLE notification on the classification
 * characteristic.
 */

/* Public init — call once at boot from main(), after the UART is set up. */
void audio_input_service_init(const struct device *uart_to_stm32);

#endif /* AUDIO_INPUT_SERVICE_H */
