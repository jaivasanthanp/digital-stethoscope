#ifndef AUDIO_BRIDGE_H
#define AUDIO_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Initialise the audio-bridge UART RX listener. Must be called once at boot
 * AFTER inference_init() / mel_spec_init() / ble_client_init(). Spawns a
 * dedicated worker thread that consumes 16 KB int16 PCM windows arriving
 * on USART2 RX (PD6) from the nRF52840 DK and routes the classification
 * back to the phone over the existing BLE service via ble_client_send().
 *
 * Wire protocol on USART2 (115200 baud, 8N1):
 *   nRF -> STM32 : 'B' (0x42) + 16000 bytes int16 LE PCM   (16001 B)
 *   STM32 -> nRF : <legacy 6-byte ble_client_send packet>  (6 B)
 *
 * The 6-byte response shape matches what ble_client_send() already emits
 * for the synthetic-loop and dashboard-upload paths, so the nRF parser in
 * ble_peripheral/src/main.c needs no changes to handle it.
 */
void audio_bridge_init(void);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_BRIDGE_H */
