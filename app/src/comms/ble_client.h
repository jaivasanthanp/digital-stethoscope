#ifndef BLE_CLIENT_H
#define BLE_CLIENT_H

#include <stdint.h>

/*
 * Initialize UART bridge to nRF52840.
 * Configures UART at 115200 baud (usart3 in devicetree overlay).
 */
void ble_client_init(void);

/*
 * Send a 6-byte heart sound classification result packet to nRF52840.
 * Packet format (little-endian):
 *   [0]   class_id     : uint8  (0=Normal 1=SysMurmur 2=DiaMurmur 3=S3Gallop 0xFF=NoResult)
 *   [1]   confidence   : uint8  (0–100)
 *   [2]   reserved     : uint8  (0x00)
 *   [3–5] timestamp_ms : uint24 little-endian (low 24 bits of uptime)
 */
void ble_client_send(uint8_t class_id, uint8_t confidence, uint32_t timestamp_ms);

#endif /* BLE_CLIENT_H */
