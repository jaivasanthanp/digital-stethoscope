/*
 * ble_client.c — UART bridge to nRF52840 BLE peripheral
 *
 * Transmits 6-byte packets at 115200 baud over UART (usart2, PD5 TX
 * -> NUCLEO D53 -> nRF P0.08 UART1 RX).
 * nRF52840 receives these and forwards them as BLE GATT notifications
 * on the Heart Sound Classification characteristic.
 *
 * Packet format (6 bytes, little-endian):
 *   [0]   class_id     uint8   (0=Absent 1=Present 2=Unknown)
 *   [1]   confidence   uint8   (0–100)
 *   [2]   reserved     uint8   (0x00)
 *   [3]   ts_low       uint8   (timestamp_ms bits 0–7)
 *   [4]   ts_mid       uint8   (timestamp_ms bits 8–15)
 *   [5]   ts_high      uint8   (timestamp_ms bits 16–23)
 */

#include "ble_client.h"

#include <zephyr/kernel.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

LOG_MODULE_REGISTER(ble_client, LOG_LEVEL_INF);

static const struct device *uart_dev;

void ble_client_init(void)
{
    uart_dev = DEVICE_DT_GET(DT_NODELABEL(usart2));

    if (!device_is_ready(uart_dev)) {
        LOG_WRN("UART (usart2) not ready — BLE bridge disabled");
        uart_dev = NULL;
        return;
    }

    LOG_INF("BLE client: UART bridge ready (115200 baud)");
}

void ble_client_send(uint8_t class_id, uint8_t confidence, uint32_t timestamp_ms)
{
    uint8_t pkt[6];
    pkt[0] = class_id;
    pkt[1] = confidence;
    pkt[2] = 0x00;
    pkt[3] = (uint8_t)(timestamp_ms & 0xFF);
    pkt[4] = (uint8_t)((timestamp_ms >> 8) & 0xFF);
    pkt[5] = (uint8_t)((timestamp_ms >> 16) & 0xFF);

    if (uart_dev == NULL) {
        /* No UART — just log the result (useful during early dev) */
        static const char *names[] = {"Absent", "Present", "Unknown"};
        const char *name = (class_id < ARRAY_SIZE(names)) ? names[class_id] : "Invalid";
        LOG_INF("BLE stub: class=%s conf=%u%% ts=%ums", name, confidence, timestamp_ms);
        return;
    }

    for (int i = 0; i < 6; i++) {
        uart_poll_out(uart_dev, pkt[i]);
    }
}
