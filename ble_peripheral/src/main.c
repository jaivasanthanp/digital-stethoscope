/*
 * ble_peripheral/src/main.c — nRF52840 BLE GATT peripheral
 *
 * Receives 6-byte packets from STM32U575 over UART1 and forwards them
 * as BLE GATT notifications to a connected central (phone / nRF Connect app).
 */

#include <zephyr/kernel.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/logging/log.h>

#include "heart_sound_service.h"

LOG_MODULE_REGISTER(ble_main, LOG_LEVEL_INF);

/* -------------------------------------------------------------------------
 * BLE advertising data
 * ------------------------------------------------------------------------- */
static const struct bt_data ad[] = {
    BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
    BT_DATA(BT_DATA_NAME_COMPLETE, "HeartSound", sizeof("HeartSound") - 1),
};

static struct bt_conn *current_conn;

static void connected(struct bt_conn *conn, uint8_t err)
{
    if (err) {
        LOG_ERR("Connection failed: %u", err);
        return;
    }
    current_conn = bt_conn_ref(conn);
    LOG_INF("BLE connected");
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
    LOG_INF("BLE disconnected (reason 0x%02x)", reason);
    if (current_conn) {
        bt_conn_unref(current_conn);
        current_conn = NULL;
    }
    /* Restart advertising */
    bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad), NULL, 0);
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
    .connected    = connected,
    .disconnected = disconnected,
};

/* -------------------------------------------------------------------------
 * UART receiver — parses 6-byte packets from STM32
 * ------------------------------------------------------------------------- */
static const struct device *uart_dev;
static uint8_t uart_buf[6];
static int uart_buf_pos;

static void uart_rx_cb(const struct device *dev, void *user_data)
{
    ARG_UNUSED(user_data);

    if (!uart_irq_update(dev)) {
        return;
    }

    while (uart_irq_rx_ready(dev)) {
        uint8_t byte;
        int n = uart_fifo_read(dev, &byte, 1);
        if (n <= 0) continue;

        uart_buf[uart_buf_pos++] = byte;

        if (uart_buf_pos == 6) {
            uart_buf_pos = 0;

            uint8_t  class_id    = uart_buf[0];
            uint8_t  confidence  = uart_buf[1];
            uint32_t ts_ms       = (uint32_t)uart_buf[3]
                                 | ((uint32_t)uart_buf[4] << 8)
                                 | ((uint32_t)uart_buf[5] << 16);

            static const char *names[] = {"Absent", "Present", "Unknown"};
            const char *name = (class_id < ARRAY_SIZE(names)) ? names[class_id] : "Invalid";
            LOG_INF("RX: %s  conf=%u%%  ts=%ums", name, confidence, ts_ms);

            if (current_conn) {
                hsc_service_notify(current_conn, class_id, confidence, ts_ms);
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */
int main(void)
{
    LOG_INF("=== HeartSound BLE Peripheral ===");

    /* Init UART */
    uart_dev = DEVICE_DT_GET(DT_NODELABEL(uart1));
    if (device_is_ready(uart_dev)) {
        uart_irq_callback_set(uart_dev, uart_rx_cb);
        uart_irq_rx_enable(uart_dev);
        LOG_INF("UART1 ready (115200 baud)");
    } else {
        LOG_WRN("UART1 not ready");
    }

    /* Init BLE */
    int err = bt_enable(NULL);
    if (err) {
        LOG_ERR("bt_enable failed: %d", err);
        return err;
    }

    hsc_service_init();

    err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad), NULL, 0);
    if (err) {
        LOG_ERR("Advertising start failed: %d", err);
        return err;
    }

    LOG_INF("Advertising as 'HeartSound' — connect with nRF Connect app");
    return 0;
}
