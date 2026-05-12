/*
 * audio_input_service.c — phone -> nRF -> STM32 audio streaming path
 *
 * Implements the AudioIn write handler + 16 KB accumulator + UART forward
 * worker. The GATT characteristic itself is declared inside the single
 * HeartSound primary service in heart_sound_service.c so that both
 * characteristics live under the same primary-service handle (required by
 * Web Bluetooth's getPrimaryService model).
 *
 * UUID of the AudioIn characteristic: 12345678-1234-1234-1234-123456789ABE
 * Properties: WRITE | WRITE_WITHOUT_RESPONSE
 *
 * Phone writes int16 LE PCM (4 kHz mono) in chunks up to (ATT MTU - 3) B
 * each. A full 2-second window is 16000 bytes. When the accumulator fills,
 * we hand it off to a system-workqueue worker that writes 'B' + 16000 bytes
 * to the STM32 on UART1 TX (P1.02 -> STM32 PD6). The STM32 audio_bridge
 * thread runs DSP + INT8 inference and replies via ble_client_send() over
 * USART2 TX (PD5 -> nRF P0.08), which the main UART parser turns into a
 * BLE notification on the classification characteristic.
 */

#include "audio_input_service.h"

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <string.h>

LOG_MODULE_REGISTER(audio_in, LOG_LEVEL_INF);

#define WINDOW_BYTES   16000U   /* 8000 samples * 2 bytes (int16 LE) */
#define STM32_CMD_AUDIO  'B'    /* 0x42 — matches audio_bridge.c on STM32 */

/* 16 KB accumulator for one audio window. */
static uint8_t s_window[WINDOW_BYTES];
static volatile size_t s_window_pos = 0;
static uint32_t s_window_counter = 0;

/* UART device handle, set by audio_input_service_init(). */
static const struct device *s_uart_to_stm32;

/* Worker that forwards a fully-buffered window over UART. We swap an internal
 * "ready" pointer so the GATT write handler can start filling the next window
 * while the worker drains the current one. */
static uint8_t s_window_ready[WINDOW_BYTES];
static struct k_work s_forward_work;

static void forward_work_handler(struct k_work *work)
{
    ARG_UNUSED(work);

    if (s_uart_to_stm32 == NULL) {
        LOG_WRN("forward: UART device not initialised");
        return;
    }

    uint32_t t0 = k_uptime_get_32();

    uart_poll_out(s_uart_to_stm32, STM32_CMD_AUDIO);
    for (size_t i = 0; i < WINDOW_BYTES; i++) {
        uart_poll_out(s_uart_to_stm32, s_window_ready[i]);
    }

    uint32_t dt = k_uptime_get_32() - t0;
    LOG_INF("forwarded window #%u to STM32 (%u bytes in %u ms)",
            s_window_counter, (unsigned)WINDOW_BYTES, dt);
}

ssize_t audio_input_write(struct bt_conn *conn,
                          const struct bt_gatt_attr *attr,
                          const void *buf, uint16_t len,
                          uint16_t offset, uint8_t flags)
{
    ARG_UNUSED(conn);
    ARG_UNUSED(attr);
    ARG_UNUSED(flags);

    if (offset != 0) {
        return BT_GATT_ERR(BT_ATT_ERR_INVALID_OFFSET);
    }
    if (len == 0) {
        return 0;
    }

    /* Disallow concurrent fills while a previous window is being forwarded.
     * The window swap below clears s_window_pos before re-enabling fills. */
    size_t pos = s_window_pos;
    size_t space = (pos < WINDOW_BYTES) ? (WINDOW_BYTES - pos) : 0;
    if (space == 0) {
        /* Drop — previous window not yet handed off. The phone-side client
         * should pace its writes so this doesn't happen, but never crash. */
        LOG_WRN("audio_in_write: window full, dropping %u bytes", len);
        return len;
    }

    size_t copy = (len < space) ? len : space;
    memcpy(&s_window[pos], buf, copy);
    pos += copy;
    s_window_pos = pos;

    if (pos >= WINDOW_BYTES) {
        /* Swap buffers: copy filled window into the worker's buffer, reset
         * the accumulator so the phone can start writing the next window
         * immediately. */
        memcpy(s_window_ready, s_window, WINDOW_BYTES);
        s_window_pos = 0;
        s_window_counter++;
        k_work_submit(&s_forward_work);
    }

    return copy;
}

void audio_input_service_init(const struct device *uart_to_stm32)
{
    s_uart_to_stm32 = uart_to_stm32;
    k_work_init(&s_forward_work, forward_work_handler);
    LOG_INF("AudioIn accumulator ready (16 KB window, write_without_response)");
}
