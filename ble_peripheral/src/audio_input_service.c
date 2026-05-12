/*
 * audio_input_service.c — phone -> nRF -> STM32 audio streaming path
 *
 * Adds a new GATT characteristic on the existing HeartSound service:
 *   UUID: 12345678-1234-1234-1234-123456789ABE
 *   Properties: WRITE | WRITE_WITHOUT_RESPONSE
 *   Format: int16 LE PCM at 4 kHz mono, accumulated into 16000-byte windows
 *
 * The phone writes the audio in chunks (chunk size = negotiated ATT MTU - 3).
 * With Data Length Extension and CONFIG_BT_L2CAP_TX_MTU=247 a single write
 * can carry up to 244 bytes, so a 16 KB window fits in ~66 writes (~500 ms
 * over BLE 1M PHY at typical 30 ms connection intervals + 4 pkts/event).
 *
 * Each completed window is forwarded over UART1 TX (P0.06 -> STM32 PD6,
 * USART2 RX) as 'B' (0x42) + 16000 raw bytes. The STM32 audio_bridge
 * thread runs DSP + INT8 inference and replies via ble_client_send() over
 * USART2 TX (PD5 -> nRF P0.08), which the main parser in ble_peripheral/
 * src/main.c picks up and BLE-notifies back to the phone.
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

/* UUIDs — must match audio_input_service.h doc + Python bleak client. */
#define BT_UUID_HSC_SERVICE_VAL \
    BT_UUID_128_ENCODE(0x12345678, 0x1234, 0x1234, 0x1234, 0x123456789ABCULL)
#define BT_UUID_HSC_AUDIO_IN_VAL \
    BT_UUID_128_ENCODE(0x12345678, 0x1234, 0x1234, 0x1234, 0x123456789ABEULL)

static struct bt_uuid_128 hsc_service_uuid   = BT_UUID_INIT_128(BT_UUID_HSC_SERVICE_VAL);
static struct bt_uuid_128 hsc_audio_in_uuid  = BT_UUID_INIT_128(BT_UUID_HSC_AUDIO_IN_VAL);

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

/* GATT write handler — phone fills the audio_in characteristic with PCM. */
static ssize_t audio_in_write(struct bt_conn *conn,
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
        /* Drop — previous window not yet handed off. Phone client should
         * pace its writes so this doesn't happen, but never crash on it. */
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

/* GATT service definition — adds the AudioIn characteristic on the same
 * service the existing classification characteristic lives on (the latter
 * is declared in heart_sound_service.c and is at index 1; this service
 * declaration is INDEPENDENT and only contains the AudioIn characteristic
 * so we don't disturb the existing one).
 *
 * Both BT_GATT_SERVICE_DEFINE blocks declare the same service UUID. Zephyr
 * stitches them in declaration order, but for cleanliness and to match
 * nRF Connect's expectation of a single service, we use the SAME primary
 * service declaration prefix on both. Discovery sees a single service with
 * two characteristics.
 */
BT_GATT_SERVICE_DEFINE(hsc_audio_in_svc,
    BT_GATT_PRIMARY_SERVICE(&hsc_service_uuid),
    BT_GATT_CHARACTERISTIC(&hsc_audio_in_uuid.uuid,
                           BT_GATT_CHRC_WRITE | BT_GATT_CHRC_WRITE_WITHOUT_RESP,
                           BT_GATT_PERM_WRITE,
                           NULL, audio_in_write, NULL),
    BT_GATT_CUD("Audio In (4 kHz int16 mono, 16000 B / 2 s window)",
                BT_GATT_PERM_READ),
);

void audio_input_service_init(const struct device *uart_to_stm32)
{
    s_uart_to_stm32 = uart_to_stm32;
    k_work_init(&s_forward_work, forward_work_handler);
    LOG_INF("AudioIn characteristic registered (UUID ...ABE, "
            "%u-byte window, write_without_response)", (unsigned)WINDOW_BYTES);
}
