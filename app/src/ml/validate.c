/*
 * validate.c — On-device inference validation via UART
 *
 * Runs as a low-priority Zephyr thread. Listens on the console UART for
 * test commands from ml/05_validate_on_device.py, runs inference on the
 * corresponding pre-computed test vector from test_vectors.h, and returns
 * the result as a 3-byte framed response.
 *
 * Protocol:
 *   Host -> STM32 : 'T' (0x54) + uint16 index (little-endian, 3 bytes total)
 *   STM32 -> Host : 0xA5 (magic) + class_id (uint8) + confidence (uint8)
 *
 * The 0xA5 magic byte is never produced by Zephyr LOG text output (all
 * LOG characters are printable ASCII), so Python can unambiguously find
 * the response amid concurrent log messages on the same UART.
 *
 * RX implementation: interrupt-driven ring buffer via uart_irq_callback.
 * uart_poll_in() does not work reliably when CONFIG_UART_INTERRUPT_DRIVEN=y
 * is active on the same UART device.
 *
 * Thread safety: g_inference_mutex (defined in inference.cc) serialises
 * calls to inference_run() between this thread and inference_thread.
 */

#include "validate.h"
#include "inference.h"
#include "test_vectors.h"

#include <zephyr/kernel.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/sys/ring_buffer.h>
#include <zephyr/logging/log.h>
#include <zephyr/devicetree.h>

LOG_MODULE_REGISTER(validate, LOG_LEVEL_INF);

/* ── Response framing ───────────────────────────────────────────────────── */
#define VALIDATE_MAGIC   0xA5u
#define CMD_TEST_VECTOR  'T'

/* ── Console UART ───────────────────────────────────────────────────────── */
static const struct device *s_console;

/* ── RX ring buffer — stores incoming bytes from the host ───────────────── */
/* 32 bytes is plenty: max 3-byte command × a few commands in flight         */
#define RX_BUF_SIZE  32
RING_BUF_DECLARE(s_rx_rb, RX_BUF_SIZE);
static K_SEM_DEFINE(s_rx_sem, 0, RX_BUF_SIZE);

/* UART RX interrupt handler — called by the UART driver ISR */
static void console_rx_isr(const struct device *dev, void *user_data)
{
    ARG_UNUSED(user_data);

    if (!uart_irq_update(dev)) {
        return;
    }

    while (uart_irq_rx_ready(dev)) {
        uint8_t byte;
        int n = uart_fifo_read(dev, &byte, 1);
        if (n == 1) {
            ring_buf_put(&s_rx_rb, &byte, 1);
            k_sem_give(&s_rx_sem);
        }
    }
}

/* Block until one byte is available from the RX ring buffer. */
static uint8_t recv_byte(void)
{
    k_sem_take(&s_rx_sem, K_FOREVER);
    uint8_t b;
    ring_buf_get(&s_rx_rb, &b, 1);
    return b;
}

/* Send one raw byte back to the host over the console UART. */
static inline void send_byte(uint8_t b)
{
    uart_poll_out(s_console, b);
}

/* ── Thread entry ─────────────────────────────────────────────────────────── */

void validate_thread_fn(void *p1, void *p2, void *p3)
{
    ARG_UNUSED(p1); ARG_UNUSED(p2); ARG_UNUSED(p3);

    s_console = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
    if (!device_is_ready(s_console)) {
        LOG_ERR("Console UART not ready — validate thread disabled");
        return;
    }

    /* Install interrupt-driven RX handler on the console UART.
     * This does not affect LOG TX output (TX and RX are independent). */
    uart_irq_callback_user_data_set(s_console, console_rx_isr, NULL);
    uart_irq_rx_enable(s_console);

    LOG_INF("Validate thread ready (%d test vectors)", TEST_VECTOR_COUNT);

    while (1) {
        /* Wait for command byte 'T' (0x54), discard anything else */
        uint8_t cmd = recv_byte();
        if (cmd != CMD_TEST_VECTOR) {
            continue;
        }

        /* Read 2-byte vector index (little-endian uint16) */
        uint8_t  lo  = recv_byte();
        uint8_t  hi  = recv_byte();
        uint16_t idx = (uint16_t)lo | ((uint16_t)hi << 8);

        if (idx >= TEST_VECTOR_COUNT) {
            LOG_WRN("Validate: invalid index %u (max %d)", idx, TEST_VECTOR_COUNT - 1);
            send_byte(VALIDATE_MAGIC);
            send_byte(0xFF);   /* class = invalid */
            send_byte(0);
            continue;
        }

        /* Run inference on the pre-normalised test vector.
         * Take mutex to serialise with inference_thread. */
        uint8_t confidence = 0;
        k_mutex_lock(&g_inference_mutex, K_FOREVER);
        int class_id = inference_run(test_vectors[idx],
                                     TEST_VECTOR_SAMPLES,
                                     &confidence);
        k_mutex_unlock(&g_inference_mutex);

        uint8_t out_class = (class_id < 0) ? 0xFF : (uint8_t)class_id;

        /* Send framed response: 0xA5 + class_id + confidence */
        send_byte(VALIDATE_MAGIC);
        send_byte(out_class);
        send_byte(confidence);

        static const char *class_names[] = {
            "Normal", "SysMurmur", "DiaMurmur", "S3Gallop"
        };
        const char *name = (out_class < 4) ? class_names[out_class] : "INVALID";
        LOG_INF("Validate[%2u] -> %s  conf=%u%%", idx, name, confidence);
    }
}
