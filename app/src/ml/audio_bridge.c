/*
 * audio_bridge.c — phone -> BLE -> nRF -> STM32 audio streaming path
 *
 * Listens on USART2 RX (PD6) for windows of audio forwarded by the nRF52840
 * after the phone has written them to the new audio-input GATT characteristic.
 * Per arrived window, runs the same on-chip DSP + inference path that the
 * dashboard's 'A' command uses, then emits the result via ble_client_send()
 * over USART2 TX (PD5) — which the nRF then turns into a BLE notification
 * back to the phone.
 *
 * Wire protocol on USART2:
 *   nRF -> STM32 : 'B' (0x42) + 16000 bytes int16 LE PCM (8000 samples @ 4 kHz)
 *   STM32 -> nRF : 6-byte ble_client_send packet (existing format,
 *                  unchanged so the nRF parser keeps working).
 */

#include "audio_bridge.h"
#include "inference.h"
#include "comms/ble_client.h"
#include "dsp/mel_spec.h"

#include <zephyr/kernel.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/devicetree.h>
#include <zephyr/sys/ring_buffer.h>
#include <zephyr/logging/log.h>
#include <string.h>

LOG_MODULE_REGISTER(audio_bridge, LOG_LEVEL_INF);

#define BRIDGE_CMD_AUDIO        'B'    /* 0x42 */
#define BRIDGE_AUDIO_SAMPLES    8000   /* 2 s @ 4 kHz */
#define BRIDGE_AUDIO_BYTES      (BRIDGE_AUDIO_SAMPLES * 2)
#define BRIDGE_SPEC_ELEMENTS    (MEL_OUT_ROWS * MEL_OUT_COLS)

/* ── Ring buffer ─ 18 KB ─ absorbs a full 16 KB upload plus jitter ───────── */
#define BRIDGE_RX_BUF_SIZE  18000
RING_BUF_DECLARE(s_bridge_rx_rb, BRIDGE_RX_BUF_SIZE);
static K_SEM_DEFINE(s_bridge_rx_sem, 0, BRIDGE_RX_BUF_SIZE);

/* ── Private float32 audio buffer ─ 32 KB ─ owned by the bridge thread ──── */
static float s_bridge_audio[BRIDGE_AUDIO_SAMPLES];

/* ── Private mel-spec buffer ─ 16 KB ─ inference output is written here ── */
static float s_bridge_spec[BRIDGE_SPEC_ELEMENTS];

static const struct device *s_uart;

static void bridge_rx_isr(const struct device *dev, void *user_data)
{
    ARG_UNUSED(user_data);

    if (!uart_irq_update(dev)) {
        return;
    }

    while (uart_irq_rx_ready(dev)) {
        uint8_t byte;
        int n = uart_fifo_read(dev, &byte, 1);
        if (n == 1) {
            ring_buf_put(&s_bridge_rx_rb, &byte, 1);
            k_sem_give(&s_bridge_rx_sem);
        }
    }
}

/* Block until a byte arrives from the bridge ring buffer. */
static uint8_t recv_byte(void)
{
    k_sem_take(&s_bridge_rx_sem, K_FOREVER);
    uint8_t b;
    ring_buf_get(&s_bridge_rx_rb, &b, 1);
    return b;
}

/* Read one 16 KB int16 LE window into the float buffer, normalised to [-1,1]. */
static void recv_one_window(float *dest)
{
    for (size_t i = 0; i < BRIDGE_AUDIO_SAMPLES; i++) {
        uint8_t lo = recv_byte();
        uint8_t hi = recv_byte();
        int16_t sample = (int16_t)((uint16_t)lo | ((uint16_t)hi << 8));
        dest[i] = (float)sample / 32768.0f;
    }
}

/* Bridge worker thread — only ever runs when nRF pushes audio over the wire. */
static void audio_bridge_thread(void *p1, void *p2, void *p3)
{
    ARG_UNUSED(p1); ARG_UNUSED(p2); ARG_UNUSED(p3);

    LOG_INF("audio_bridge: listening on USART2 RX (PD6) for 'B' + %u byte windows",
            BRIDGE_AUDIO_BYTES);

    while (1) {
        /* Wait for the 'B' sync byte. Tolerate junk in between (resync). */
        uint8_t b = recv_byte();
        if (b != BRIDGE_CMD_AUDIO) {
            continue;
        }

        /* Fill the audio window. ~1.4 s at 115200 baud for 16 KB. */
        recv_one_window(s_bridge_audio);

        /* DSP: mel spectrogram — mutex-protected, shared with validate / dsp threads */
        k_mutex_lock(&g_mel_spec_mutex, K_FOREVER);
        mel_spec_compute(s_bridge_audio, BRIDGE_AUDIO_SAMPLES,
                         s_bridge_spec, MEL_OUT_ROWS, MEL_OUT_COLS);
        k_mutex_unlock(&g_mel_spec_mutex);

        /* Inference: INT8 ResNet — mutex-protected, shared with the rest */
        uint8_t confidence = 0;
        uint8_t raw_probs[3] = {0};
        uint8_t gate_applied = 0;
        k_mutex_lock(&g_inference_mutex, K_FOREVER);
        int class_id = inference_run_probs(s_bridge_spec, BRIDGE_SPEC_ELEMENTS,
                                           &confidence, raw_probs, &gate_applied);
        k_mutex_unlock(&g_inference_mutex);

        uint8_t out_class = (class_id < 0) ? 0xFFu : (uint8_t)class_id;

        /* Send classification back to nRF via the existing 6-byte format.
         * ble_client_send() writes to USART2 TX (PD5); the nRF parses the
         * same 6 bytes it already handles from the dashboard-upload and
         * synthetic-loop paths, and BLE-notifies the result to the phone. */
        ble_client_send(out_class, confidence, k_uptime_get_32());

        LOG_INF("audio_bridge: class=%u conf=%u%% raw=[%u,%u,%u] gate=%u",
                out_class, confidence,
                raw_probs[0], raw_probs[1], raw_probs[2], gate_applied);
    }
}

#define BRIDGE_STACK_SIZE 4096
#define BRIDGE_PRIORITY   7

K_THREAD_STACK_DEFINE(s_bridge_stack, BRIDGE_STACK_SIZE);
static struct k_thread s_bridge_thread_data;

void audio_bridge_init(void)
{
    s_uart = DEVICE_DT_GET(DT_NODELABEL(usart2));
    if (!device_is_ready(s_uart)) {
        LOG_WRN("audio_bridge: usart2 not ready - phone-to-BLE path disabled");
        return;
    }

    uart_irq_rx_disable(s_uart);
    uart_irq_callback_user_data_set(s_uart, bridge_rx_isr, NULL);
    uart_irq_rx_enable(s_uart);

    k_thread_create(&s_bridge_thread_data, s_bridge_stack, BRIDGE_STACK_SIZE,
                    audio_bridge_thread, NULL, NULL, NULL,
                    BRIDGE_PRIORITY, 0, K_NO_WAIT);
    k_thread_name_set(&s_bridge_thread_data, "audio_bridge");
}
