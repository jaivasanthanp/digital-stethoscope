/*
 * validate.c — On-device inference + audio-upload UART harness.
 *
 * Three commands are supported on the console UART:
 *
 *   'T' + uint16 index    : run inference on test_vectors[index]
 *                           STM32 returns 0xA5 + class + confidence (3 bytes).
 *
 *   'A' + 8000 LE int16   : receive 2-second PCM at 4 kHz, run DSP + inference
 *                           on-chip. STM32 ACKs each 128-sample chunk with 0xA6
 *                           and returns an extended 0xA5 response packet
 *                           containing class, raw probabilities, latency, audio
 *                           statistics and the full 64x64 mel spectrogram.
 *
 *   'S' / 'P'             : legacy synthetic-injection enable/disable.
 *
 * Response framing relies on CONFIG_LOG being disabled in prj.conf — no log
 * text can interleave with the binary payload on the same UART.
 *
 * RX uses an interrupt-driven ring buffer (40 KB) so the entire audio upload
 * can be absorbed without blocking the ISR.
 */

#include "validate.h"
#include "inference.h"
#include "test_vectors.h"
#include "audio/i2s_capture.h"
#include "comms/ble_client.h"
#include "dsp/mel_spec.h"

#include <zephyr/kernel.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/sys/ring_buffer.h>
#include <zephyr/sys/util.h>
#include <zephyr/devicetree.h>

#include <string.h>
#include <math.h>

/* ── Response framing ───────────────────────────────────────────────────── */
#define VALIDATE_MAGIC          0xA5u
#define UPLOAD_ACK_MAGIC        0xA6u
#define CONTROL_MAGIC           0xA6u
#define CMD_TEST_VECTOR         'T'
#define CMD_AUDIO_UPLOAD        'A'
#define CMD_SYNTH_START         'S'
#define CMD_SYNTH_PAUSE         'P'

#define UPLOAD_AUDIO_SAMPLES        8000
#define UPLOAD_AUDIO_CHUNK_ELEMENTS 128
#define UPLOAD_SPEC_ELEMENTS        (MEL_OUT_ROWS * MEL_OUT_COLS)

/* ── Console UART ───────────────────────────────────────────────────────── */
static const struct device *s_console;

/* ── RX ring buffer — absorbs an entire 16 KB audio upload plus jitter ──── */
#define RX_BUF_SIZE  40000
RING_BUF_DECLARE(s_rx_rb, RX_BUF_SIZE);
static K_SEM_DEFINE(s_rx_sem, 0, RX_BUF_SIZE);

/* Reusable static buffers for the upload pipeline. */
static float s_uploaded_audio[UPLOAD_AUDIO_SAMPLES];
static float s_uploaded_spec[UPLOAD_SPEC_ELEMENTS];

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

static inline void send_u16_le(uint16_t value)
{
    send_byte((uint8_t)(value & 0xFFu));
    send_byte((uint8_t)((value >> 8) & 0xFFu));
}

static void send_float_array(const float *data, size_t n_elements)
{
    const uint8_t *bytes = (const uint8_t *)data;
    const size_t total = n_elements * sizeof(float);
    for (size_t i = 0; i < total; i++) {
        send_byte(bytes[i]);
    }
}

/* Receive 8000 LE int16 samples; ACK each 128-sample chunk with 0xA6. */
static void recv_uploaded_audio(float *dest)
{
    for (size_t i = 0; i < UPLOAD_AUDIO_SAMPLES; i++) {
        uint8_t lo = recv_byte();
        uint8_t hi = recv_byte();
        int16_t sample = (int16_t)((uint16_t)lo | ((uint16_t)hi << 8));
        dest[i] = (float)sample / 32768.0f;

        if (((i + 1) % UPLOAD_AUDIO_CHUNK_ELEMENTS) == 0) {
            send_byte(UPLOAD_ACK_MAGIC);
        }
    }
    /* Final ACK so the host knows the STM32 has the full window before DSP. */
    send_byte(UPLOAD_ACK_MAGIC);
}

/* Compute simple audio statistics for the dashboard. */
struct audio_stats {
    uint16_t rms_q15;
    uint16_t peak_q15;
    uint16_t zcr;
};

static void compute_audio_stats(const float *audio, size_t n,
                                struct audio_stats *out)
{
    float sum_sq = 0.0f;
    float peak = 0.0f;
    uint32_t crossings = 0;
    float prev = audio[0];

    for (size_t i = 0; i < n; i++) {
        float v = audio[i];
        sum_sq += v * v;
        float av = v < 0.0f ? -v : v;
        if (av > peak) peak = av;
        if (i > 0) {
            int signed_now = (v >= 0.0f) ? 1 : -1;
            int signed_prev = (prev >= 0.0f) ? 1 : -1;
            if (signed_now != signed_prev) {
                crossings++;
            }
        }
        prev = v;
    }

    float rms = sqrtf(sum_sq / (float)n);
    if (rms > 1.0f) rms = 1.0f;
    if (peak > 1.0f) peak = 1.0f;

    out->rms_q15  = (uint16_t)(rms * 32768.0f + 0.5f);
    out->peak_q15 = (uint16_t)(peak * 32768.0f + 0.5f);
    out->zcr      = (crossings > 0xFFFFu) ? 0xFFFFu : (uint16_t)crossings;
}

static void run_audio_dsp_inference_and_send(const float *audio)
{
    /* 1. Audio statistics (host shows RMS / peak / ZCR for diagnostics). */
    struct audio_stats stats;
    compute_audio_stats(audio, UPLOAD_AUDIO_SAMPLES, &stats);

    /* 2. STFT + mel filterbank + normalisation. */
    uint32_t t_dsp_start = k_uptime_get_32();
    k_mutex_lock(&g_mel_spec_mutex, K_FOREVER);
    mel_spec_compute(audio, UPLOAD_AUDIO_SAMPLES,
                     s_uploaded_spec, MEL_OUT_ROWS, MEL_OUT_COLS);
    k_mutex_unlock(&g_mel_spec_mutex);
    uint32_t dsp_ms = k_uptime_get_32() - t_dsp_start;

    /* 3. INT8 TFLite Micro inference + calibrated unknown gate. */
    uint8_t confidence = 0;
    uint8_t raw_probs[3] = {0};
    uint8_t gate_applied = 0;

    uint32_t t_infer_start = k_uptime_get_32();
    k_mutex_lock(&g_inference_mutex, K_FOREVER);
    int class_id = inference_run_probs(s_uploaded_spec, UPLOAD_SPEC_ELEMENTS,
                                       &confidence, raw_probs, &gate_applied);
    k_mutex_unlock(&g_inference_mutex);
    uint32_t infer_ms = k_uptime_get_32() - t_infer_start;

    uint8_t out_class = (class_id < 0) ? 0xFFu : (uint8_t)class_id;

    ble_client_send(out_class, confidence, k_uptime_get_32());

    /* 4. Extended response packet. */
    send_byte(VALIDATE_MAGIC);
    send_byte(out_class);
    send_byte(confidence);
    send_byte(raw_probs[0]);
    send_byte(raw_probs[1]);
    send_byte(raw_probs[2]);
    send_byte(gate_applied);
    send_u16_le((uint16_t)(dsp_ms > 0xFFFFu ? 0xFFFFu : dsp_ms));
    send_u16_le((uint16_t)(infer_ms > 0xFFFFu ? 0xFFFFu : infer_ms));
    send_u16_le(stats.rms_q15);
    send_u16_le(stats.peak_q15);
    send_u16_le(stats.zcr);
    send_float_array(s_uploaded_spec, UPLOAD_SPEC_ELEMENTS);
}

/* ── Thread entry ─────────────────────────────────────────────────────────── */

void validate_thread_fn(void *p1, void *p2, void *p3)
{
    ARG_UNUSED(p1); ARG_UNUSED(p2); ARG_UNUSED(p3);

    s_console = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
    if (!device_is_ready(s_console)) {
        return;
    }

    /* Install interrupt-driven RX handler on the console UART. */
    uart_irq_callback_user_data_set(s_console, console_rx_isr, NULL);
    uart_irq_rx_enable(s_console);

    while (1) {
        uint8_t cmd = recv_byte();

        if (cmd == CMD_AUDIO_UPLOAD) {
            recv_uploaded_audio(s_uploaded_audio);
            run_audio_dsp_inference_and_send(s_uploaded_audio);
            continue;
        }

        if (cmd == CMD_SYNTH_START) {
            audio_capture_set_enabled(true);
            send_byte(CONTROL_MAGIC);
            send_byte(CMD_SYNTH_START);
            send_byte(1);
            continue;
        }
        if (cmd == CMD_SYNTH_PAUSE) {
            audio_capture_set_enabled(false);
            send_byte(CONTROL_MAGIC);
            send_byte(CMD_SYNTH_PAUSE);
            send_byte(0);
            continue;
        }
        if (cmd != CMD_TEST_VECTOR) {
            continue;
        }

        /* Read 2-byte vector index (little-endian uint16) */
        uint8_t  lo  = recv_byte();
        uint8_t  hi  = recv_byte();
        uint16_t idx = (uint16_t)lo | ((uint16_t)hi << 8);

        if (idx >= TEST_VECTOR_COUNT) {
            send_byte(VALIDATE_MAGIC);
            send_byte(0xFF);   /* class = invalid */
            send_byte(0);
            continue;
        }

        uint8_t confidence = 0;
        k_mutex_lock(&g_inference_mutex, K_FOREVER);
        int class_id = inference_run(test_vectors[idx],
                                     TEST_VECTOR_SAMPLES,
                                     &confidence);
        k_mutex_unlock(&g_inference_mutex);

        uint8_t out_class = (class_id < 0) ? 0xFF : (uint8_t)class_id;
        send_byte(VALIDATE_MAGIC);
        send_byte(out_class);
        send_byte(confidence);
    }
}
