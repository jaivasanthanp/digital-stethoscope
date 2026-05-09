/*
 * i2s_capture.c - synthetic PCG source
 *
 * The ICS-43434 path is intentionally inactive in this project revision.
 * The STM32U575 receives a compact string that describes a synthetic
 * 2-second heart-sound window, renders it to float32 PCM, and feeds that
 * through the normal DSP + TFLite Micro inference chain.
 */

#include "i2s_capture.h"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

LOG_MODULE_REGISTER(i2s_capture, LOG_LEVEL_INF);

#define AUDIO_BUF_SAMPLES  8000
#define SAMPLE_RATE_HZ     4000.0f
#define SYNTH_PERIOD_MS    2000

/*
 * Synthetic script format:
 *   LABEL|start,duration,frequency,amplitude;start,duration,frequency,amplitude
 *
 * Each event renders a short triangular-windowed tone. This keeps the input as
 * an explicit string while preserving the real on-device DSP and ML path.
 */
static const char *const synth_scripts[] = {
    "NORMAL|0.05,0.10,100,0.40;0.45,0.08,120,0.24;1.05,0.10,100,0.40;1.45,0.08,120,0.24",
    "SYSTOLIC|0.05,0.10,100,0.32;0.14,0.30,260,0.35;0.45,0.08,120,0.20;1.05,0.10,100,0.32;1.14,0.30,260,0.35;1.45,0.08,120,0.20",
    "DIASTOLIC|0.05,0.10,100,0.30;0.45,0.08,120,0.22;0.54,0.44,190,0.35;1.05,0.10,100,0.30;1.45,0.08,120,0.22;1.54,0.40,190,0.35",
    "S3|0.05,0.10,100,0.34;0.45,0.08,120,0.22;0.56,0.08,45,0.35;1.05,0.10,100,0.34;1.45,0.08,120,0.22;1.56,0.08,45,0.35",
};

static K_SEM_DEFINE(audio_ready_sem, 0, 1);
static float synth_buffer[AUDIO_BUF_SAMPLES];
static uint32_t synth_cycle;
static bool synth_enabled;

static void synth_timer_cb(struct k_timer *timer)
{
    ARG_UNUSED(timer);
    k_sem_give(&audio_ready_sem);
}

static K_TIMER_DEFINE(synth_timer, synth_timer_cb, NULL);

static bool is_digit(char c)
{
    return c >= '0' && c <= '9';
}

static float parse_float_field(const char **p)
{
    float value = 0.0f;
    float scale = 1.0f;

    while (is_digit(**p)) {
        value = (value * 10.0f) + (float)(**p - '0');
        (*p)++;
    }

    if (**p == '.') {
        (*p)++;
        while (is_digit(**p)) {
            scale *= 0.1f;
            value += (float)(**p - '0') * scale;
            (*p)++;
        }
    }

    if (**p == ',' || **p == ';') {
        (*p)++;
    }

    return value;
}

static float triangle_wave(float phase)
{
    float frac = phase - (int)phase;

    return (frac < 0.5f) ? (4.0f * frac - 1.0f) : (3.0f - 4.0f * frac);
}

static void render_event(float start_s, float duration_s, float freq_hz, float amp)
{
    int start = (int)(start_s * SAMPLE_RATE_HZ);
    int count = (int)(duration_s * SAMPLE_RATE_HZ);

    if (start < 0) {
        start = 0;
    }
    if (start >= AUDIO_BUF_SAMPLES || count <= 0) {
        return;
    }
    if (start + count > AUDIO_BUF_SAMPLES) {
        count = AUDIO_BUF_SAMPLES - start;
    }

    for (int i = 0; i < count; i++) {
        float pos = (float)i / (float)count;
        float env = (pos < 0.5f) ? (pos * 2.0f) : ((1.0f - pos) * 2.0f);
        float t = (float)(start + i) / SAMPLE_RATE_HZ;

        synth_buffer[start + i] += amp * env * triangle_wave(t * freq_hz);
    }
}

static void fill_synth_buffer_from_script(const char *script)
{
    for (int i = 0; i < AUDIO_BUF_SAMPLES; i++) {
        synth_buffer[i] = 0.0f;
    }

    const char *p = script;
    while (*p != '\0' && *p != '|') {
        p++;
    }
    if (*p == '|') {
        p++;
    }

    while (*p != '\0') {
        float start_s = parse_float_field(&p);
        float duration_s = parse_float_field(&p);
        float freq_hz = parse_float_field(&p);
        float amp = parse_float_field(&p);

        render_event(start_s, duration_s, freq_hz, amp);
    }
}

static void log_script_label(const char *script)
{
    char label[16];
    size_t i = 0;

    while (script[i] != '\0' && script[i] != '|' && i < (sizeof(label) - 1)) {
        label[i] = script[i];
        i++;
    }
    label[i] = '\0';

    LOG_INF("Synthetic input string: %s", label);
}

static void fill_synth_buffer(void)
{
    const char *script = synth_scripts[synth_cycle % ARRAY_SIZE(synth_scripts)];

    fill_synth_buffer_from_script(script);
    log_script_label(script);
    synth_cycle++;
}

void audio_capture_init(void)
{
    LOG_INF("Audio: synthetic string source ready (paused, ICS-43434 disabled)");
    fill_synth_buffer();
    k_timer_start(&synth_timer, K_MSEC(SYNTH_PERIOD_MS), K_MSEC(SYNTH_PERIOD_MS));
}

void audio_capture_set_enabled(bool enabled)
{
    synth_enabled = enabled;

    if (enabled) {
        LOG_INF("Synthetic injection started");
        k_sem_give(&audio_ready_sem);
    } else {
        LOG_INF("Synthetic injection paused");
        k_sem_reset(&audio_ready_sem);
    }
}

bool audio_capture_is_enabled(void)
{
    return synth_enabled;
}

void audio_capture_get_window(float *buf, size_t n_samples)
{
    do {
        k_sem_take(&audio_ready_sem, K_FOREVER);
    } while (!synth_enabled);

    fill_synth_buffer();

    size_t copy = (n_samples < AUDIO_BUF_SAMPLES) ? n_samples : AUDIO_BUF_SAMPLES;
    for (size_t i = 0; i < copy; i++) {
        buf[i] = synth_buffer[i];
    }
}
