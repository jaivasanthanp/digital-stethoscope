/*
 * i2s_capture.c — ICS-43434 I2S audio capture
 *
 * Phase 4: real I2S DMA capture.
 * Phase 0–3: stub mode — loads test_vectors.h and replays on a 2-second timer.
 *
 * To switch to real I2S, enable CONFIG_I2S in prj.conf and wire the mic.
 * The stub is active when TEST_VECTOR_MODE is defined (see below).
 */

#include "i2s_capture.h"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(i2s_capture, LOG_LEVEL_INF);

/* --------------------------------------------------------------------------
 * Stub mode: replays a hardcoded test vector every 2 seconds.
 * Replace this section with real I2S driver code in Phase 4.
 * -------------------------------------------------------------------------- */
#define AUDIO_BUF_SAMPLES  8000
#define STUB_PERIOD_MS     2000   /* mimic 2-second acquisition window */

/* Semaphore signalled by the stub timer when data is ready */
static K_SEM_DEFINE(audio_ready_sem, 0, 1);
static float stub_buffer[AUDIO_BUF_SAMPLES];

static void stub_timer_cb(struct k_timer *timer)
{
    ARG_UNUSED(timer);
    k_sem_give(&audio_ready_sem);
}

static K_TIMER_DEFINE(stub_timer, stub_timer_cb, NULL);

/*
 * Generates synthetic heartbeat-like patterns cycling through 4 modes.
 * Each 2-second window produces a different spectral profile so the
 * classifier can demonstrate varied outputs during bench testing.
 *
 * Mode 0 (Normal):    60 BPM S1/S2 lubb-dupp — 50 Hz + 100 Hz bursts
 * Mode 1 (Systolic):  murmur between S1-S2    — 200-400 Hz continuous
 * Mode 2 (Diastolic): murmur between S2-S1    — 150-300 Hz continuous
 * Mode 3 (S3 Gallop): extra S3 sound          — 30-60 Hz extra burst
 *
 * Real test vectors from test_vectors.h will replace this in Phase 4
 * once the ICS-43434 mic is wired and real audio is captured.
 */
static uint32_t stub_cycle = 0;

static void fill_stub_buffer(void)
{
    /* Approximate sin(2*pi*f/sr * t) using a phase accumulator */
    /* sin(x) approximated by 4x(pi-x)/pi^2 for 0<x<pi (Bhaskara I formula) */
    uint32_t mode = stub_cycle % 4;

    for (int i = 0; i < AUDIO_BUF_SAMPLES; i++) {
        float t = (float)i / 4000.0f;   /* time in seconds */
        float s = 0.0f;

        switch (mode) {
        case 0:
            /* Normal: 60 BPM S1+S2 pulses at t=0,0.1,0.5,0.6,1.0,1.1,1.5,1.6 */
            {
                float env = 0.0f;
                /* S1 beats */
                float b;
                b = t - 0.05f; if (b >= 0.0f && b < 0.1f) env += 1.0f - b * 10.0f;
                b = t - 1.05f; if (b >= 0.0f && b < 0.1f) env += 1.0f - b * 10.0f;
                /* S2 beats */
                b = t - 0.45f; if (b >= 0.0f && b < 0.08f) env += 0.6f - b * 7.5f;
                b = t - 1.45f; if (b >= 0.0f && b < 0.08f) env += 0.6f - b * 7.5f;
                /* 100 Hz carrier */
                float phase100 = t * 100.0f;
                float frac = phase100 - (int)phase100;
                float sin100 = (frac < 0.5f) ? (4.0f * frac - 1.0f)
                                             : (3.0f - 4.0f * frac);
                s = env * sin100 * 0.4f;
            }
            break;

        case 1:
            /* Systolic murmur: 250 Hz continuous between S1-S2 (0.1-0.45s each beat) */
            {
                float on = 0.0f;
                float b;
                b = t - 0.1f; if (b >= 0.0f && b < 0.35f) on = 1.0f;
                b = t - 1.1f; if (b >= 0.0f && b < 0.35f) on = 1.0f;
                float phase250 = t * 250.0f;
                float frac = phase250 - (int)phase250;
                float sin250 = (frac < 0.5f) ? (4.0f * frac - 1.0f)
                                             : (3.0f - 4.0f * frac);
                s = on * sin250 * 0.35f;
            }
            break;

        case 2:
            /* Diastolic murmur: 200 Hz between S2-S1 (0.5-1.0s and 1.5-2.0s) */
            {
                float on = 0.0f;
                float b;
                b = t - 0.5f; if (b >= 0.0f && b < 0.55f) on = 1.0f;
                b = t - 1.5f; if (b >= 0.0f && b < 0.50f) on = 1.0f;
                float phase200 = t * 200.0f;
                float frac = phase200 - (int)phase200;
                float sin200 = (frac < 0.5f) ? (4.0f * frac - 1.0f)
                                             : (3.0f - 4.0f * frac);
                s = on * sin200 * 0.35f;
            }
            break;

        case 3:
            /* S3 Gallop: extra 40 Hz low thump 0.08s after S2 */
            {
                float env = 0.0f;
                float b;
                /* S1 */
                b = t - 0.05f; if (b >= 0.0f && b < 0.1f) env += 1.0f - b * 10.0f;
                b = t - 1.05f; if (b >= 0.0f && b < 0.1f) env += 1.0f - b * 10.0f;
                /* S2 */
                b = t - 0.45f; if (b >= 0.0f && b < 0.08f) env += 0.6f;
                b = t - 1.45f; if (b >= 0.0f && b < 0.08f) env += 0.6f;
                /* S3 (extra low thump) */
                b = t - 0.55f; if (b >= 0.0f && b < 0.08f) env += 0.4f;
                b = t - 1.55f; if (b >= 0.0f && b < 0.08f) env += 0.4f;
                float phase40 = t * 40.0f;
                float frac = phase40 - (int)phase40;
                float sin40 = (frac < 0.5f) ? (4.0f * frac - 1.0f)
                                            : (3.0f - 4.0f * frac);
                s = env * sin40 * 0.45f;
            }
            break;
        }

        stub_buffer[i] = s;
    }

    stub_cycle++;
}

void audio_capture_init(void)
{
    LOG_INF("Audio: stub mode (real I2S not yet wired)");
    fill_stub_buffer();
    k_timer_start(&stub_timer, K_MSEC(STUB_PERIOD_MS), K_MSEC(STUB_PERIOD_MS));
}

void audio_capture_get_window(float *buf, size_t n_samples)
{
    /* Wait until the 2-second window is ready */
    k_sem_take(&audio_ready_sem, K_FOREVER);

    /* Refresh the stub data each cycle so it doesn't go stale */
    fill_stub_buffer();

    size_t copy = (n_samples < AUDIO_BUF_SAMPLES) ? n_samples : AUDIO_BUF_SAMPLES;
    for (size_t i = 0; i < copy; i++) {
        buf[i] = stub_buffer[i];
    }
}

/* --------------------------------------------------------------------------
 * Phase 4: Real I2S DMA capture (replace stub above with this section)
 *
 * #include <zephyr/drivers/i2s.h>
 * static const struct device *i2s_dev;
 *
 * void audio_capture_init(void) {
 *     i2s_dev = DEVICE_DT_GET(DT_NODELABEL(i2s1));
 *     struct i2s_config cfg = {
 *         .word_size   = 32,
 *         .channels    = 2,
 *         .format      = I2S_FMT_DATA_FORMAT_I2S,
 *         .options     = I2S_OPT_BIT_CLK_MASTER | I2S_OPT_FRAME_CLK_MASTER,
 *         .frame_clk_freq = 4000,
 *         .mem_slab    = &audio_slab,
 *         .block_size  = 512 * sizeof(int32_t) * 2,  // 512-sample stereo block
 *         .timeout     = 2000,
 *     };
 *     i2s_configure(i2s_dev, I2S_DIR_RX, &cfg);
 *     i2s_trigger(i2s_dev, I2S_DIR_RX, I2S_TRIGGER_START);
 * }
 *
 * void audio_capture_get_window(float *buf, size_t n_samples) {
 *     // Accumulate 512-sample blocks into 8000-sample ring buffer
 *     // Extract left channel only (L/R pin tied to GND)
 *     // Convert int32 -> float32, scale by 1/2^31
 * }
 * -------------------------------------------------------------------------- */
