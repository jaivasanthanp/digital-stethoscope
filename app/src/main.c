/*
 * Digital Stethoscope — main.c
 * STM32U575 NUCLEO-U575ZI-Q
 *
 * Spawns four Zephyr threads communicating via message queues:
 *
 *   AudioCaptureThread  (prio 2)  → audio_q
 *   DSPThread           (prio 4)  → spectrogram_q
 *   InferenceThread     (prio 6)  → result_q
 *   CommThread          (prio 8)  → UART TX to nRF52840
 *
 * Current revision: AudioCaptureThread uses synthetic PCG strings on a
 * dashboard-controlled 2-second timer. The STM32U575 still runs DSP and
 * ML inference locally.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include "audio/i2s_capture.h"
#include "dsp/mel_spec.h"
#include "ml/inference.h"
#include "ml/validate.h"
#include "comms/ble_client.h"

LOG_MODULE_REGISTER(main, LOG_LEVEL_INF);

/* -------------------------------------------------------------------------
 * Message queue definitions
 * ------------------------------------------------------------------------- */

/* Audio buffer: 8000 samples × float32 = 32 KB per message
 * Depth 1 is enough for the current synthetic string source. */
#define AUDIO_BUF_SAMPLES  8000
K_MSGQ_DEFINE(audio_q, sizeof(float) * AUDIO_BUF_SAMPLES, 1, 4);

/* Spectrogram: 64×64 float32 = 16 KB per message, depth 1 */
#define SPEC_ROWS  64
#define SPEC_COLS  64
K_MSGQ_DEFINE(spectrogram_q, sizeof(float) * SPEC_ROWS * SPEC_COLS, 1, 4);

/* Result packet */
struct heart_result {
    uint8_t  class_id;       /* 0=Normal 1=SysMurmur 2=DiaMurmur 3=S3Gallop */
    uint8_t  confidence;     /* 0–100 */
    uint32_t timestamp_ms;
};
K_MSGQ_DEFINE(result_q, sizeof(struct heart_result), 4, 4);

/* -------------------------------------------------------------------------
 * Thread stacks
 * ------------------------------------------------------------------------- */
#define AUDIO_STACK     2048
#define DSP_STACK       4096
#define INFER_STACK     8192
#define COMM_STACK      2048
#define VALIDATE_STACK  2048

K_THREAD_STACK_DEFINE(audio_stack,    AUDIO_STACK);
K_THREAD_STACK_DEFINE(dsp_stack,      DSP_STACK);
K_THREAD_STACK_DEFINE(infer_stack,    INFER_STACK);
K_THREAD_STACK_DEFINE(comm_stack,     COMM_STACK);
K_THREAD_STACK_DEFINE(validate_stack, VALIDATE_STACK);

static struct k_thread audio_thread_data;
static struct k_thread dsp_thread_data;
static struct k_thread infer_thread_data;
static struct k_thread comm_thread_data;
static struct k_thread validate_thread_data;

/* -------------------------------------------------------------------------
 * Thread functions
 * ------------------------------------------------------------------------- */

static void audio_capture_thread(void *p1, void *p2, void *p3)
{
    ARG_UNUSED(p1); ARG_UNUSED(p2); ARG_UNUSED(p3);

    static float audio_buf[AUDIO_BUF_SAMPLES];

    LOG_INF("AudioCaptureThread started");

    /* Starts the synthetic source in paused mode. Dashboard sends 'S' to run. */
    audio_capture_init();

    while (1) {
        /* Blocking call: fills audio_buf with one 2-second window */
        audio_capture_get_window(audio_buf, AUDIO_BUF_SAMPLES);

        /* Push to DSP queue; drop oldest if full (prevents audio overrun) */
        if (k_msgq_put(&audio_q, audio_buf, K_NO_WAIT) != 0) {
            LOG_WRN("audio_q full — dropped oldest");
            k_msgq_purge(&audio_q);
            k_msgq_put(&audio_q, audio_buf, K_NO_WAIT);
        }
    }
}

static void dsp_thread(void *p1, void *p2, void *p3)
{
    ARG_UNUSED(p1); ARG_UNUSED(p2); ARG_UNUSED(p3);

    static float audio_in[AUDIO_BUF_SAMPLES];
    static float spec_out[SPEC_ROWS * SPEC_COLS];

    LOG_INF("DSPThread started");

    mel_spec_init();

    while (1) {
        k_msgq_get(&audio_q, audio_in, K_FOREVER);

        mel_spec_compute(audio_in, AUDIO_BUF_SAMPLES, spec_out, SPEC_ROWS, SPEC_COLS);

        if (k_msgq_put(&spectrogram_q, spec_out, K_NO_WAIT) != 0) {
            LOG_WRN("spectrogram_q full — dropped");
        }
    }
}

static void inference_thread(void *p1, void *p2, void *p3)
{
    ARG_UNUSED(p1); ARG_UNUSED(p2); ARG_UNUSED(p3);

    static float spec_in[SPEC_ROWS * SPEC_COLS];
    struct heart_result res;

    LOG_INF("InferenceThread started");

    inference_init();

    static const char *class_names[] = {
        "Normal", "SysMurmur", "DiaMurmur", "S3Gallop"
    };

    while (1) {
        k_msgq_get(&spectrogram_q, spec_in, K_FOREVER);

        uint32_t t0 = k_uptime_get_32();
        k_mutex_lock(&g_inference_mutex, K_FOREVER);
        int class_id = inference_run(spec_in, SPEC_ROWS * SPEC_COLS, &res.confidence);
        k_mutex_unlock(&g_inference_mutex);
        uint32_t latency = k_uptime_get_32() - t0;

        if (class_id < 0) {
            LOG_ERR("Inference failed");
            continue;
        }

        res.class_id     = (uint8_t)class_id;
        res.timestamp_ms = k_uptime_get_32();

        LOG_INF("Class: %-12s  Confidence: %3u%%  Latency: %ums",
                class_names[class_id], res.confidence, latency);

        k_msgq_put(&result_q, &res, K_NO_WAIT);
    }
}

static void comm_thread(void *p1, void *p2, void *p3)
{
    ARG_UNUSED(p1); ARG_UNUSED(p2); ARG_UNUSED(p3);

    struct heart_result res;

    LOG_INF("CommThread started");

    ble_client_init();

    while (1) {
        k_msgq_get(&result_q, &res, K_FOREVER);
        ble_client_send(res.class_id, res.confidence, res.timestamp_ms);
    }
}

/* -------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */

int main(void)
{
    LOG_INF("=== Digital Stethoscope v0.1 ===");
    LOG_INF("Board: nucleo_u575zi_q");

    k_thread_create(&audio_thread_data, audio_stack, AUDIO_STACK,
                    audio_capture_thread, NULL, NULL, NULL,
                    2, 0, K_NO_WAIT);
    k_thread_name_set(&audio_thread_data, "audio_capture");

    k_thread_create(&dsp_thread_data, dsp_stack, DSP_STACK,
                    dsp_thread, NULL, NULL, NULL,
                    4, 0, K_NO_WAIT);
    k_thread_name_set(&dsp_thread_data, "dsp");

    k_thread_create(&infer_thread_data, infer_stack, INFER_STACK,
                    inference_thread, NULL, NULL, NULL,
                    6, 0, K_NO_WAIT);
    k_thread_name_set(&infer_thread_data, "inference");

    k_thread_create(&comm_thread_data, comm_stack, COMM_STACK,
                    comm_thread, NULL, NULL, NULL,
                    8, 0, K_NO_WAIT);
    k_thread_name_set(&comm_thread_data, "comm");

    /* Validate thread: lowest priority — only active during bench testing.
     * Listens on console UART for 'T'+index commands from
     * ml/05_validate_on_device.py and returns inference results. */
    k_thread_create(&validate_thread_data, validate_stack, VALIDATE_STACK,
                    validate_thread_fn, NULL, NULL, NULL,
                    9, 0, K_NO_WAIT);
    k_thread_name_set(&validate_thread_data, "validate");

    return 0;
}
