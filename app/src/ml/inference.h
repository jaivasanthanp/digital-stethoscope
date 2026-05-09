#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include <stddef.h>
#include <zephyr/kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Mutex that serialises inference_run() calls.
 * Both inference_thread and validate_thread must hold this before calling
 * inference_run() — the TFLite Micro interpreter is not thread-safe.
 */
extern struct k_mutex g_inference_mutex;

/*
 * Initialize TFLite Micro interpreter.
 * Must be called once before inference_run().
 * Allocates tensor arena (static buffer, ~95 KB).
 */
void inference_init(void);

/*
 * Run inference on a 64×64 log-mel spectrogram.
 *
 * spec_in      : float32[64*64] row-major, normalized to zero-mean unit-var
 * n_elements   : must be 64*64 = 4096
 * confidence   : output — confidence of returned class, 0–100
 *
 * Returns: class_id (0=Absent, 1=Present, 2=Unknown). A validation-calibrated
 *          uncertainty gate may promote low-margin Absent/Present results to
 *          Unknown.
 *          -1 on error
 */
int inference_run(const float *spec_in, size_t n_elements, uint8_t *confidence);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_H */
