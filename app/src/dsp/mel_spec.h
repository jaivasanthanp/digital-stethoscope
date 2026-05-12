#ifndef MEL_SPEC_H
#define MEL_SPEC_H

#include <stdint.h>
#include <stddef.h>
#include <zephyr/kernel.h>

/*
 * mel_spec_compute() uses shared static DSP work buffers. Hold this mutex
 * around any call to mel_spec_compute() that may race with other threads.
 */
extern struct k_mutex g_mel_spec_mutex;

/*
 * DSP parameters — must match the Python preprocessing pipeline exactly.
 * Any change here requires regenerating mel_filterbank_weights.h.
 */
#define MEL_SR          4000    /* sample rate Hz */
#define MEL_N_FFT       512     /* FFT size */
#define MEL_HOP         128     /* hop length in samples (32 ms) */
#define MEL_N_MELS      64      /* mel filterbank bins */
#define MEL_N_FRAMES    62      /* floor((8000 - 512) / 128) + 1 */
#define MEL_OUT_ROWS    64      /* output height (padded/cropped to square) */
#define MEL_OUT_COLS    64      /* output width */

/*
 * Initialize mel spectrogram processing.
 * Loads precomputed Hann window and mel filterbank weights from
 * mel_filterbank_weights.h into internal buffers.
 */
void mel_spec_init(void);

/*
 * Compute a 64×64 log-mel spectrogram from raw PCM audio.
 *
 * pcm_in    : float32 audio samples, length n_samples (should be 8000)
 * n_samples : number of samples
 * spec_out  : output buffer, row-major float32[out_rows × out_cols]
 * out_rows  : MEL_OUT_ROWS (64)
 * out_cols  : MEL_OUT_COLS (64)
 *
 * Processing steps:
 *   1. Frame signal with Hann window (512-sample frames, 128-sample hop)
 *   2. arm_rfft_fast_f32 → 257 power bins per frame
 *   3. Mel filterbank matrix multiply (64×257 weights)
 *   4. log10 compression
 *   5. Pad/crop time axis to 64 frames → 64×64
 *   6. Per-spectrogram zero-mean, unit-variance normalization
 */
void mel_spec_compute(const float *pcm_in, size_t n_samples,
                      float *spec_out, int out_rows, int out_cols);

#endif /* MEL_SPEC_H */
