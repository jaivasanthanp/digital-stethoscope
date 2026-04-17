/*
 * mel_spec.c — STFT + Mel filterbank + log compression
 *
 * Pipeline (matches ml/01_preprocess.py exactly):
 *   1. Frame with Hann window (512-sample, 128-sample hop) → 62 frames
 *   2. arm_rfft_fast_f32 → 257 power spectrum bins per frame
 *   3. Mel filterbank matrix multiply (64×257, from mel_filterbank_weights.h)
 *   4. log10 compression (with floor at 1e-10 to avoid log(0))
 *   5. Crop/pad time axis: 62 frames → 64 columns (pad last 2 with zeros)
 *   6. Per-spectrogram zero-mean unit-variance normalisation
 *
 * Output: float32[64][64] row-major, mel bin = row, time frame = column.
 */

#include "mel_spec.h"
#include "mel_filterbank_weights.h"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <arm_math.h>

#include <string.h>
#include <math.h>

LOG_MODULE_REGISTER(mel_spec, LOG_LEVEL_INF);

/* -------------------------------------------------------------------------
 * Internal buffers (static — no heap allocation at runtime)
 * ------------------------------------------------------------------------- */
#define FFT_SIZE     MEL_N_FFT          /* 512 */
#define FFT_BINS     (FFT_SIZE / 2 + 1) /* 257 */
#define N_FRAMES     MEL_N_FRAMES       /* 62 */

static arm_rfft_fast_instance_f32 rfft_inst;

/* Precomputed Hann window: w[n] = 0.5*(1 - cos(2πn/(N-1))) */
static float hann_window[FFT_SIZE];

/* Intermediate buffers */
static float frame_buf[FFT_SIZE];       /* one windowed frame */
static float fft_out[FFT_SIZE];         /* FFT complex output */
static float power_spec[FFT_BINS];      /* power spectrum of one frame */
/* mel_frame used in Phase 3 DSP integration — declared here for clarity */

/* Output accumulator before padding/normalization */
static float mel_raw[MEL_N_MELS][N_FRAMES];

/* -------------------------------------------------------------------------
 * Hann window initialisation
 * ------------------------------------------------------------------------- */
static void init_hann_window(void)
{
    for (int n = 0; n < FFT_SIZE; n++) {
        /* arm_cos_f32(x) returns float — no output pointer */
        float cos_val = arm_cos_f32((2.0f * 3.14159265358979f * n) / (FFT_SIZE - 1));
        hann_window[n] = 0.5f * (1.0f - cos_val);
    }
}

/* -------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */
void mel_spec_init(void)
{
    arm_rfft_fast_init_f32(&rfft_inst, FFT_SIZE);
    init_hann_window();
    LOG_INF("mel_spec: init OK (FFT=%d, mels=%d, frames=%d)",
            FFT_SIZE, MEL_N_MELS, N_FRAMES);
}

void mel_spec_compute(const float *pcm_in, size_t n_samples,
                      float *spec_out, int out_rows, int out_cols)
{
    ARG_UNUSED(n_samples);

    /* --- Step 1–4: STFT + mel + log ------------------------------------ */
    for (int frame = 0; frame < N_FRAMES; frame++) {
        int offset = frame * MEL_HOP;

        /* Apply Hann window */
        for (int n = 0; n < FFT_SIZE; n++) {
            frame_buf[n] = pcm_in[offset + n] * hann_window[n];
        }

        /* FFT */
        arm_rfft_fast_f32(&rfft_inst, frame_buf, fft_out, 0);

        /* Power spectrum: |X[k]|^2 */
        /* arm_rfft_fast_f32 output layout:
         *   fft_out[0]  = DC  (real only)
         *   fft_out[1]  = Nyquist (real only)
         *   fft_out[2k] = real part of bin k
         *   fft_out[2k+1] = imag part of bin k
         */
        power_spec[0] = fft_out[0] * fft_out[0];
        power_spec[FFT_BINS - 1] = fft_out[1] * fft_out[1];
        for (int k = 1; k < FFT_BINS - 1; k++) {
            float re = fft_out[2 * k];
            float im = fft_out[2 * k + 1];
            power_spec[k] = re * re + im * im;
        }

        /* Mel filterbank multiply: mel_frame[m] = sum_k(filterbank[m][k] * power[k]) */
        for (int m = 0; m < MEL_N_MELS; m++) {
            float sum = 0.0f;
            const float *row = mel_filterbank[m];
            arm_dot_prod_f32(row, power_spec, FFT_BINS, &sum);
            /* log10 compression with floor */
            mel_raw[m][frame] = log10f(sum > 1e-10f ? sum : 1e-10f);
        }
    }

    /* --- Step 5: Pad time axis 62 → 64 --------------------------------- */
    /* spec_out is row-major [out_rows][out_cols] = [64][64]               */
    for (int m = 0; m < MEL_N_MELS; m++) {
        for (int t = 0; t < out_cols; t++) {
            float val = (t < N_FRAMES) ? mel_raw[m][t] : 0.0f;
            spec_out[m * out_cols + t] = val;
        }
    }

    /* --- Step 6: Per-spectrogram zero-mean unit-variance normalisation -- */
    float mean = 0.0f;
    arm_mean_f32(spec_out, (uint32_t)(out_rows * out_cols), &mean);

    /* Variance */
    float var = 0.0f;
    for (int i = 0; i < out_rows * out_cols; i++) {
        float d = spec_out[i] - mean;
        var += d * d;
    }
    var /= (float)(out_rows * out_cols);
    float std = sqrtf(var > 1e-10f ? var : 1e-10f);

    for (int i = 0; i < out_rows * out_cols; i++) {
        spec_out[i] = (spec_out[i] - mean) / std;
    }
}
