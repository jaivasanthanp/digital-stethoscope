#ifndef I2S_CAPTURE_H
#define I2S_CAPTURE_H

#include <stdint.h>
#include <stddef.h>

/*
 * Initialize I2S audio capture.
 * In stub mode (CONFIG_AUDIO_STUB=y) this sets up a 2-second timer
 * that replays test vectors from test_vectors.h.
 * In hardware mode it configures the ICS-43434 via the I2S peripheral.
 */
void audio_capture_init(void);

/*
 * Blocking call. Returns when a full 2-second window (8000 samples at 4 kHz)
 * has been collected into buf.
 *
 * buf  : caller-supplied float32 array of length n_samples
 * n_samples : must equal 8000
 */
void audio_capture_get_window(float *buf, size_t n_samples);

#endif /* I2S_CAPTURE_H */
