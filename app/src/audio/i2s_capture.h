#ifndef I2S_CAPTURE_H
#define I2S_CAPTURE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * Initialize the synthetic PCG source.
 *
 * The current project revision has no active ICS-43434 path. The STM32U575
 * receives a compact synthetic heart-sound script, renders it into a 2-second
 * float32 PCM window, and feeds that through the on-device DSP + ML pipeline.
 * A real SAI/I2S microphone source can be added later behind this API.
 */
void audio_capture_init(void);

/*
 * Runtime control used by the serial dashboard.
 * Injection starts paused; audio_capture_set_enabled(true) begins feeding
 * synthetic windows into the DSP/ML pipeline.
 */
void audio_capture_set_enabled(bool enabled);
bool audio_capture_is_enabled(void);

/*
 * Blocking call. Returns when a full 2-second window (8000 samples at 4 kHz)
 * has been collected into buf.
 *
 * buf  : caller-supplied float32 array of length n_samples
 * n_samples : must equal 8000
 */
void audio_capture_get_window(float *buf, size_t n_samples);

#endif /* I2S_CAPTURE_H */
