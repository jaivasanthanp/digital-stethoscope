cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(digital_stethoscope)

# Core application sources
target_sources(app PRIVATE
    src/main.c
)

# Audio capture (I2S / stub)
target_sources(app PRIVATE
    src/audio/i2s_capture.c
)

# DSP chain — STFT + mel filterbank
target_sources(app PRIVATE
    src/dsp/mel_spec.c
)

# TFLite Micro inference wrapper (C++ file) + on-device validation
target_sources(app PRIVATE
    src/ml/inference.cc
    src/ml/model_data.cc
    src/ml/validate.c
)

# BLE/UART communication to nRF52840
target_sources(app PRIVATE
    src/comms/ble_client.c
)

target_include_directories(app PRIVATE
    src
    src/audio
    src/dsp
    src/ml
    src/comms
)
