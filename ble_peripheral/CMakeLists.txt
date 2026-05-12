cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(ble_heart_sound)

target_sources(app PRIVATE
    src/main.c
    src/heart_sound_service.c
    src/audio_input_service.c
)

target_include_directories(app PRIVATE src)
