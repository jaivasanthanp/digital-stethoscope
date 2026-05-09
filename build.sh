#!/usr/bin/env bash
# build.sh — Quick build helper for Digital Stethoscope
#
# Tested environment:
#   Zephyr 4.4.0-rc1 workspace : C:/Users/jaiva/zephyrproject/zephyr
#   Zephyr SDK 1.0.1           : Desktop/zephyr/zephyr-sdk-1.0.1_windows-x86_64_gnu/
#   Python 3.14 (required for Zephyr 4.4+)  : AppData/Local/Programs/Python/Python314
#   Ninja                      : C:/ProgramData/chocolatey/bin (Chocolatey)
#
# Usage:
#   ./build.sh              — build STM32 app (nucleo_u575zi_q)
#   ./build.sh --ble        — build nRF52840 BLE peripheral
#   ./build.sh --clean      — pristine rebuild (full reconfigure)

set -e

ZEPHYR_BASE_PATH="C:/Users/jaiva/zephyrproject/zephyr"
SDK_PATH="C:/Users/jaiva/Desktop/zephyr/zephyr-sdk-1.0.1_windows-x86_64_gnu/zephyr-sdk-1.0.1"
PYTHON314="C:/Users/jaiva/AppData/Local/Programs/Python/Python314/python.exe"
ARM_GCC="$SDK_PATH/gnu/arm-zephyr-eabi/bin"
CHOCO_BIN="/c/ProgramData/chocolatey/bin"

export ZEPHYR_BASE="$ZEPHYR_BASE_PATH"
export ZEPHYR_SDK_INSTALL_DIR="$SDK_PATH"
export PATH="$CHOCO_BIN:$ARM_GCC:$PATH"

echo "Toolchain : $(arm-zephyr-eabi-gcc --version | head -1)"
echo "Zephyr    : $ZEPHYR_BASE"

BLE_MODE=0
PRISTINE=""
EXTRA_ARGS=()

for arg in "$@"; do
    case $arg in
        --ble)   BLE_MODE=1 ;;
        --clean) PRISTINE="--pristine" ;;
    esac
done

if [ "$BLE_MODE" -eq 1 ]; then
    APP_DIR="ble_peripheral"
    BOARD="nrf52840dk_nrf52840"
    BUILD_DIR="build_ble"
else
    APP_DIR="app"
    BOARD="nucleo_u575zi_q"
    BUILD_DIR="build_stm32_synth"
fi

echo "Building  : $APP_DIR  (board: $BOARD)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

west build $PRISTINE \
    --board "$BOARD" \
    --build-dir "$BUILD_DIR" \
    "$APP_DIR" \
    -- "-DPython3_EXECUTABLE=$PYTHON314"

echo ""
echo "Build complete:"
echo "  Binary : $BUILD_DIR/zephyr/zephyr.elf"
if [ "$BLE_MODE" -eq 1 ]; then
    echo "  Flash  : west flash --build-dir $BUILD_DIR"
else
    echo "  Flash  : west flash --build-dir $BUILD_DIR --runner openocd"
fi
echo "  Debug  : west debug --build-dir $BUILD_DIR"
