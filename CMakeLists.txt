cmake_minimum_required(VERSION 3.20.0)

# Top-level CMakeLists — delegates to app/ subdirectory.
# To build the STM32U575 application:
#
#   cd app
#   west build -b nucleo_u575zi_q .
#
# Or from the zephyrproject workspace:
#   west build -b nucleo_u575zi_q <path-to-this-repo>/app

message(STATUS "Digital Stethoscope — top-level CMakeLists")
message(STATUS "  STM32 app : app/")
message(STATUS "  BLE app   : ble_peripheral/")
message(STATUS "  ML pipeline: ml/  (Python, run on host)")
