cmake_minimum_required(VERSION 3.10)
project(mediapipe_driver)

set(CMAKE_CXX_STANDARD 17)

# Make sure sdl2 is installed
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# Point to your extracted OpenVR SDK
set(OPENVR_ROOT_DIR "" CACHE PATH "Path to OpenVR SDK")

find_path(OPENVR_INCLUDE_DIR openvr_driver.h
    PATHS ${OPENVR_ROOT_DIR}/headers)
find_library(OPENVR_LIBRARIES
    NAMES openvr_api
    PATHS ${OPENVR_ROOT_DIR}/bin/linux64 ${OPENVR_ROOT_DIR}/lib/linux64)

add_library(driver_mediapipe SHARED driver.cpp)
target_include_directories(driver_mediapipe PRIVATE ${OPENVR_INCLUDE_DIR})
target_link_libraries(driver_mediapipe ${OPENVR_LIBRARIES})
set_target_properties(driver_mediapipe PROPERTIES PREFIX "")
