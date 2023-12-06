cmake_minimum_required(VERSION 3.0.0)
project(FastDose LANGUAGES CXX CUDA)

# set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_BUILD_TYPE Release)

# Set the binary directory
set(BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${BINARY_DIR})

add_subdirectory(FastDoseLib)
add_subdirectory(FastDoseExample)