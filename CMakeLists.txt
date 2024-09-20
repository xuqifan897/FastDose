cmake_minimum_required(VERSION 3.27)
project(FastDose LANGUAGES C CXX CUDA)

set(CMAKE_BUILD_TYPE Release)

# Set the binary directory
set(BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${BINARY_DIR})

set(CUDA_ARCH 86)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -use_fast_math")
include_directories(/usr/local/cuda/include)

add_subdirectory(cubeScore)
add_subdirectory(KernelGen)
add_subdirectory(FastDoseLib)
add_subdirectory(FastDoseExample)
add_subdirectory(PlanOptm)
add_subdirectory(IMRTOpt)
add_subdirectory(PreProcess)