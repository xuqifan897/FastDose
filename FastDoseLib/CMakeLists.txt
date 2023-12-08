file(GLOB sources ./src/*.cpp ./src/*.cu)
file(GLOB headers ./include/*.h ./include/*.cuh)

set(CUDA_DTYPE "float")
configure_file(${CMAKE_SOURCE_DIR}/FastDoseLib/include/macros.h.in
    ${CMAKE_SOURCE_DIR}/FastDoseLib/include/macros.h)

add_library(FastDose SHARED ${sources} ${headers})

target_include_directories(FastDose PUBLIC ./include)

set_target_properties(FastDose
    PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${BINARY_DIR}
)

find_package(CUDA REQUIRED)
target_include_directories(FastDose PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(FastDose PRIVATE)
set_property(TARGET FastDose
    PROPERTY CUDA_ARCHITECTURES 89)

# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
set(CMAKE_BUILD_TYPE Release)