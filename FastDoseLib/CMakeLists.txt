file(GLOB sources ./src/*.cpp ./src/*.cu)
file(GLOB headers ./include/*.h ./include/*.cuh)

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
    PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
target_include_directories(FastDose PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(FastDose PRIVATE Boost::program_options Boost::filesystem)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info -use_fast_math")
set(CMAKE_BUILD_TYPE Release)