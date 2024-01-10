file(GLOB files ./*.h ./*.cpp ./*.cu)
add_executable(example ${files})
target_link_libraries(example 
    PRIVATE
    FastDose
)

find_package(CUDA REQUIRED)
target_include_directories(example PRIVATE ${CUDA_INCLUDE_DIRS})

set_target_properties(example
    PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR}
)

set_property(TARGET example 
    PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH}
)

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
target_include_directories(example PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(example PRIVATE Boost::program_options Boost::filesystem)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)