add_executable(example argparse.h argparse.cpp init.cuh init.cu main.cu)
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


add_executable(singleBeamBEV singleBeamBEV.cu)
target_link_libraries(singleBeamBEV PRIVATE FastDose)
target_include_directories(singleBeamBEV PRIVATE ${CUDA_INCLUDE_DIRS})
set_target_properties(singleBeamBEV PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR})
set_property(TARGET singleBeamBEV PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
target_include_directories(singleBeamBEV PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(singleBeamBEV PRIVATE Boost::program_options Boost::filesystem)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)