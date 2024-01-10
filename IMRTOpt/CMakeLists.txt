file(GLOB IMRTSources ./src/*.cu ./src/*.cpp)
file(GLOB IMRTHeaders ./include/*.cuh ./include/*.h)

add_executable(IMRT main.cu ${IMRTSources} ${IMRTHeaders})
target_link_libraries(IMRT PRIVATE FastDose)
target_link_libraries(IMRT PRIVATE cusparse)
target_include_directories(IMRT PRIVATE ./include)

find_package(CUDA REQUIRED)
target_include_directories(IMRT PRIVATE ${CUDA_INCLUDE_DIRS})

set_target_properties(IMRT
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR}
)

set_property(TARGET IMRT
    PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH}
)

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
target_include_directories(IMRT PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(IMRT PRIVATE Boost::program_options Boost::filesystem)

find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
target_include_directories(IMRT PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(IMRT PRIVATE 
    ${HDF5_CXX_HL_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_C_HL_LIBRARIES}
    ${HDF5_C_LIBRARIES}
    -lz
)

add_executable(MaskGen MaskGen.cu)
target_include_directories(MaskGen PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(MaskGen PRIVATE Boost::program_options Boost::filesystem)
target_include_directories(MaskGen PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(MaskGen PRIVATE
    ${HDF5_CXX_HL_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_C_HL_LIBRARIES}
    ${HDF5_C_LIBRARIES}
    -lz
)

set_target_properties(MaskGen
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR}
)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)