file(GLOB PreProcessSources ./src/*.cpp ./src/*.cu)
file(GLOB PreProcessHeaders ./include/*.h ./include/*.cuh)

add_executable(preprocess main.cpp ${PreProcessSources} ${PreProcessHeaders})
set_target_properties(preprocess
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR}
)
target_include_directories(preprocess PRIVATE ./include)

target_include_directories(preprocess PRIVATE ${CUDA_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
target_include_directories(preprocess PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(preprocess PRIVATE Boost::program_options Boost::filesystem)

find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
target_include_directories(preprocess PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(preprocess PRIVATE 
    ${HDF5_CXX_HL_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_C_HL_LIBRARIES}
    ${HDF5_C_LIBRARIES}
    -lz
)

find_package(RapidJSON REQUIRED)
target_include_directories(preprocess PRIVATE ${RapidJSON_INCLUDE_DIRS})
target_link_libraries(preprocess PRIVATE ${RapidJSON_LIBRARIES})

find_package(DCMTK REQUIRED)
target_include_directories(preprocess PRIVATE ${DCMTK_INCLUDE_DIRS})
target_link_libraries(preprocess PRIVATE ${DCMTK_LIBRARIES})

find_package(OpenCV REQUIRED)
target_include_directories(preprocess PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(preprocess PRIVATE ${OpenCV_LIBS})

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)