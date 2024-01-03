file(GLOB PlanOptmSources ./*.cu ./*.cpp)
file(GLOB PlanOptmHeaders ./*.cuh ./*.h)

add_executable(PlanOptm ${PlanOptmSources} ${PlanOptmHeaders})
target_link_libraries(PlanOptm PRIVATE FastDose)
target_link_libraries(PlanOptm PRIVATE cusparse)

find_package(CUDA REQUIRED)
target_include_directories(PlanOptm PRIVATE ${CUDA_INCLUDE_DIRS})

set_target_properties(PlanOptm
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR}
)

set_property(TARGET PlanOptm
    PROPERTY CUDA_ARCHITECTURES 89
)

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
target_include_directories(PlanOptm PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(PlanOptm PRIVATE Boost::program_options Boost::filesystem)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)