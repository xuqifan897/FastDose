file(GLOB cubeScoreSources ./src/*.cpp)
file(GLOB cubeScoreHeaders ./include/*.h)
add_executable(cubeScore cubeScoreMain.cpp ${cubeScoreSources} ${cubeScoreHeaders})
target_include_directories(cubeScore PRIVATE ./include)
set_target_properties(cubeScore PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR})

find_package(Geant4 REQUIRED)
target_include_directories(cubeScore PRIVATE ${Geant4_INCLUDE_DIR})
target_link_libraries(cubeScore PRIVATE ${Geant4_LIBRARIES})

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
target_include_directories(cubeScore PRIVATE ${Boost_INCLUDE_DIR})
target_link_libraries(cubeScore PRIVATE ${Boost_LIBRARIES})

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)