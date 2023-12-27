file(GLOB boxScoreSources ./src/*.cpp)
file(GLOB boxScoreHeaders ./include/*.h)

add_executable(boxScore main.cpp ${boxScoreSources} ${boxScoreHeaders})

set_target_properties(boxScore
    PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR}
)

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
target_include_directories(boxScore PRIVATE ./include)
target_include_directories(boxScore PRIVATE ${Boost_INCLUDE_DIR})
target_link_libraries(boxScore ${Boost_LIBRARIES})

option(WITH_GEANT4_UIVIS "Build example with Geant4 UI and Vis drivers" ON)
if(WITH_GEANT4_UIVIS)
  find_package(Geant4 REQUIRED ui_all vis_all)
else()
  find_package(Geant4 REQUIRED)
endif()

include(${Geant4_USE_FILE})

target_include_directories(boxScore PRIVATE ${Geant4_INCLUDE_DIR})
target_link_libraries(boxScore ${Geant4_LIBRARIES})