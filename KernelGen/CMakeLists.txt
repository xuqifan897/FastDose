file(GLOB KernelGenSources "./src/*.cpp")
file(GLOB KernelGenHeaders "./include/*.h")

add_executable(KernelGen ${KernelGenSources} ${KernelGenHeaders})
target_include_directories(KernelGen PRIVATE ./include)
set_target_properties(KernelGen
    PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR}
)

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
target_include_directories(KernelGen PRIVATE ${Boost_INCLUDE_DIR})
target_link_libraries(KernelGen ${Boost_LIBRARIES})

option(WITH_GEANT4_UIVIS "Build example with Geant4 UI and Vis drivers" OFF)
if(WITH_GEANT4_UIVIS)
    find_package(Geant4 REQUIRED ui_all vis_all)
else()
    find_package(Geant4 REQUIRED)
endif()

include(${Geant4_USE_FILE})
target_include_directories(KernelGen PRIVATE ${Geant4_INCLUDE_DIR})
target_link_libraries(KernelGen ${Geant4_LIBRARIES})