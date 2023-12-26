file(GLOB MCReferenceSources "./src/*.cpp")
file(GLOB MCReferenceHeaders "./headers/*.h")

add_executable(MCReference ${MCReferenceSources} ${MCReferenceHeaders})
target_include_directories(MCReference PRIVATE ./include)
set_target_properties(MCReference
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR}
)

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
target_include_directories(MCReference PRIVATE ${Boost_INCLUDE_DIR})
target_link_libraries(MCReference PRIVATE ${Boost_LIBRARIES})

option(WITH_GEANT4_UIVIS "Build example with Geant4 UI and Vis drivers" OFF)
if(WITH_GEANT4_UIVIS)
    find_package(Geant4 REQUIRED ui_all vis_all)
else()
    find_package(Geant4 REQUIRED)
endif()

include(${Geant4_USE_FILE})
target_include_directories(MCReference PRIVATE ${Geant4_INCLUDE_DIR})
target_link_libraries(MCReference PRIVATE ${Geant4_LIBRARIES})