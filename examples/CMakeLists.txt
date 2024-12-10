# Get all project dir
FILE(GLOB examples ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# For each source file, make an executable
FOREACH(example ${examples})

    # Get project name and all sources
    get_filename_component(example-name ${example} NAME_WE)

    # Create executable and link target
    add_executable(${example-name} ${example})
    target_link_libraries(${example-name} METAL_CPP_COMPUTE)

    message(STATUS "adding target: ${example}")

ENDFOREACH()
