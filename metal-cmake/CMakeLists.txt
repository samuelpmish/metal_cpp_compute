# Library definition
add_library(METAL_CPP_COMPUTE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compute_definition.cpp
)

# Metal cpp headers
target_include_directories(METAL_CPP_COMPUTE PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp-compute"
)

# Metal cpp library (linker)
target_link_libraries(METAL_CPP_COMPUTE
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
    #"-framework MetalKit" # is this necessary?
)
