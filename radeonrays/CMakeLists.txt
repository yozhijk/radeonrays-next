find_package(Vulkan)

add_library(RadeonRays
    ${CMAKE_CURRENT_SOURCE_DIR}/src/world.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/radeonrays.cpp)

target_include_directories(RadeonRays PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_include_directories(RadeonRays PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_sources(RadeonRays PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/world.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shape.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mesh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/instance.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bvh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/radeonrays.h)

target_link_libraries(RadeonRays PUBLIC Vulkan::Vulkan)