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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bvh_encoder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/qbvh_encoder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bvh_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vk_mem_manager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vk_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/radeonrays.h)

target_link_libraries(RadeonRays PUBLIC Vulkan::Vulkan)