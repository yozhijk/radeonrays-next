cmake_minimum_required(VERSION 3.8.0)

project(RadeonRays)

if (MSVC)
    add_compile_options(/W3 /WX)
else()
    add_compile_options(-W -Wall -Werror -std=c++14)
endif()

add_subdirectory(radeonrays)
add_subdirectory(googletest)
add_subdirectory(unittests)
