add_executable(UnitTests main.cpp world_test.h bvh_test.h lib_test.h)
target_link_libraries(UnitTests PUBLIC RadeonRays gtest)