add_executable(UnitTests main.cpp world_test.h bvh_test.h bvh_test1.h lib_test.h tiny_obj_loader.h stb_image_write.h)
target_link_libraries(UnitTests PUBLIC RadeonRays gtest)