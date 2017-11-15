#include <iostream>
#include <gtest/gtest.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "world_test.h"
#include "bvh_test.h"
#include "lib_test.h"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}