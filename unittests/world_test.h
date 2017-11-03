/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#pragma once
#include "gtest/gtest.h"

#include "world.h"
#include "mesh.h"

#include <vector>

class WorldTest : public ::testing::Test {
public:

    void SetUp() override {
    }

    void TearDown() override {
    }

    RadeonRays::World world;
};


TEST_F(WorldTest, CreateWorld) {
}

TEST_F(WorldTest, CreateMesh) {
    std::vector<float> vertices{
        0.f, 0.f, 0.f,
        0.f, 1.f, 0.f,
        1.f, 1.f, 0.f
    };

    std::vector<std::uint32_t> indices{
        0, 1, 2
    };

    ASSERT_NO_THROW(auto mesh = new RadeonRays::Mesh(
        &vertices[0],
        static_cast<std::uint32_t>(vertices.size()),
        0u,
        &indices[0],
        0u,
        1u));
}

