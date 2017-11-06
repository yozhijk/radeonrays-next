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
        0.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        1.f, 1.f, 0.f, 0.f
    };

    std::vector<std::uint32_t> indices{
        0, 1, 2
    };

    RadeonRays::Mesh* mesh = nullptr;
    ASSERT_NO_THROW(mesh = new RadeonRays::Mesh(
        &vertices[0],
        static_cast<std::uint32_t>(vertices.size()),
        0u,
        &indices[0],
        0u,
        1u));

    delete mesh;
}

TEST_F(WorldTest, MeshData) {
    std::vector<float> vertices{
        0.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        1.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f,
        0.f, 2.f, 2.f, 0.f,
        2.f, 2.f, 0.f, 0.f,
    };

    std::vector<std::uint32_t> indices{
        0, 1, 2, 3, 4, 5
    };

    RadeonRays::Mesh* mesh = nullptr;
    ASSERT_NO_THROW(mesh = new RadeonRays::Mesh(
        &vertices[0],
        static_cast<std::uint32_t>(vertices.size()),
        0u,
        &indices[0],
        0u,
        2u));

    auto v = mesh->GetVertexData();

    ASSERT_EQ(v[0].x, 0.f);
    ASSERT_EQ(v[1].y, 1.f);
    ASSERT_EQ(v[2].z, 0.f);


    auto v4 = mesh->GetVertexData(4);

    ASSERT_EQ(v4.x, 0.f);
    ASSERT_EQ(v4.y, 2.f);
    ASSERT_EQ(v4.z, 2.f);

    auto i = mesh->GetIndexData(1);

    ASSERT_EQ(i.idx[0], 3);
    ASSERT_EQ(i.idx[1], 4);
    ASSERT_EQ(i.idx[2], 5);

    delete mesh;
}

TEST_F(WorldTest, MeshBounds) {
    std::vector<float> vertices{
        0.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        1.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f,
        0.f, 2.f, 2.f, 0.f,
        2.f, 2.f, 0.f, 0.f,
    };

    std::vector<std::uint32_t> indices{
        0, 1, 2, 3, 4, 5
    };

    RadeonRays::Mesh* mesh = nullptr;
    ASSERT_NO_THROW(mesh = new RadeonRays::Mesh(
        &vertices[0],
        static_cast<std::uint32_t>(vertices.size()),
        0u,
        &indices[0],
        0u,
        2u));

    auto bounds = mesh->GetBounds();

    ASSERT_EQ(bounds.pmin.x, 0.f);
    ASSERT_EQ(bounds.pmin.y, 0.f);
    ASSERT_EQ(bounds.pmin.z, 0.f);

    ASSERT_EQ(bounds.pmax.x, 2.f);
    ASSERT_EQ(bounds.pmax.y, 2.f);
    ASSERT_EQ(bounds.pmax.z, 2.f);

    mesh->SetTransform(RadeonRays::translation(RadeonRays::float3(1.f, 1.f, 0.f)));

    bounds = mesh->GetBounds();

    ASSERT_EQ(bounds.pmin.x, 1.f);
    ASSERT_EQ(bounds.pmin.y, 1.f);
    ASSERT_EQ(bounds.pmin.z, 0.f);

    ASSERT_EQ(bounds.pmax.x, 3.f);
    ASSERT_EQ(bounds.pmax.y, 3.f);
    ASSERT_EQ(bounds.pmax.z, 2.f);

    delete mesh;
}