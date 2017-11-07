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
#include "bvh.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <vector>

struct BvhNode {
    RadeonRays::float3 pmin;
    RadeonRays::float3 pmax;
    std::uint32_t refs[4];
    std::uint32_t count;
    std::uint32_t child_base = 0xffffffffu;
};

struct BvhNodeTraits {
    static std::uint32_t constexpr kMaxLeafPrimitives = 4u;
    static std::uint32_t constexpr kMinSAHPrimitives = 16u;
    static std::uint32_t constexpr kTraversalCost = 10u;

    static void EncodeLeaf(BvhNode& node, std::uint32_t num_refs) {
        node.count = num_refs;
    }

    static void EncodeInternal(
        BvhNode& node,
        __m128 aabb_min,
        __m128 aabb_max,
        std::uint32_t child) {
        _mm_store_ps(&node.pmin.x, aabb_min);
        _mm_store_ps(&node.pmax.x, aabb_max);
        node.child_base = child;
    }

    static void SetPrimitive(BvhNode& node, std::uint32_t index, std::uint32_t ref) {
        node.refs[index] = ref;
    }
};

class BvhTest : public ::testing::Test {
public:

    void SetUp() override {
        std::string err;
        auto ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, "../../data/cornellbox.obj");
        ASSERT_TRUE(ret);
        ASSERT_GT(shapes.size(), 0u);

        vertices.resize(attrib.vertices.size() / 3);
        for (auto i = 0u; i < attrib.vertices.size() / 3; ++i) {
            vertices[i].x = attrib.vertices[3 * i];
            vertices[i].y = attrib.vertices[3 * i + 1];
            vertices[i].z = attrib.vertices[3 * i + 2];
            vertices[i].w = 1.f;
        }

        attrib.vertices.clear();
        for (auto& shape : shapes) {
            auto mesh = new RadeonRays::Mesh(
                &vertices[0].x,
                (std::uint32_t)vertices.size(),
                sizeof(RadeonRays::float3),
                (std::uint32_t*)&shape.mesh.indices[0].vertex_index,
                (std::uint32_t)sizeof(tinyobj::index_t),
                (std::uint32_t)(shape.mesh.indices.size() / 3));

            world.AttachShape(mesh);
        }
    }

    void TearDown() override {
    }

    RadeonRays::World world;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::vector<RadeonRays::float3> vertices;
    tinyobj::attrib_t attrib;
    RadeonRays::Bvh<BvhNode, BvhNodeTraits> bvh;
};


TEST_F(BvhTest, BuildBVH) {
    bvh.Build(world.cbegin(), world.cend());
}


