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

#include <vector>
#include <stack>
#include <chrono>

struct BvhNode {
    static auto constexpr kInvalidIndex = 0xffffffffu;

    RadeonRays::bbox bounds;
    RadeonRays::Mesh const* mesh;
    std::uint32_t face_index = kInvalidIndex;
    std::uint32_t child[2]{ kInvalidIndex, kInvalidIndex };
};

struct BvhNodeTraits {
    static std::uint32_t constexpr kMaxLeafPrimitives = 4u;
    static std::uint32_t constexpr kMinSAHPrimitives = 64u;
    static std::uint32_t constexpr kTraversalCost = 10u;

    static void EncodeLeaf(BvhNode& node, std::uint32_t num_refs) {
    }

    static void EncodeInternal(
        BvhNode& node,
        __m128 aabb_min,
        __m128 aabb_max,
        std::uint32_t child0,
        std::uint32_t child1
        ) {
        _mm_store_ps(&node.bounds.pmin.x, aabb_min);
        _mm_store_ps(&node.bounds.pmax.x, aabb_max);
        node.child[0] = child0;
        node.child[1] = child1;
    }

    static void SetPrimitive(
        BvhNode& node,
        std::uint32_t index,
        std::pair<RadeonRays::Mesh const*, std::size_t> ref) {
        node.mesh = ref.first;
        node.face_index = (std::uint32_t)ref.second;
    }

    static bool IsInternal(BvhNode& node) {
        return node.child[0] != 0xffffffffu;
    }

    static std::uint32_t GetChildIndex(BvhNode& node, std::uint8_t idx) {
        return IsInternal(node) ? (node.child[idx]) : 0xffffffffu;
    }
};

class BvhTest : public ::testing::Test {
public:

    void LoadScene(std::string const& file) {
        std::string err;
        auto ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, file.c_str());
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

    void CleanUp() {
        for (auto iter = world.cbegin(); iter != world.cend(); ++iter) {
            delete *iter;
        }

        world.DetachAll();
        shapes.clear();
        materials.clear();
        vertices.clear();
        attrib = tinyobj::attrib_t{};
        bvh.Clear();
    }

    void SetUp() override {
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

#define CORNELL_BOX "../../data/cornellbox.obj"
#define CRYTEK_SPONZA "../../data/sponza.obj"
#define KITCHEN "../../data/kitchen.obj"

TEST_F(BvhTest, Build) {
    LoadScene(CORNELL_BOX);
    ASSERT_NO_THROW(bvh.Build(world.cbegin(), world.cend()));
    CleanUp();
}

TEST_F(BvhTest, Invariant) {
    LoadScene(CORNELL_BOX);
    ASSERT_NO_THROW(bvh.Build(world.cbegin(), world.cend()));

    std::stack<std::uint32_t> stack;
    stack.push(0u);

    while (!stack.empty()) {
        auto node = bvh.GetNode(stack.top());
        stack.pop();

        if (BvhNodeTraits::IsInternal(*node)) {
            for (auto i = 0u; i < 2u; ++i) {
                auto child = bvh.GetNode(BvhNodeTraits::GetChildIndex(*node, i));

                if (BvhNodeTraits::IsInternal(*child)) {
                    ASSERT_TRUE(RadeonRays::contains(node->bounds, child->bounds));
                }
                else {
                    auto mesh = child->mesh;
                    auto face_index = child->face_index;
                    auto vertices = mesh->GetFaceVertexData(face_index);

                    for (auto& v : vertices) {
                        ASSERT_TRUE(RadeonRays::contains(node->bounds, v));
                    }
                }
            }
        }
    }

    CleanUp();
}

TEST_F(BvhTest, CrytekSponza) {
    using namespace std::chrono;
    LoadScene(CRYTEK_SPONZA);
    auto start = high_resolution_clock::now();
    ASSERT_NO_THROW(bvh.Build(world.cbegin(), world.cend()));
    auto delta = high_resolution_clock::now() - start;
    auto delta_in_ms = duration_cast<milliseconds>(delta).count();

#ifndef _DEBUG
    std::size_t num_primitives = 0;
    for (auto& iter = world.cbegin(); iter != world.cend(); ++iter) {
        num_primitives += static_cast<RadeonRays::Mesh const*>(*iter)->num_faces();
    }

    std::cout << "Num primitives: " << num_primitives << "\n";
    std::cout << "Time spent: " << delta_in_ms << " ms.\n";
    std::cout << "Throughput: " << (float)num_primitives / (delta_in_ms / 1000.f) / 1000000.f << " Mprims/s.\n";
#endif

    std::stack<std::uint32_t> stack;
    stack.push(0u);

    while (!stack.empty()) {
        auto node = bvh.GetNode(stack.top());
        stack.pop();

        if (BvhNodeTraits::IsInternal(*node)) {
            for (auto i = 0u; i < 2u; ++i) {
                auto child = bvh.GetNode(BvhNodeTraits::GetChildIndex(*node, i));

                if (BvhNodeTraits::IsInternal(*child)) {
                    ASSERT_TRUE(RadeonRays::contains(node->bounds, child->bounds));
                }
                else {
                    auto mesh = child->mesh;
                    auto face_index = child->face_index;
                    auto vertices = mesh->GetFaceVertexData(face_index);

                    for (auto& v : vertices) {
                        ASSERT_TRUE(RadeonRays::contains(node->bounds, v));
                    }
                }
            }
        }
    }

    CleanUp();
}

TEST_F(BvhTest, Kitchen) {
    using namespace std::chrono;
    LoadScene(KITCHEN);
    auto start = high_resolution_clock::now();
    ASSERT_NO_THROW(bvh.Build(world.cbegin(), world.cend()));
    auto delta = high_resolution_clock::now() - start;
    auto delta_in_ms = duration_cast<milliseconds>(delta).count();

#ifndef _DEBUG
    std::size_t num_primitives = 0;
    for (auto& iter = world.cbegin(); iter != world.cend(); ++iter) {
        num_primitives += static_cast<RadeonRays::Mesh const*>(*iter)->num_faces();
    }

    std::cout << "Num primitives: " << num_primitives << "\n";
    std::cout << "Time spent: " << delta_in_ms << " ms.\n";
    std::cout << "Throughput: " << (float)num_primitives / (delta_in_ms / 1000.f) / 1000000.f << " Mprims/s.\n";
#endif

    std::stack<std::uint32_t> stack;
    stack.push(0u);

    while (!stack.empty()) {
        auto node = bvh.GetNode(stack.top());
        stack.pop();

        if (BvhNodeTraits::IsInternal(*node)) {
            for (auto i = 0u; i < 2u; ++i) {
                auto child = bvh.GetNode(BvhNodeTraits::GetChildIndex(*node, i));

                if (BvhNodeTraits::IsInternal(*child)) {
                    ASSERT_TRUE(RadeonRays::contains(node->bounds, child->bounds));
                }
                else {
                    auto mesh = child->mesh;
                    auto face_index = child->face_index;
                    auto vertices = mesh->GetFaceVertexData(face_index);

                    for (auto& v : vertices) {
                        ASSERT_TRUE(RadeonRays::contains(node->bounds, v));
                    }
                }
            }
        }
    }

    CleanUp();
}


