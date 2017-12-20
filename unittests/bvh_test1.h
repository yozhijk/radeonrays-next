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
#include "bvh_encoder.h"
#include "qbvh_encoder.h"

#include <vector>
#include <stack>
#include <chrono>

class BvhTest1 : public ::testing::Test {
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
        auto id = 1u;
        for (auto& shape : shapes) {
            auto mesh = new RadeonRays::Mesh(
                &vertices[0].x,
                (std::uint32_t)vertices.size(),
                sizeof(RadeonRays::float3),
                (std::uint32_t*)&shape.mesh.indices[0].vertex_index,
                (std::uint32_t)sizeof(tinyobj::index_t),
                (std::uint32_t)(shape.mesh.indices.size() / 3));

            mesh->SetId(id++);
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
    RadeonRays::BVH<RadeonRays::BVHNode, RadeonRays::BVHNodeTraits, RadeonRays::PrimitiveTraits> bvh;
};

#define CORNELL_BOX "../../data/cornellbox.obj"
#define CRYTEK_SPONZA "../../data/sponza.obj"
#define KITCHEN "../../data/kitchen.obj"

TEST_F(BvhTest1, Build) {
    LoadScene(CORNELL_BOX);
    ASSERT_NO_THROW(bvh.Build(world.cbegin(), world.cend()));
    CleanUp();
}


TEST_F(BvhTest1, CrytekSponza) {
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

    CleanUp();
}


TEST_F(BvhTest1, CornellBox_QBVH) {

    using namespace RadeonRays;

    float v = 1.592342394298347298347f;
    float vm = half_to_float(float_to_half_max(v));

    v = 0.592342394298347298347f;
    vm = half_to_float(float_to_half_max(v));

    v = -0.592342394298347298347f;
    vm = half_to_float(float_to_half_max(v));



    LoadScene(CORNELL_BOX);
    ASSERT_NO_THROW(bvh.Build(world.cbegin(), world.cend()));

    RadeonRays::QBVH qbvh;
    ASSERT_NO_THROW(qbvh.Build(world.cbegin(), world.cend()));

    std::stack<std::uint32_t> stack;
    stack.push(0u);

    auto CheckInvariant = [](
        float const* aabb_min,
        float const* aabb_max,
        RadeonRays::QBVHNode const& node
        ) -> bool {
        using namespace RadeonRays;

        if (node.addr0 != RR_INVALID_ID) {
            // unpack here
            float aabb0_min[3];
            float aabb0_max[3];
            float aabb1_min[3];
            float aabb1_max[3];
            float aabb2_min[3];
            float aabb2_max[3];
            float aabb3_min[3];
            float aabb3_max[3];

            copy3unpack_lo(node.aabb01_min_or_v0, aabb0_min);
            copy3unpack_lo(node.aabb01_max_or_v1, aabb0_max);
            copy3unpack_hi(node.aabb01_min_or_v0, aabb1_min);
            copy3unpack_hi(node.aabb01_max_or_v1, aabb1_max);
            copy3unpack_lo(node.aabb23_min_or_v2, aabb2_min);
            copy3unpack_lo(node.aabb23_max, aabb2_max);
            copy3unpack_hi(node.aabb23_min_or_v2, aabb3_min);
            copy3unpack_hi(node.aabb23_max, aabb3_max);


            return aabb_contains_point(aabb_min, aabb_max, aabb0_min) &&
                   aabb_contains_point(aabb_min, aabb_max, aabb0_max) &&
                   (node.addr1_or_mesh_id == RR_INVALID_ID || (aabb_contains_point(aabb_min, aabb_max, aabb1_min) &&
                   aabb_contains_point(aabb_min, aabb_max, aabb1_max))) &&
                   aabb_contains_point(aabb_min, aabb_max, aabb2_min) &&
                   aabb_contains_point(aabb_min, aabb_max, aabb2_max) &&
                   (node.addr3 == RR_INVALID_ID || (aabb_contains_point(aabb_min, aabb_max, aabb3_min) &&
                   aabb_contains_point(aabb_min, aabb_max, aabb3_max)));
        }
        else {
            return aabb_contains_point(aabb_min, aabb_max, (float*)node.aabb01_min_or_v0) &&
                aabb_contains_point(aabb_min, aabb_max, (float*)node.aabb01_max_or_v1) &&
                aabb_contains_point(aabb_min, aabb_max, (float*)node.aabb23_min_or_v2);
        }
    };

    while (!stack.empty()) {
        using namespace RadeonRays;
        auto& node = *qbvh.node(stack.top());
        stack.pop();

        float cmin[3];
        float cmax[3];

        auto& c0 = *qbvh.node(node.addr0);

        copy3unpack_lo(node.aabb01_min_or_v0, cmin);
        copy3unpack_lo(node.aabb01_max_or_v1, cmax);

        ASSERT_TRUE(CheckInvariant(
            cmin,
            cmax,
            c0));

        if (c0.addr0 != RR_INVALID_ID) {
            stack.push(node.addr0);
        }

        if (node.addr1_or_mesh_id != RR_INVALID_ID) {
            auto& c1 = *qbvh.node(node.addr1_or_mesh_id);
            copy3unpack_hi(node.aabb01_min_or_v0, cmin);
            copy3unpack_hi(node.aabb01_max_or_v1, cmax);

            ASSERT_TRUE(CheckInvariant(
                cmin,
                cmax,
                c1));

            if (c1.addr0 != RR_INVALID_ID) {
                stack.push(node.addr1_or_mesh_id);
            }
        }

        auto& c2 = *qbvh.node(node.addr2_or_prim_id);
        copy3unpack_lo(node.aabb23_min_or_v2, cmin);
        copy3unpack_lo(node.aabb23_max, cmax);

        ASSERT_TRUE(CheckInvariant(
            cmin,
            cmax,
            c2));

        if (c2.addr0 != RR_INVALID_ID) {
            stack.push(node.addr2_or_prim_id);
        }

        if (node.addr3 != RR_INVALID_ID) {
            auto& c3 = *qbvh.node(node.addr3);
            copy3unpack_hi(node.aabb23_min_or_v2, cmin);
            copy3unpack_hi(node.aabb23_max, cmax);
            ASSERT_TRUE(CheckInvariant(
                cmin,
                cmax,
                c3));

            if (c3.addr0 != RR_INVALID_ID) {
                stack.push(node.addr3);
            }
        }
    }


    CleanUp();
}

TEST_F(BvhTest1, Sponza_QBVH) {
    LoadScene(CRYTEK_SPONZA);
    ASSERT_NO_THROW(bvh.Build(world.cbegin(), world.cend()));

    RadeonRays::QBVH qbvh;
    ASSERT_NO_THROW(qbvh.Build(world.cbegin(), world.cend()));

    std::stack<std::uint32_t> stack;
    stack.push(0u);

    auto CheckInvariant = [](
        float const* aabb_min,
        float const* aabb_max,
        RadeonRays::QBVHNode const& node
        ) -> bool {
        using namespace RadeonRays;

        if (node.addr0 != RR_INVALID_ID) {
            // unpack here
            float aabb0_min[3];
            float aabb0_max[3];
            float aabb1_min[3];
            float aabb1_max[3];
            float aabb2_min[3];
            float aabb2_max[3];
            float aabb3_min[3];
            float aabb3_max[3];

            copy3unpack_lo(node.aabb01_min_or_v0, aabb0_min);
            copy3unpack_lo(node.aabb01_max_or_v1, aabb0_max);
            copy3unpack_hi(node.aabb01_min_or_v0, aabb1_min);
            copy3unpack_hi(node.aabb01_max_or_v1, aabb1_max);
            copy3unpack_lo(node.aabb23_min_or_v2, aabb2_min);
            copy3unpack_lo(node.aabb23_max, aabb2_max);
            copy3unpack_hi(node.aabb23_min_or_v2, aabb3_min);
            copy3unpack_hi(node.aabb23_max, aabb3_max);


            return aabb_contains_point(aabb_min, aabb_max, aabb0_min) &&
                aabb_contains_point(aabb_min, aabb_max, aabb0_max) &&
                (node.addr1_or_mesh_id == RR_INVALID_ID || (aabb_contains_point(aabb_min, aabb_max, aabb1_min) &&
                    aabb_contains_point(aabb_min, aabb_max, aabb1_max))) &&
                aabb_contains_point(aabb_min, aabb_max, aabb2_min) &&
                aabb_contains_point(aabb_min, aabb_max, aabb2_max) &&
                (node.addr3 == RR_INVALID_ID || (aabb_contains_point(aabb_min, aabb_max, aabb3_min) &&
                    aabb_contains_point(aabb_min, aabb_max, aabb3_max)));
        }
        else {
            // unpack here
            return aabb_contains_point(aabb_min, aabb_max, (float*)node.aabb01_min_or_v0) &&
                aabb_contains_point(aabb_min, aabb_max, (float*)node.aabb01_max_or_v1) &&
                aabb_contains_point(aabb_min, aabb_max, (float*)node.aabb23_min_or_v2);
        }
    };

    while (!stack.empty()) {
        using namespace RadeonRays;
        auto& node = *qbvh.node(stack.top());
        stack.pop();

        float cmin[3];
        float cmax[3];

        auto& c0 = *qbvh.node(node.addr0);

        copy3unpack_lo(node.aabb01_min_or_v0, cmin);
        copy3unpack_lo(node.aabb01_max_or_v1, cmax);

        ASSERT_TRUE(CheckInvariant(
            cmin,
            cmax,
            c0));

        if (c0.addr0 != RR_INVALID_ID) {
            stack.push(node.addr0);
        }

        if (node.addr1_or_mesh_id != RR_INVALID_ID) {
            auto& c1 = *qbvh.node(node.addr1_or_mesh_id);
            copy3unpack_hi(node.aabb01_min_or_v0, cmin);
            copy3unpack_hi(node.aabb01_max_or_v1, cmax);

            ASSERT_TRUE(CheckInvariant(
                cmin,
                cmax,
                c1));

            if (c1.addr0 != RR_INVALID_ID) {
                stack.push(node.addr1_or_mesh_id);
            }
        }

        auto& c2 = *qbvh.node(node.addr2_or_prim_id);
        copy3unpack_lo(node.aabb23_min_or_v2, cmin);
        copy3unpack_lo(node.aabb23_max, cmax);

        ASSERT_TRUE(CheckInvariant(
            cmin,
            cmax,
            c2));

        if (c2.addr0 != RR_INVALID_ID) {
            stack.push(node.addr2_or_prim_id);
        }

        if (node.addr3 != RR_INVALID_ID) {
            auto& c3 = *qbvh.node(node.addr3);
            copy3unpack_hi(node.aabb23_min_or_v2, cmin);
            copy3unpack_hi(node.aabb23_max, cmax);
            ASSERT_TRUE(CheckInvariant(
                cmin,
                cmax,
                c3));

            if (c3.addr0 != RR_INVALID_ID) {
                stack.push(node.addr3);
            }
        }
    }

    CleanUp();
}

