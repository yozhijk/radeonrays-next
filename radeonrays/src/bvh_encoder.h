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

#include <radeonrays.h>
#include <cstdint>
#include <bvh.h>
#include <shape.h>

namespace RadeonRays
{
    // Encoded node format.
    struct BVHNode
    {
        // Left AABB min or vertex 0 for a leaf node
        float aabb_left_min_or_v0[3] = { 0.f, 0.f, 0.f };
        // Left child node address
        uint32_t addr_left = RR_INVALID_ID;
        // Left AABB max or vertex 1 for a leaf node
        float aabb_left_max_or_v1[3] = { 0.f, 0.f, 0.f };
        // Mesh ID for a leaf node
        uint32_t mesh_id = RR_INVALID_ID;
        // Right AABB min or vertex 2 for a leaf node
        float aabb_right_min_or_v2[3] = { 0.f, 0.f, 0.f };
        // Right child node address
        uint32_t addr_right = RR_INVALID_ID;
        // Left AABB max
        float aabb_right_max[3] = { 0.f, 0.f, 0.f };
        // Primitive ID for a leaf node
        uint32_t prim_id = RR_INVALID_ID;

        static constexpr char const* kTraversalKernelFileName = "isect.comp.spv";
    };

    struct PrimitiveTraits
    {
        using MetadataPtr = Shape const*;

        static std::size_t GetNumAABBs(Shape const* shape)
        {
            auto mesh = static_cast<Mesh const*>(shape);
            return mesh->num_faces();
        }

        static void GetAABB(Shape const* shape,
            std::size_t index,
            __m128& pmin,
            __m128& pmax)
        {
            auto mesh = static_cast<Mesh const*>(shape);
            auto face = mesh->GetIndexData(index);
            auto v0 = _mm_load_ps((float*)mesh->GetVertexDataPtr(face.idx[0]));
            auto v1 = _mm_load_ps((float*)mesh->GetVertexDataPtr(face.idx[1]));
            auto v2 = _mm_load_ps((float*)mesh->GetVertexDataPtr(face.idx[2]));
            pmin = _mm_min_ps(_mm_min_ps(v0, v1), v2);
            pmax = _mm_max_ps(_mm_max_ps(v0, v1), v2);
        }
    };

    // Properties of BVHNode
    struct BVHNodeTraits
    {
        // Max triangles per leaf
        static std::uint32_t constexpr kMaxLeafPrimitives = 1u;
        // Threshold number of primitives to disable SAH split
        static std::uint32_t constexpr kMinSAHPrimitives = 32u;
        // Traversal vs intersection cost ratio
        static std::uint32_t constexpr kTraversalCost = 10u;

        // Create leaf node
        static void EncodeLeaf(BVHNode& node, std::uint32_t num_refs)
        {
            // This node only supports 1 triangle
            assert(num_refs == 1);
            node.addr_left = RR_INVALID_ID;
            node.addr_right = RR_INVALID_ID;
        }

        // Create internal node
        static void EncodeInternal(BVHNode& node,
            __m128 aabb_min,
            __m128 aabb_max,
            std::uint32_t child0,
            std::uint32_t child1)
        {
            _mm_store_ps(node.aabb_left_min_or_v0, aabb_min);
            _mm_store_ps(node.aabb_left_max_or_v1, aabb_max);
            node.addr_left = child0;
            node.addr_right = child1;
        }

        // Add primitive
        static void SetPrimitive(BVHNode& node,
            std::uint32_t index,
            std::pair<Shape const*, std::size_t> ref)
        {
            auto mesh = static_cast<Mesh const*>(ref.first);
            auto vertices = mesh->GetFaceVertexData(ref.second);
            node.aabb_left_min_or_v0[0] = vertices[0].x;
            node.aabb_left_min_or_v0[1] = vertices[0].y;
            node.aabb_left_min_or_v0[2] = vertices[0].z;
            node.aabb_left_max_or_v1[0] = vertices[1].x;
            node.aabb_left_max_or_v1[1] = vertices[1].y;
            node.aabb_left_max_or_v1[2] = vertices[1].z;
            node.aabb_right_min_or_v2[0] = vertices[2].x;
            node.aabb_right_min_or_v2[1] = vertices[2].y;
            node.aabb_right_min_or_v2[2] = vertices[2].z;
            node.mesh_id = mesh->GetId();
            node.prim_id = static_cast<std::uint32_t>(ref.second);
        }

        static bool IsInternal(BVHNode& node)
        {
            return node.addr_left != RR_INVALID_ID;
        }

        static std::uint32_t GetChildIndex(BVHNode& node, std::uint8_t idx)
        {
            return IsInternal(node) ? (idx == 0 ? node.addr_left
                : node.addr_right) : RR_INVALID_ID;
        }

        // We set 1 AABB for each node during BVH build process,
        // however our resulting structure keeps two AABBs for 
        // left and right child nodes in the parent node. To 
        // convert 1 AABB per node -> 2 AABBs for child nodes 
        // we need to traverse the tree pulling child node AABBs 
        // into their parent node. That's exactly what PropagateBounds 
        // is doing.
        static void Finalize(BVH<BVHNode, BVHNodeTraits, PrimitiveTraits>& bvh)
        {
            // Traversal stack
            std::stack<std::uint32_t> s;
            s.push(0);

            while (!s.empty())
            {
                auto idx = s.top();
                s.pop();
                // Fetch the node
                auto node = bvh.node(idx);

                if (IsInternal(*node))
                {
                    // If the node is internal we fetch child nodes
                    auto idx0 = GetChildIndex(*node, 0);
                    auto idx1 = GetChildIndex(*node, 1);

                    auto child0 = bvh.node(idx0);
                    auto child1 = bvh.node(idx1);

                    // If the child is internal node itself we pull it
                    // up the tree into its parent. If the child node is
                    // a leaf, then we do not have AABB for it (we store 
                    // vertices directly in the leaf), so we calculate 
                    // AABB on the fly.
                    if (IsInternal(*child0))
                    {
                        node->aabb_left_min_or_v0[0] = child0->aabb_left_min_or_v0[0];
                        node->aabb_left_min_or_v0[1] = child0->aabb_left_min_or_v0[1];
                        node->aabb_left_min_or_v0[2] = child0->aabb_left_min_or_v0[2];
                        node->aabb_left_max_or_v1[0] = child0->aabb_left_max_or_v1[0];
                        node->aabb_left_max_or_v1[1] = child0->aabb_left_max_or_v1[1];
                        node->aabb_left_max_or_v1[2] = child0->aabb_left_max_or_v1[2];
                        s.push(idx0);
                    }
                    else
                    {
                        node->aabb_left_min_or_v0[0] = min3(
                            child0->aabb_left_min_or_v0[0],
                            child0->aabb_left_max_or_v1[0],
                            child0->aabb_right_min_or_v2[0]);

                        node->aabb_left_min_or_v0[1] = min3(
                            child0->aabb_left_min_or_v0[1],
                            child0->aabb_left_max_or_v1[1],
                            child0->aabb_right_min_or_v2[1]);

                        node->aabb_left_min_or_v0[2] = min3(
                            child0->aabb_left_min_or_v0[2],
                            child0->aabb_left_max_or_v1[2],
                            child0->aabb_right_min_or_v2[2]);

                        node->aabb_left_max_or_v1[0] = max3(
                            child0->aabb_left_min_or_v0[0],
                            child0->aabb_left_max_or_v1[0],
                            child0->aabb_right_min_or_v2[0]);

                        node->aabb_left_max_or_v1[1] = max3(
                            child0->aabb_left_min_or_v0[1],
                            child0->aabb_left_max_or_v1[1],
                            child0->aabb_right_min_or_v2[1]);

                        node->aabb_left_max_or_v1[2] = max3(
                            child0->aabb_left_min_or_v0[2],
                            child0->aabb_left_max_or_v1[2],
                            child0->aabb_right_min_or_v2[2]);
                    }

                    // If the child is internal node itself we pull it
                    // up the tree into its parent. If the child node is
                    // a leaf, then we do not have AABB for it (we store 
                    // vertices directly in the leaf), so we calculate 
                    // AABB on the fly.
                    if (IsInternal(*child1))
                    {
                        node->aabb_right_min_or_v2[0] = child1->aabb_left_min_or_v0[0];
                        node->aabb_right_min_or_v2[1] = child1->aabb_left_min_or_v0[1];
                        node->aabb_right_min_or_v2[2] = child1->aabb_left_min_or_v0[2];
                        node->aabb_right_max[0] = child1->aabb_left_max_or_v1[0];
                        node->aabb_right_max[1] = child1->aabb_left_max_or_v1[1];
                        node->aabb_right_max[2] = child1->aabb_left_max_or_v1[2];
                        s.push(idx1);
                    }
                    else
                    {
                        node->aabb_right_min_or_v2[0] = min3(
                            child1->aabb_left_min_or_v0[0],
                            child1->aabb_left_max_or_v1[0],
                            child1->aabb_right_min_or_v2[0]);

                        node->aabb_right_min_or_v2[1] = min3(
                            child1->aabb_left_min_or_v0[1],
                            child1->aabb_left_max_or_v1[1],
                            child1->aabb_right_min_or_v2[1]);

                        node->aabb_right_min_or_v2[2] = min3(
                            child1->aabb_left_min_or_v0[2],
                            child1->aabb_left_max_or_v1[2],
                            child1->aabb_right_min_or_v2[2]);

                        node->aabb_right_max[0] = max3(
                            child1->aabb_left_min_or_v0[0],
                            child1->aabb_left_max_or_v1[0],
                            child1->aabb_right_min_or_v2[0]);

                        node->aabb_right_max[1] = max3(
                            child1->aabb_left_min_or_v0[1],
                            child1->aabb_left_max_or_v1[1],
                            child1->aabb_right_min_or_v2[1]);

                        node->aabb_right_max[2] = max3(
                            child1->aabb_left_min_or_v0[2],
                            child1->aabb_left_max_or_v1[2],
                            child1->aabb_right_min_or_v2[2]);
                    }
                }
            }
        }
    };
}
