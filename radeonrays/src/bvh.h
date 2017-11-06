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

#include <math/float3.h>

#include <iostream>
#include <memory>
#include <limits>
#include <numeric>
#include <xmmintrin.h>
#include <smmintrin.h>

#include "mesh.h"

namespace RadeonRays {

    struct aligned_allocator {
#ifdef WIN32
        static void* allocate(std::size_t size, std::size_t alignement) {
            return _aligned_malloc(size, alignement);
        }

        static void deallocate(void* ptr) {
            return _aligned_free(ptr);
        }
#else
        static void* allocate(std::size_t size, std::size_t) {
            return malloc(size);
        }

        static void deallocate(void* ptr) {
            return free(ptr);
        }
#endif
    };

#ifdef __GNUC__
#define clz(x) __builtin_clz(x)
#define ctz(x) __builtin_ctz(x)
#else
    inline std::uint32_t popcnt(std::uint32_t x) {
        x -= ((x >> 1) & 0x55555555);
        x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
        x = (((x >> 4) + x) & 0x0f0f0f0f);
        x += (x >> 8);
        x += (x >> 16);
        return x & 0x0000003f;
    }
    inline std::uint32_t clz(std::uint32_t x) {
        x |= (x >> 1);
        x |= (x >> 2);
        x |= (x >> 4);
        x |= (x >> 8);
        x |= (x >> 16);
        return 32 - popcnt(x);
    }
    inline std::uint32_t ctz(std::uint32_t x) {
        return popcnt((std::uint32_t)(x & -(int)x) - 1);
    }
#endif

    inline auto box_surface_area(__m128 pmin, __m128 pmax) {
        __m128 ext = _mm_sub_ps(pmax, pmin);
        __m128 xxy = _mm_shuffle_ps(ext, ext, _MM_SHUFFLE(3, 1, 0, 0));
        __m128 yzz = _mm_shuffle_ps(ext, ext, _MM_SHUFFLE(3, 2, 2, 1));
        return _mm_mul_ps(_mm_dp_ps(xxy, yzz, 0xff), _mm_set_ps(2.f, 2.f, 2.f, 2.f));
    }

    inline auto max_extent_axis(__m128 pmin, __m128 pmax) {
        __m128 xyz = _mm_sub_ps(pmax, pmin);
        __m128 yzx = _mm_shuffle_ps(xyz, xyz, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 m0 = _mm_max_ps(xyz, yzx);
        __m128 m1 = _mm_shuffle_ps(m0, m0, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 m2 = _mm_max_ps(m0, m1);
        __m128 cmp = _mm_cmpeq_ps(xyz, m2);
        return ctz(_mm_movemask_ps(cmp));
    }

    inline auto mm_select(__m128 v, std::uint32_t index) {
        _MM_ALIGN16 float temp[4];
        _mm_store_ps(temp, v);
        return temp[index];
    }

    template <
        typename Node,
        typename Node_traits,
        typename Allocator = aligned_allocator>
    class Bvh {

    public:
        template<typename Iter> void Build(Iter begin, Iter end) {
            auto num_shapes = std::distance(begin, end);

            assert(num_shapes > 0);

            Clear();

            std::size_t num_items = 0;
            for (auto i = begin; i != end; ++i) {
                num_items += static_cast<Mesh const*>((*i))->num_faces();
            }

            auto deleter = [](void* p) {
                Allocator::deallocate(p);
            };

            using aligned_float3_ptr = std::unique_ptr<float3[], decltype(deleter)>;

            auto aabb_min = aligned_float3_ptr(
                reinterpret_cast<float3*>(
                    Allocator::allocate(sizeof(float3) * num_items, 16u)),
                    deleter);

            auto aabb_max = aligned_float3_ptr(
                reinterpret_cast<float3*>(
                    Allocator::allocate(sizeof(float3) * num_items, 16u)),
                    deleter);

            auto aabb_centroid = aligned_float3_ptr(
                reinterpret_cast<float3*>(
                    Allocator::allocate(sizeof(float3) * num_items, 16u)),
                    deleter);

            auto constexpr inf = std::numeric_limits<float>::infinity();

            __m128 scene_min = _mm_set_ps(inf, inf, inf, inf);
            __m128 scene_max = _mm_set_ps(-inf, -inf, -inf, -inf);
            __m128 centroid_scene_min = _mm_set_ps(inf, inf, inf, inf);
            __m128 centroid_scene_max = _mm_set_ps(-inf, -inf, -inf, -inf);

            std::size_t current_face = 0;
            for (auto iter = begin; iter != end; ++iter) {
                auto mesh = static_cast<Mesh const*>(*iter);
                for (std::size_t face_index = 0;
                    face_index < mesh->num_faces();
                    ++face_index, ++current_face) {
                    auto face = mesh->GetIndexData(face_index);

                    __m128 v0 = _mm_load_ps((float*)mesh->GetVertexDataPtr(face.idx[0]));
                    __m128 v1 = _mm_load_ps((float*)mesh->GetVertexDataPtr(face.idx[1]));
                    __m128 v2 = _mm_load_ps((float*)mesh->GetVertexDataPtr(face.idx[2]));

                    __m128 pmin = _mm_min_ps(_mm_min_ps(v0, v1), v2);
                    __m128 pmax = _mm_max_ps(_mm_min_ps(v0, v1), v2);
                    __m128 centroid = _mm_mul_ps(
                        _mm_add_ps(pmin, pmax),
                        _mm_set_ps(0.5f, 0.5f, 0.5f, 0.5f));

                    scene_min = _mm_min_ps(scene_min, pmin);
                    scene_max = _mm_max_ps(scene_max, pmax);

                    centroid_scene_min = _mm_min_ps(centroid_scene_min, centroid);
                    centroid_scene_max = _mm_max_ps(centroid_scene_max, centroid);

                    _mm_store_ps(&aabb_min[current_face].x, pmin);
                    _mm_store_ps(&aabb_max[current_face].x, pmax);
                    _mm_store_ps(&aabb_centroid[current_face].x, centroid);
                }
            }

            BuildImpl(
                scene_min,
                scene_max,
                centroid_scene_min,
                centroid_scene_max,
                aabb_min.get(),
                aabb_max.get(),
                aabb_centroid.get(),
                num_items);
        }

        void Clear() {
            delete nodes_;
            nodes_ = nullptr;
        }

    private:
        static constexpr std::uint32_t kStackSize = 1024u;
        struct SplitRequest {
            __m128 aabb_min;
            __m128 aabb_max;
            __m128 centroid_aabb_min;
            __m128 centroid_aabb_max;
            std::size_t start_index;
            std::size_t num_refs;
            std::uint32_t level;
            std::uint32_t index;
        };

        void BuildImpl(
            __m128 scene_min,
            __m128 scene_max,
            __m128 centroid_scene_min,
            __m128 centroid_scene_max,
            float3 const* aabb_min,
            float3 const* aabb_max,
            float3 const* aabb_centroid,
            std::size_t num_aabbs) {

            std::vector<std::uint32_t> refs(num_aabbs);
            std::iota(refs.begin(), refs.end(), 0);

            _MM_ALIGN16 SplitRequest requests[kStackSize];
            std::uint32_t sptr = 0u;

            auto max_nodes = num_aabbs * 2 - 1;
            nodes_ = reinterpret_cast<Node*>(
                Allocator::allocate(sizeof(Node) * max_nodes, 16u));
            auto free_node_idx = 1;

            requests[sptr++] = SplitRequest {
                scene_min,
                scene_max,
                centroid_scene_min,
                centroid_scene_max,
                0,
                num_aabbs,
                0u,
                0u
            };

            auto constexpr inf = std::numeric_limits<float>::infinity();
            auto m128_plus_inf = _mm_set_ps(inf, inf, inf, inf);
            auto m128_minus_inf = _mm_set_ps(-inf, -inf, -inf, -inf);

            while (sptr > 0) {
                auto request = requests[--sptr];

                if (request.num_refs <= 2u) {
                    // Create leaf here
                    if (request.num_refs == 1u) {
                        //nodes[request.index].SetAsLeaf(
                        //    tri_indices[3 * refs[request.start_index]],
                        //    tri_indices[3 * refs[request.start_index] + 1],
                        //    tri_indices[3 * refs[request.start_index] + 2],
                        //    0u, 0u, 0u,
                        //    1u);
                    }
                    else {
                        //nodes[request.index].SetAsLeaf(
                        //    tri_indices[3 * refs[request.start_index]],
                        //    tri_indices[3 * refs[request.start_index] + 1],
                        //    tri_indices[3 * refs[request.start_index] + 2],
                        //    tri_indices[3 * refs[request.start_index + 1]],
                        //    tri_indices[3 * refs[request.start_index + 1] + 1],
                        //    tri_indices[3 * refs[request.start_index + 1] + 2],
                        //    2u);
                    }
                    continue;
                }

                auto split_axis = max_extent_axis(
                    request.centroid_aabb_min,
                    request.centroid_aabb_max);

                auto split_axis_extent = mm_select(
                    _mm_sub_ps(request.centroid_aabb_max,
                        request.centroid_aabb_min),
                    split_axis);

                auto split_value = mm_select(
                    _mm_mul_ps(
                        _mm_set_ps(0.5f, 0.5f, 0.5f, 0.5),
                        _mm_add_ps(request.centroid_aabb_max,
                            request.centroid_aabb_min)),
                    split_axis);

                std::size_t split_idx = request.start_index;

                __m128 lmin = m128_plus_inf;
                __m128 lmax = m128_minus_inf;
                __m128 rmin = m128_plus_inf;
                __m128 rmax = m128_minus_inf;

                __m128 lcmin = m128_plus_inf;
                __m128 lcmax = m128_minus_inf;
                __m128 rcmin = m128_plus_inf;
                __m128 rcmax = m128_minus_inf;

                // Partition
                if (split_axis_extent > 0.f) {

                    /*split_value = request.num_refs > 16 ?
                    FindSahSplit(
                    request,
                    split_axis,
                    scene_min,
                    scene_max,
                    centroid_scene_min,
                    centroid_scene_max,
                    aabb_min,
                    aabb_max,
                    aabb_centroid,
                    &refs[0]
                    ) : split_value;*/

                    auto first = request.start_index;
                    auto last = request.start_index + request.num_refs;

                    while (1) {
                        while ((first != last) &&
                            aabb_centroid[refs[first]][split_axis] < split_value) {
                            auto idx = refs[first];
                            lmin = _mm_min_ps(lmin, _mm_load_ps(&aabb_min[idx].x));
                            lmax = _mm_max_ps(lmax, _mm_load_ps(&aabb_max[idx].x));

                            __m128 c = _mm_load_ps(&aabb_centroid[idx].x);
                            lcmin = _mm_min_ps(lcmin, c);
                            lcmax = _mm_max_ps(lcmax, c);

                            ++first;
                        }

                        if (first == last--) break;

                        auto idx = refs[first];
                        rmin = _mm_min_ps(rmin, _mm_load_ps(&aabb_min[idx].x));
                        rmax = _mm_max_ps(rmax, _mm_load_ps(&aabb_max[idx].x));

                        __m128 c = _mm_load_ps(&aabb_centroid[idx].x);
                        rcmin = _mm_min_ps(rcmin, c);
                        rcmax = _mm_max_ps(rcmax, c);

                        while ((first != last) &&
                            aabb_centroid[refs[last]][split_axis] >= split_value) {
                            auto idx = refs[last];
                            rmin = _mm_min_ps(rmin, _mm_load_ps(&aabb_min[idx].x));
                            rmax = _mm_max_ps(rmax, _mm_load_ps(&aabb_max[idx].x));

                            __m128 c = _mm_load_ps(&aabb_centroid[idx].x);
                            rcmin = _mm_min_ps(rcmin, c);
                            rcmax = _mm_max_ps(rcmax, c);

                            --last;
                        }

                        if (first == last) break;

                        idx = refs[last];
                        lmin = _mm_min_ps(lmin, _mm_load_ps(&aabb_min[idx].x));
                        lmax = _mm_max_ps(lmax, _mm_load_ps(&aabb_max[idx].x));

                        c = _mm_load_ps(&aabb_centroid[idx].x);
                        lcmin = _mm_min_ps(lcmin, c);
                        lcmax = _mm_max_ps(lcmax, c);

                        std::swap(refs[first++], refs[last]);
                    }

                    split_idx = first;

#ifdef TEST
                    {
                        for (auto i = request.start_index;
                            i < request.start_index + request.num_refs;
                            ++i) {
                            if (i < split_idx) {
                                ASSERT_LT(aabb_centroid[refs[i]][split_axis], split_value);
                            }
                            else {
                                ASSERT_GE(aabb_centroid[refs[i]][split_axis], split_value);
                            }
                        }
                    }
#endif
                }

                if (split_idx == request.start_index ||
                    split_idx == request.start_index + request.num_refs) {
                    split_idx = request.start_index + (request.num_refs >> 1);

                    lmin = m128_plus_inf;
                    lmax = m128_minus_inf;
                    rmin = m128_plus_inf;
                    rmax = m128_minus_inf;

                    lcmin = m128_plus_inf;
                    lcmax = m128_minus_inf;
                    rcmin = m128_plus_inf;
                    rcmax = m128_minus_inf;

                    for (auto i = request.start_index; i < split_idx; ++i) {
                        auto idx = refs[i];
                        lmin = _mm_min_ps(lmin, _mm_load_ps(&aabb_min[idx].x));
                        lmax = _mm_max_ps(lmax, _mm_load_ps(&aabb_max[idx].x));

                        __m128 c = _mm_load_ps(&aabb_centroid[idx].x);
                        lcmin = _mm_min_ps(lcmin, c);
                        lcmax = _mm_max_ps(lcmax, c);
                    }

                    for (auto i = split_idx;
                        i < request.start_index + request.num_refs;
                        ++i) {
                        auto idx = refs[i];
                        rmin = _mm_min_ps(rmin, _mm_load_ps(&aabb_min[idx].x));
                        rmax = _mm_max_ps(rmax, _mm_load_ps(&aabb_max[idx].x));

                        __m128 c = _mm_load_ps(&aabb_centroid[idx].x);
                        rcmin = _mm_min_ps(rcmin, c);
                        rcmax = _mm_max_ps(rcmax, c);
                    }
                }

#ifdef TEST
                {
                    bbox left, right, parent;
                    _mm_store_ps(&left.pmin.x, lmin);
                    _mm_store_ps(&left.pmax.x, lmax);
                    _mm_store_ps(&right.pmin.x, rmin);
                    _mm_store_ps(&right.pmax.x, rmax);
                    _mm_store_ps(&parent.pmin.x, request.aabb_min);
                    _mm_store_ps(&parent.pmax.x, request.aabb_max);

                    ASSERT_TRUE(contains(parent, left));
                    ASSERT_TRUE(contains(parent, right));
                }
#endif

                // Now we have split_idx
                if (sptr == kStackSize) {
                    throw std::runtime_error("Build stack overflow");
                }

                auto& request_left = requests[sptr++];
                request_left.aabb_min = lmin;
                request_left.aabb_max = lmax;
                request_left.centroid_aabb_min = lcmin;
                request_left.centroid_aabb_max = lcmax;
                request_left.start_index = request.start_index;
                request_left.num_refs = split_idx - request.start_index;
                request_left.level = request.level + 1;
                //auto child_base = request_left.index = free_node_idx++;

                if (sptr == kStackSize) {
                    throw std::runtime_error("Build stack overflow");
                }

                auto& request_right = requests[sptr++];
                request_right.aabb_min = rmin;
                request_right.aabb_max = rmax;
                request_right.centroid_aabb_min = rcmin;
                request_right.centroid_aabb_max = rcmax;
                request_right.start_index = split_idx;
                request_right.num_refs = request.num_refs - request_left.num_refs;
                request_right.level = request.level + 1;
                request_right.index = free_node_idx++;

                // Create leaf node
                //nodes[request.index].SetAsInternal(request.aabb_min, request.aabb_max, child_base);
            }
        }

        Node* nodes_ = nullptr;
    };
}
