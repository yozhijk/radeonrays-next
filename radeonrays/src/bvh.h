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
#include <chrono>
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

    inline auto aabb_surface_area(__m128 pmin, __m128 pmax) {
        auto ext = _mm_sub_ps(pmax, pmin);
        auto xxy = _mm_shuffle_ps(ext, ext, _MM_SHUFFLE(3, 1, 0, 0));
        auto yzz = _mm_shuffle_ps(ext, ext, _MM_SHUFFLE(3, 2, 2, 1));
        return _mm_mul_ps(_mm_dp_ps(xxy, yzz, 0xff), _mm_set_ps(2.f, 2.f, 2.f, 2.f));
    }

    inline auto aabb_extents(__m128 pmin, __m128 pmax) {
        return _mm_sub_ps(pmax, pmin);
    }

    inline auto aabb_max_extent_axis(__m128 pmin, __m128 pmax) {
        auto xyz = _mm_sub_ps(pmax, pmin);
        auto yzx = _mm_shuffle_ps(xyz, xyz, _MM_SHUFFLE(3, 0, 2, 1));
        auto m0 = _mm_max_ps(xyz, yzx);
        auto m1 = _mm_shuffle_ps(m0, m0, _MM_SHUFFLE(3, 0, 2, 1));
        auto m2 = _mm_max_ps(m0, m1);
        auto cmp = _mm_cmpeq_ps(xyz, m2);
        return ctz(_mm_movemask_ps(cmp));
    }

    inline auto mm_select(__m128 v, std::uint32_t index) {
        _MM_ALIGN16 float temp[4];
        _mm_store_ps(temp, v);
        return temp[index];
    }

    template <
        typename Node,
        typename NodeTraits,
        typename Allocator = aligned_allocator>
    class Bvh {
        using MetaDataArray = std::vector<std::pair<Mesh const*, std::size_t>>;
        using RefArray = std::vector<std::uint32_t>;

        enum class NodeType {
            kLeaf,
            kInternal
        };

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

            MetaDataArray metadata(num_items);

#ifndef _DEBUG
#ifdef TEST
            auto start = std::chrono::high_resolution_clock::now();
#endif
#endif
            auto constexpr inf = std::numeric_limits<float>::infinity();

            auto scene_min = _mm_set_ps(inf, inf, inf, inf);
            auto scene_max = _mm_set_ps(-inf, -inf, -inf, -inf);
            auto centroid_scene_min = _mm_set_ps(inf, inf, inf, inf);
            auto centroid_scene_max = _mm_set_ps(-inf, -inf, -inf, -inf);

            std::size_t current_face = 0;
            for (auto iter = begin; iter != end; ++iter) {
                auto mesh = static_cast<Mesh const*>(*iter);
                for (std::size_t face_index = 0;
                    face_index < mesh->num_faces();
                    ++face_index, ++current_face) {
                    auto face = mesh->GetIndexData(face_index);

                    auto v0 = _mm_load_ps((float*)mesh->GetVertexDataPtr(face.idx[0]));
                    auto v1 = _mm_load_ps((float*)mesh->GetVertexDataPtr(face.idx[1]));
                    auto v2 = _mm_load_ps((float*)mesh->GetVertexDataPtr(face.idx[2]));

                    auto pmin = _mm_min_ps(_mm_min_ps(v0, v1), v2);
                    auto pmax = _mm_max_ps(_mm_min_ps(v0, v1), v2);
                    auto centroid = _mm_mul_ps(
                        _mm_add_ps(pmin, pmax),
                        _mm_set_ps(0.5f, 0.5f, 0.5f, 0.5f));

                    scene_min = _mm_min_ps(scene_min, pmin);
                    scene_max = _mm_max_ps(scene_max, pmax);

                    centroid_scene_min = _mm_min_ps(centroid_scene_min, centroid);
                    centroid_scene_max = _mm_max_ps(centroid_scene_max, centroid);

                    _mm_store_ps(&aabb_min[current_face].x, pmin);
                    _mm_store_ps(&aabb_max[current_face].x, pmax);
                    _mm_store_ps(&aabb_centroid[current_face].x, centroid);

                    metadata[current_face] = std::make_pair(mesh, face_index);
                }
            }

#ifndef _DEBUG
#ifdef TEST
            auto delta = std::chrono::high_resolution_clock::now() - start;
            std::cout << "AABB calculation time " << std::chrono::duration_cast<
                std::chrono::milliseconds>(delta).count() << " ms\n";
#endif
#endif

#ifndef _DEBUG
#ifdef TEST
            start = std::chrono::high_resolution_clock::now();
#endif
#endif
            BuildImpl(
                scene_min,
                scene_max,
                centroid_scene_min,
                centroid_scene_max,
                aabb_min.get(),
                aabb_max.get(),
                aabb_centroid.get(),
                metadata,
                num_items);

#ifndef _DEBUG
#ifdef TEST
            delta = std::chrono::high_resolution_clock::now() - start;
            std::cout << "Pure build time " << std::chrono::duration_cast<
                std::chrono::milliseconds>(delta).count() << " ms\n";
#endif
#endif
        }


        void Clear() {
            for (auto i = 0u; i < num_nodes_; ++i) {
                nodes_[i].~Node();
            }
            Allocator::deallocate(nodes_);
            nodes_ = nullptr;
            num_nodes_ = 0;
            free_node_idx_ = 0u;
        }

        auto root() const {
            return nodes_[0];
        }

        auto num_nodes() const {
            return num_nodes_;
        }

        auto GetNode(std::size_t idx) const {
            return nodes_ + idx;
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

        NodeType HandleRequest(
            SplitRequest const& request,
            float3 const* aabb_min,
            float3 const* aabb_max,
            float3 const* aabb_centroid,
            MetaDataArray const& metadata,
            RefArray& refs,
            std::size_t num_aabbs,
            SplitRequest& request_left,
            SplitRequest& request_right
        ) {
            if (request.num_refs <= NodeTraits::kMaxLeafPrimitives) {
                NodeTraits::EncodeLeaf(nodes_[request.index],
                    static_cast<std::uint32_t>(request.num_refs));
                for (auto i = 0u; i < request.num_refs; ++i) {
                    auto face_data = metadata[refs[request.start_index + i]];
                    NodeTraits::SetPrimitive(
                        nodes_[request.index],
                        i,
                        face_data
                    );
                }

                return NodeType::kLeaf;
            }

            auto split_axis = aabb_max_extent_axis(
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

            auto split_idx = request.start_index;

            auto constexpr inf = std::numeric_limits<float>::infinity();
            auto m128_plus_inf = _mm_set_ps(inf, inf, inf, inf);
            auto m128_minus_inf = _mm_set_ps(-inf, -inf, -inf, -inf);

            auto lmin = m128_plus_inf;
            auto lmax = m128_minus_inf;
            auto rmin = m128_plus_inf;
            auto rmax = m128_minus_inf;

            auto lcmin = m128_plus_inf;
            auto lcmax = m128_minus_inf;
            auto rcmin = m128_plus_inf;
            auto rcmax = m128_minus_inf;

            if (split_axis_extent > 0.f) {
                if (request.num_refs > NodeTraits::kMinSAHPrimitives) {
                    switch (split_axis) {
                    case 0:
                        split_value = FindSahSplit<0>(
                            request,
                            aabb_min,
                            aabb_max,
                            aabb_centroid,
                            &refs[0]);
                        break;
                    case 1:
                        split_value = FindSahSplit<1>(
                            request,
                            aabb_min,
                            aabb_max,
                            aabb_centroid,
                            &refs[0]);
                        break;
                    case 2:
                        split_value = FindSahSplit<2>(
                            request,
                            aabb_min,
                            aabb_max,
                            aabb_centroid,
                            &refs[0]);
                        break;
                    }
                }

                auto first = request.start_index;
                auto last = request.start_index + request.num_refs;

                while (1) {
                    while ((first != last) &&
                        aabb_centroid[refs[first]][split_axis] < split_value) {
                        auto idx = refs[first];
                        lmin = _mm_min_ps(lmin, _mm_load_ps(&aabb_min[idx].x));
                        lmax = _mm_max_ps(lmax, _mm_load_ps(&aabb_max[idx].x));

                        auto c = _mm_load_ps(&aabb_centroid[idx].x);
                        lcmin = _mm_min_ps(lcmin, c);
                        lcmax = _mm_max_ps(lcmax, c);

                        ++first;
                    }

                    if (first == last--) break;

                    auto idx = refs[first];
                    rmin = _mm_min_ps(rmin, _mm_load_ps(&aabb_min[idx].x));
                    rmax = _mm_max_ps(rmax, _mm_load_ps(&aabb_max[idx].x));

                    auto c = _mm_load_ps(&aabb_centroid[idx].x);
                    rcmin = _mm_min_ps(rcmin, c);
                    rcmax = _mm_max_ps(rcmax, c);

                    while ((first != last) &&
                        aabb_centroid[refs[last]][split_axis] >= split_value) {
                        auto idx = refs[last];
                        rmin = _mm_min_ps(rmin, _mm_load_ps(&aabb_min[idx].x));
                        rmax = _mm_max_ps(rmax, _mm_load_ps(&aabb_max[idx].x));

                        auto c = _mm_load_ps(&aabb_centroid[idx].x);
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
#ifdef _DEBUG
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

                    auto c = _mm_load_ps(&aabb_centroid[idx].x);
                    lcmin = _mm_min_ps(lcmin, c);
                    lcmax = _mm_max_ps(lcmax, c);
                }

                for (auto i = split_idx;
                    i < request.start_index + request.num_refs;
                    ++i) {
                    auto idx = refs[i];
                    rmin = _mm_min_ps(rmin, _mm_load_ps(&aabb_min[idx].x));
                    rmax = _mm_max_ps(rmax, _mm_load_ps(&aabb_max[idx].x));

                    auto c = _mm_load_ps(&aabb_centroid[idx].x);
                    rcmin = _mm_min_ps(rcmin, c);
                    rcmax = _mm_max_ps(rcmax, c);
                }
            }

#ifdef _DEBUG
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
#endif

            request_left.aabb_min = lmin;
            request_left.aabb_max = lmax;
            request_left.centroid_aabb_min = lcmin;
            request_left.centroid_aabb_max = lcmax;
            request_left.start_index = request.start_index;
            request_left.num_refs = split_idx - request.start_index;
            request_left.level = request.level + 1;
            auto child_base = request_left.index = free_node_idx_++;

            request_right.aabb_min = rmin;
            request_right.aabb_max = rmax;
            request_right.centroid_aabb_min = rcmin;
            request_right.centroid_aabb_max = rcmax;
            request_right.start_index = split_idx;
            request_right.num_refs = request.num_refs - request_left.num_refs;
            request_right.level = request.level + 1;
            request_right.index = free_node_idx_++;

            NodeTraits::EncodeInternal(
                nodes_[request.index],
                request.aabb_min,
                request.aabb_max,
                child_base);

            return NodeType::kInternal;
        }

        void BuildImpl(
            __m128 scene_min,
            __m128 scene_max,
            __m128 centroid_scene_min,
            __m128 centroid_scene_max,
            float3 const* aabb_min,
            float3 const* aabb_max,
            float3 const* aabb_centroid,
            MetaDataArray const& metadata,
            std::size_t num_aabbs) {

            RefArray refs(num_aabbs);
            std::iota(refs.begin(), refs.end(), 0);

            _MM_ALIGN16 SplitRequest requests[kStackSize];
            auto sptr = 0u;

            num_nodes_ = num_aabbs * 2 - 1;
            nodes_ = reinterpret_cast<Node*>(
                Allocator::allocate(sizeof(Node) * num_nodes_, 16u));

            for (auto i = 0u; i < num_nodes_; ++i) {
                new (&nodes_[i]) Node;
            }

            free_node_idx_ = 1;

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

                auto& request_left{ requests[sptr++] };

                if (sptr == kStackSize) {
                    throw std::runtime_error("Build stack overflow");
                }

                auto& request_right{ requests[sptr++] };


                if (sptr == kStackSize) {
                    throw std::runtime_error("Build stack overflow");
                }

                if (HandleRequest(
                    request,
                    aabb_min,
                    aabb_max,
                    aabb_centroid,
                    metadata,
                    refs,
                    num_aabbs,
                    request_left,
                    request_right) == 
                    NodeType::kLeaf) {
                    --sptr;
                    --sptr;
                }
            }
        }

        template <std::uint32_t axis> float FindSahSplit(
            SplitRequest const& request,
            float3 const* aabb_min,
            float3 const* aabb_max,
            float3 const* aabb_centroid,
            std::uint32_t const* refs
        ) {
            auto sah = std::numeric_limits<float>::max();

            auto constexpr kNumBins = 16u;
            __m128 bin_min[kNumBins];
            __m128 bin_max[kNumBins];
            std::uint32_t bin_count[kNumBins];

            auto constexpr inf = std::numeric_limits<float>::infinity();
            for (auto i = 0u; i < kNumBins; ++i)
            {
                bin_count[i] = 0;
                bin_min[i] = _mm_set_ps(inf, inf, inf, inf);
                bin_max[i] = _mm_set_ps(-inf, -inf, -inf, -inf);
            }

            auto centroid_extent = aabb_extents(request.centroid_aabb_min,
                request.centroid_aabb_max);
            auto centroid_min = _mm_shuffle_ps(request.centroid_aabb_min,
                request.centroid_aabb_min,
                _MM_SHUFFLE(axis, axis, axis, axis));
            centroid_extent = _mm_shuffle_ps(centroid_extent,
                centroid_extent,
                _MM_SHUFFLE(axis, axis, axis, axis));
            auto centroid_extent_inv = _mm_rcp_ps(centroid_extent);
            auto area_inv = mm_select(
                _mm_rcp_ps(
                    aabb_surface_area(
                        request.aabb_min,
                        request.aabb_max)
                ), 0);

            auto full4 = request.num_refs & ~0x3;
            auto num_bins = _mm_set_ps(
                (float)kNumBins, (float)kNumBins,
                (float)kNumBins, (float)kNumBins);

            for (auto i = request.start_index;
                i < request.start_index + full4;
                i += 4u)
            {
                auto idx0 = refs[i];
                auto idx1 = refs[i + 1];
                auto idx2 = refs[i + 2];
                auto idx3 = refs[i + 3];

                auto c = _mm_set_ps(
                    aabb_centroid[idx0][axis],
                    aabb_centroid[idx1][axis],
                    aabb_centroid[idx2][axis],
                    aabb_centroid[idx3][axis]);

                auto bin_idx = _mm_mul_ps(
                    _mm_mul_ps(
                        _mm_sub_ps(c, centroid_min),
                        centroid_extent_inv), num_bins);

                auto bin_idx0 = std::min(static_cast<uint32_t>(mm_select(bin_idx, 0u)), kNumBins - 1);
                auto bin_idx1 = std::min(static_cast<uint32_t>(mm_select(bin_idx, 1u)), kNumBins - 1);
                auto bin_idx2 = std::min(static_cast<uint32_t>(mm_select(bin_idx, 2u)), kNumBins - 1);
                auto bin_idx3 = std::min(static_cast<uint32_t>(mm_select(bin_idx, 3u)), kNumBins - 1);

#ifdef _DEBUG
#ifdef TEST
                assert(bin_idx0 >= 0u); assert(bin_idx0 < kNumBins);
                assert(bin_idx1 >= 0u); assert(bin_idx1 < kNumBins);
                assert(bin_idx3 >= 0u); assert(bin_idx2 < kNumBins);
                assert(bin_idx3 >= 0u); assert(bin_idx3 < kNumBins);
#endif
#endif

                ++bin_count[bin_idx0];
                ++bin_count[bin_idx1];
                ++bin_count[bin_idx2];
                ++bin_count[bin_idx3];

                bin_min[bin_idx0] = _mm_min_ps(
                    bin_min[bin_idx0],
                    _mm_load_ps(&aabb_min[idx0].x));
                bin_max[bin_idx0] = _mm_max_ps(
                    bin_max[bin_idx0],
                    _mm_load_ps(&aabb_max[idx0].x));
                bin_min[bin_idx1] = _mm_min_ps(
                    bin_min[bin_idx1],
                    _mm_load_ps(&aabb_min[idx1].x));
                bin_max[bin_idx1] = _mm_max_ps(
                    bin_max[bin_idx1],
                    _mm_load_ps(&aabb_max[idx1].x));
                bin_min[bin_idx2] = _mm_min_ps(
                    bin_min[bin_idx2],
                    _mm_load_ps(&aabb_min[idx2].x));
                bin_max[bin_idx2] = _mm_max_ps(
                    bin_max[bin_idx2],
                    _mm_load_ps(&aabb_max[idx2].x));
                bin_min[bin_idx3] = _mm_min_ps(
                    bin_min[bin_idx3],
                    _mm_load_ps(&aabb_min[idx3].x));
                bin_max[bin_idx3] = _mm_max_ps(
                    bin_max[bin_idx3],
                    _mm_load_ps(&aabb_max[idx3].x));
            }

            auto cm = mm_select(centroid_min, 0u);
            auto cei = mm_select(centroid_extent_inv, 0u);
            for (auto i = request.start_index + full4; i < request.start_index + request.num_refs; ++i)
            {
                auto idx = refs[i];
                auto bin_idx = static_cast<uint32_t>(
                    kNumBins * ((1.f - 1e-6f) *
                    (aabb_centroid[idx][axis] - cm) *
                        cei));
                ++bin_count[bin_idx];

                bin_min[bin_idx] = _mm_min_ps(
                    bin_min[bin_idx],
                    _mm_load_ps(&aabb_min[idx].x));
                bin_max[bin_idx] = _mm_max_ps(
                    bin_max[bin_idx],
                    _mm_load_ps(&aabb_max[idx].x));
            }

#ifdef _DEBUG
#ifdef TEST
            auto num_refs = request.num_refs;
            for (auto i = 0u; i < kNumBins; ++i) {
                num_refs -= bin_count[i];
            }
            assert(num_refs == 0);
#endif
#endif

            __m128 right_min[kNumBins - 1];
            __m128 right_max[kNumBins - 1];
            auto tmp_min = _mm_set_ps(inf, inf, inf, inf);
            auto tmp_max = _mm_set_ps(-inf, -inf, -inf, -inf);

            for (auto i = kNumBins - 1; i > 0; --i)
            {
                tmp_min = _mm_min_ps(tmp_min, bin_min[i]);
                tmp_max = _mm_max_ps(tmp_max, bin_max[i]);

                right_min[i - 1] = tmp_min;
                right_max[i - 1] = tmp_max;
            }

            tmp_min = _mm_set_ps(inf, inf, inf, inf);
            tmp_max = _mm_set_ps(-inf, -inf, -inf, -inf);
            auto  lc = 0u;
            auto  rc = request.num_refs;

            auto split_idx = -1;
            for (auto i = 0u; i < kNumBins - 1; ++i)
            {
                tmp_min = _mm_min_ps(tmp_min, bin_min[i]);
                tmp_max = _mm_max_ps(tmp_max, bin_max[i]);
                lc += bin_count[i];
                rc -= bin_count[i];

                auto lsa = mm_select(
                    aabb_surface_area(tmp_min, tmp_max), 0);
                auto rsa = mm_select(
                    aabb_surface_area(right_min[i], right_max[i]), 0);

                // Compute SAH
                auto s = static_cast<float>(NodeTraits::kTraversalCost) + (lc * lsa + rc * rsa) * area_inv;

                // Check if it is better than what we found so far
                if (s < sah)
                {
                    split_idx = i;
                    sah = s;
                }
            }

            return mm_select(centroid_min, 0u) + (split_idx + 1) * (mm_select(centroid_extent, 0u) / kNumBins);
        }

        Node* nodes_ = nullptr;
        std::size_t num_nodes_ = 0;
        std::uint32_t free_node_idx_ = 0u;
    };
}
