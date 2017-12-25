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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stack>
#include <xmmintrin.h>
#include <smmintrin.h>

#include "mesh.h"
#include "utils.h"
#include "bvh_utils.h"

#define PARALLEL_BUILD

namespace RadeonRays {

    // BVH builder class
    // Node represents the type to use as a BVH node/leaf
    // NodeTraits defines operation supported by nodes, like
    //  * Setting the node as an internal
    //  * Setting the node as a leaf
    //  * Setting the node's bounding box or triangle data
    // PrimitiveTraits defines operation on primitives:
    //  * Getting the number of sub-components(Mesh->triangle, Subdiv->triangle, etc)
    //  * Getting subcomponent AABBs
    // Allocators provide means for aligned allocations
    template <typename Node,
        typename NodeTraits,
        typename PrimitiveTraits,
        typename Allocator = aligned_allocator>
    class BVH
    {
        // Metadata is passed into the final node encoder when leaf is created
        using MetaDataArray =
            std::vector<std::pair<typename PrimitiveTraits::MetadataPtr, std::size_t>>;
        // Array of subcomponent indices
        using RefArray = std::vector<std::uint32_t>;

        // Each node could be either internal of leaf
        enum class NodeType
        {
            kLeaf,
            kInternal
        };

    public:
        // Define node type externally
        using NodeT = Node;
        // Build from the range of shapes
        template<typename Iter> void Build(Iter begin, Iter end)
        {
            // Number of shapes in the range
            auto num_shapes = std::distance(begin, end);
            // Make sure the range is non-empty
            assert(num_shapes > 0);
            // Release previous data if any
            Clear();

            // Determine total number of subcomponents
            std::size_t num_items = 0;
            for (auto i = begin; i != end; ++i)
            {
                num_items += PrimitiveTraits::GetNumAABBs(*i);
            }

            // Allocate memory for subcomponent AABBs
            auto deleter = [](void* p)
            {
                Allocator::deallocate(p);
            };

            using AlignedFloat3Ptr = std::unique_ptr<float3[], decltype(deleter)>;
            auto aabb_min = AlignedFloat3Ptr(reinterpret_cast<float3*>(
                Allocator::allocate(sizeof(float3) * num_items, 16u)), deleter);
            auto aabb_max = AlignedFloat3Ptr(reinterpret_cast<float3*>(
                Allocator::allocate(sizeof(float3) * num_items, 16u)), deleter);
            auto aabb_centroid = AlignedFloat3Ptr(reinterpret_cast<float3*>(
                Allocator::allocate(sizeof(float3) * num_items, 16u)), deleter);

            // Allocate metadata array 
            MetaDataArray metadata(num_items);

#ifndef _DEBUG
#ifdef TEST
            auto start = std::chrono::high_resolution_clock::now();
#endif
#endif
            // Calculate AABBs for sub-components, scene AABB and 
            // subcomponent centroids AABB requie
            auto constexpr inf = std::numeric_limits<float>::infinity();
            auto scene_min = _mm_set_ps(inf, inf, inf, inf);
            auto scene_max = _mm_set_ps(-inf, -inf, -inf, -inf);
            auto centroid_scene_min = _mm_set_ps(inf, inf, inf, inf);
            auto centroid_scene_max = _mm_set_ps(-inf, -inf, -inf, -inf);

            std::size_t current_index = 0;
            for (auto iter = begin; iter != end; ++iter)
            {
                for (std::size_t index = 0;
                    index < PrimitiveTraits::GetNumAABBs(*iter);
                    ++index, ++current_index)
                {

                    // Get current subcomponent AABB
                    __m128 pmin, pmax;
                    PrimitiveTraits::GetAABB(*iter, index, pmin, pmax);

                    // Calculate its centroid
                    auto centroid = _mm_mul_ps(_mm_add_ps(pmin, pmax),
                        _mm_set_ps(0.5f, 0.5f, 0.5f, 0.5f));

                    // Update scene extents
                    scene_min = _mm_min_ps(scene_min, pmin);
                    scene_max = _mm_max_ps(scene_max, pmax);

                    // Update centroids extents
                    centroid_scene_min = _mm_min_ps(centroid_scene_min, centroid);
                    centroid_scene_max = _mm_max_ps(centroid_scene_max, centroid);

                    // Store into aligned arrays of AABB mins and maxes
                    _mm_store_ps(&aabb_min[current_index].x, pmin);
                    _mm_store_ps(&aabb_max[current_index].x, pmax);
                    // Store centroid
                    _mm_store_ps(&aabb_centroid[current_index].x, centroid);

                    // Save metadata pair
                    metadata[current_index] = std::make_pair(*iter, index);
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
            // Build BVH topology
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

            // Perform user-defined post-processing
            NodeTraits::Finalize(*this);
        }

        // Release allocated memory
        void Clear()
        {
            // Destroy nodes
            for (auto i = 0u; i < num_nodes_; ++i)
            {
                nodes_[i].~Node();
            }
            // Deallocated memory
            Allocator::deallocate(nodes_);
            nodes_ = nullptr;
            num_nodes_ = 0;
        }

        // Get BVH root node pointer
        auto root() const { return nodes_[0]; }
        // Get total number of nodes
        auto num_nodes() const { return num_nodes_; }
        // Get node pointer at index
        auto node(std::size_t idx) const { return nodes_ + idx; }

    private:
        // Build stack size
        static constexpr std::uint32_t kStackSize = 1024u;

        // BVH split request
        struct SplitRequest
        {
            // Request AABB
            __m128 aabb_min;
            __m128 aabb_max;
            // Request centroids AABB
            __m128 centroid_aabb_min;
            __m128 centroid_aabb_max;
            // Start index of the range
            std::size_t start_index;
            // Number of primitives in the request
            std::size_t num_refs;
            // Depth
            std::uint32_t level;
            // Node index to store data
            std::uint32_t index;
        };

        // Handle BVH split request
        NodeType HandleRequest(
            SplitRequest const& request,
            float3 const* aabb_min,
            float3 const* aabb_max,
            float3 const* aabb_centroid,
            MetaDataArray const& metadata,
            RefArray& refs,
            std::size_t num_aabbs,
            SplitRequest& request_left,
            SplitRequest& request_right)
        {
            // If we reached primitive count for the leaf, create a leaf
            if (request.num_refs <= NodeTraits::kMaxLeafPrimitives)
            {
                auto num_refs = static_cast<std::uint32_t>(request.num_refs);
                // Encode leaf node
                NodeTraits::EncodeLeaf(nodes_[request.index], num_refs);

                // Set leaf primitives
                for (auto i = 0u; i < request.num_refs; ++i)
                {
                    // Extract metadata pointer
                    auto face_data = metadata[refs[request.start_index + i]];
                    NodeTraits::SetPrimitive(nodes_[request.index], i, face_data);
                }

                // Create a leaf
                return NodeType::kLeaf;
            }

            // Calculate median split value first
            auto split_axis = aabb_max_extent_axis(
                request.centroid_aabb_min,
                request.centroid_aabb_max);
            auto split_axis_extent = mm_select(
                _mm_sub_ps(request.centroid_aabb_max,
                request.centroid_aabb_min),
                split_axis);
            auto split_value = mm_select(_mm_mul_ps(
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

            // Try to find SAH split
            if (split_axis_extent > 0.f)
            {
                if (request.num_refs > NodeTraits::kMinSAHPrimitives)
                {
                    switch (split_axis)
                    {
                    case 0:
                        split_value =
                            FindSahSplit<0>(request, aabb_min, aabb_max, aabb_centroid, &refs[0]);
                        break;
                    case 1:
                        split_value =
                            FindSahSplit<1>(request, aabb_min, aabb_max, aabb_centroid, &refs[0]);
                        break;
                    case 2:
                        split_value =
                            FindSahSplit<2>(request, aabb_min, aabb_max, aabb_centroid, &refs[0]);
                        break;
                    }
                }

                // Perform partitioning
                auto first = request.start_index;
                auto last = request.start_index + request.num_refs;

                while (1)
                {
                    while ((first != last) &&
                        aabb_centroid[refs[first]][split_axis] < split_value)
                    {
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
                        aabb_centroid[refs[last]][split_axis] >= split_value)
                    {
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
                        ++i)
                    {
                        if (i < split_idx)
                        {
                            assert(aabb_centroid[refs[i]][split_axis] < split_value);
                        }
                        else
                        {
                            assert(aabb_centroid[refs[i]][split_axis] >= split_value);
                        }
                    }
                }
#endif
#endif
            }

            // If split was bad, do split in half
            if (split_idx == request.start_index ||
                split_idx == request.start_index + request.num_refs)
            {
                split_idx = request.start_index + (request.num_refs >> 1);

                lmin = m128_plus_inf;
                lmax = m128_minus_inf;
                rmin = m128_plus_inf;
                rmax = m128_minus_inf;

                lcmin = m128_plus_inf;
                lcmax = m128_minus_inf;
                rcmin = m128_plus_inf;
                rcmax = m128_minus_inf;

                for (auto i = request.start_index; i < split_idx; ++i)
                {
                    auto idx = refs[i];
                    lmin = _mm_min_ps(lmin, _mm_load_ps(&aabb_min[idx].x));
                    lmax = _mm_max_ps(lmax, _mm_load_ps(&aabb_max[idx].x));

                    auto c = _mm_load_ps(&aabb_centroid[idx].x);
                    lcmin = _mm_min_ps(lcmin, c);
                    lcmax = _mm_max_ps(lcmax, c);
                }

                for (auto i = split_idx;
                    i < request.start_index + request.num_refs;
                    ++i)
                {
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
                _MM_ALIGN16 bbox left, right, parent;
                _mm_store_ps(&left.pmin.x, lmin);
                _mm_store_ps(&left.pmax.x, lmax);
                _mm_store_ps(&right.pmin.x, rmin);
                _mm_store_ps(&right.pmax.x, rmax);
                _mm_store_ps(&parent.pmin.x, request.aabb_min);
                _mm_store_ps(&parent.pmax.x, request.aabb_max);

                assert(contains(parent, left));
                assert(contains(parent, right));
            }
#endif
#endif
            // Create left child request
            request_left.aabb_min = lmin;
            request_left.aabb_max = lmax;
            request_left.centroid_aabb_min = lcmin;
            request_left.centroid_aabb_max = lcmax;
            request_left.start_index = request.start_index;
            request_left.num_refs = split_idx - request.start_index;
            request_left.level = request.level + 1;
            request_left.index = request.index + 1;

            // Create right child request
            request_right.aabb_min = rmin;
            request_right.aabb_max = rmax;
            request_right.centroid_aabb_min = rcmin;
            request_right.centroid_aabb_max = rcmax;
            request_right.start_index = split_idx;
            request_right.num_refs = request.num_refs - request_left.num_refs;
            request_right.level = request.level + 1;
            request_right.index =
                static_cast<std::uint32_t>(request.index + request_left.num_refs * 2);


            // Encode internal node
            NodeTraits::EncodeInternal(nodes_[request.index],
                request.aabb_min,
                request.aabb_max,
                request_left.index,
                request_right.index);

            // Create internal node
            return NodeType::kInternal;
        }

        // Build function
        void BuildImpl(__m128 scene_min,
            __m128 scene_max,
            __m128 centroid_scene_min,
            __m128 centroid_scene_max,
            float3 const* aabb_min,
            float3 const* aabb_max,
            float3 const* aabb_centroid,
            MetaDataArray const& metadata,
            std::size_t num_aabbs)
        {
            // Create reference array
            RefArray refs(num_aabbs);
            std::iota(refs.begin(), refs.end(), 0);

            // Number of nodes in 2-BVH
            num_nodes_ = num_aabbs * 2 - 1;
            // Allocate memory for the nodes
            nodes_ = reinterpret_cast<Node*>(Allocator::allocate(sizeof(Node) * num_nodes_, 16u));

            // Construct nodes
            for (auto i = 0u; i < num_nodes_; ++i)
            {
                new (&nodes_[i]) Node;
            }

            // Start building BVH
            auto constexpr inf = std::numeric_limits<float>::infinity();
            auto m128_plus_inf = _mm_set_ps(inf, inf, inf, inf);
            auto m128_minus_inf = _mm_set_ps(-inf, -inf, -inf, -inf);

#ifndef PARALLEL_BUILD
            // Build stack
            _MM_ALIGN16 SplitRequest requests[kStackSize];
            // Stack ptr
            auto sptr = 0u;

            // Put root range into the stack
            requests[sptr++] = SplitRequest
            {
                scene_min,
                scene_max,
                centroid_scene_min,
                centroid_scene_max,
                0,
                num_aabbs,
                0u,
                0u
            };

            // Build topology
            while (sptr > 0)
            {
                // Extract request
                auto request = requests[--sptr];
                // Allocate space for left request
                auto& request_left{ requests[sptr++] };

                // Check for overflow
                if (sptr == kStackSize)
                {
                    throw std::runtime_error("Build stack overflow");
                }

                // Allocate space for right request
                auto& request_right{ requests[sptr++] };

                // Check for overflow
                if (sptr == kStackSize)
                {
                    throw std::runtime_error("Build stack overflow");
                }

                // Handle split request,
                // deallocate stack memory in case of the leaf
                if (HandleRequest(request,
                    aabb_min,
                    aabb_max,
                    aabb_centroid,
                    metadata,
                    refs,
                    num_aabbs,
                    request_left,
                    request_right) == NodeType::kLeaf)
                {
                    --sptr;
                    --sptr;
                }
            }
#else
            // Perform multithreaded build
            std::stack<SplitRequest> requests;

            std::condition_variable cv;
            std::mutex mutex;
            std::atomic_bool shutdown = false;
            std::atomic_uint32_t num_refs_processed = 0;

            // Push root range into the stack
            requests.push(SplitRequest
            {
                scene_min,
                scene_max,
                centroid_scene_min,
                centroid_scene_max,
                0,
                num_aabbs,
                0u,
                0u
            });

            // Worker thread lambda
            auto worker_thread = [&]()
            {
                // Local stack for this thread
                thread_local std::stack<SplitRequest> local_requests;

                // Start building
                while (1)
                {
                    {
                        // Wait until there are items in global stack
                        std::unique_lock<std::mutex> lock(mutex);
                        cv.wait(lock, [&]() {return !requests.empty() || shutdown; });

                        if (shutdown) return;

                        // Push it to the local stack
                        local_requests.push(requests.top());
                        requests.pop();
                    }

                    // Allocate requests memory on the stack
                    _MM_ALIGN16 SplitRequest request_left;
                    _MM_ALIGN16 SplitRequest request_right;
                    _MM_ALIGN16 SplitRequest request;

                    // Handle local requests
                    while (!local_requests.empty())
                    {
                        // Extract request
                        request = local_requests.top();
                        local_requests.pop();

                        // Handle request
                        auto node_type = HandleRequest(request,
                            aabb_min,
                            aabb_max,
                            aabb_centroid,
                            metadata,
                            refs,
                            num_aabbs,
                            request_left,
                            request_right);

                        // Update stop counter
                        if (node_type == NodeType::kLeaf)
                        {
                            num_refs_processed += static_cast<std::uint32_t>(request.num_refs);
                            continue;
                        }

                        // Small requests are going to local stack for right child
                        if (request_right.num_refs > 4096u)
                        {
                            std::unique_lock<std::mutex> lock(mutex);
                            requests.push(request_right);
                            cv.notify_one();
                        }
                        else
                        {
                            local_requests.push(request_right);
                        }

                        // Left request is always handled locally
                        local_requests.push(request_left);
                    }
                }
            };

            // Use available number of threads
            auto num_threads = std::thread::hardware_concurrency();
            std::vector<std::thread> threads(num_threads);

            // Run threads
            for (auto i = 0u; i < num_threads; ++i)
            {
                threads[i] = std::move(std::thread(worker_thread));
                threads[i].detach();
            }

            // Wait for the workload to complete
            while (num_refs_processed != num_aabbs)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }

            // Shut threads down
            shutdown = true;
#endif
        }

        // Find best split according to Surface Area Heuristic
        template <std::uint32_t axis>
        float FindSahSplit(SplitRequest const& request,
            float3 const* aabb_min,
            float3 const* aabb_max,
            float3 const* aabb_centroid,
            std::uint32_t const* refs)
        {
            auto sah = std::numeric_limits<float>::max();

            // Use 64 bins statically
            auto constexpr kNumBins = 64u;
            __m128 bin_min[kNumBins];
            __m128 bin_max[kNumBins];
            std::uint32_t bin_count[kNumBins];

            // Initalize bins
            auto constexpr inf = std::numeric_limits<float>::infinity();
            for (auto i = 0u; i < kNumBins; ++i)
            {
                bin_count[i] = 0;
                bin_min[i] = _mm_set_ps(inf, inf, inf, inf);
                bin_max[i] = _mm_set_ps(-inf, -inf, -inf, -inf);
            }

            // Precalculate some constants
            auto centroid_extent = aabb_extents(request.centroid_aabb_min,
                request.centroid_aabb_max);
            auto centroid_min = _mm_shuffle_ps(request.centroid_aabb_min,
                request.centroid_aabb_min,
                _MM_SHUFFLE(axis, axis, axis, axis));
            centroid_extent = _mm_shuffle_ps(centroid_extent,
                centroid_extent,
                _MM_SHUFFLE(axis, axis, axis, axis));
            auto centroid_extent_inv = _mm_rcp_ps(centroid_extent);
            auto area_inv = mm_select(_mm_rcp_ps(
                aabb_surface_area(
                    request.aabb_min,
                    request.aabb_max)), 0);

            // Use SSE for 4 prim grops
            auto full4 = request.num_refs & ~0x3;
            auto num_bins = _mm_set_ps((float)kNumBins, (float)kNumBins,
                (float)kNumBins, (float)kNumBins);

            // Bin groups of primitives
            for (auto i = request.start_index;
                i < request.start_index + full4;
                i += 4u)
            {
                // Load indices
                auto idx0 = refs[i];
                auto idx1 = refs[i + 1];
                auto idx2 = refs[i + 2];
                auto idx3 = refs[i + 3];

                // Load centroids
                auto c = _mm_set_ps(aabb_centroid[idx3][axis],
                    aabb_centroid[idx2][axis],
                    aabb_centroid[idx1][axis],
                    aabb_centroid[idx0][axis]);

                // Calculate the bin
                auto bin_idx = _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(c, centroid_min),
                    centroid_extent_inv), num_bins);

                // Extract indices
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
                // Update bin histogram
                ++bin_count[bin_idx0];
                ++bin_count[bin_idx1];
                ++bin_count[bin_idx2];
                ++bin_count[bin_idx3];

                // Update bin extents
                bin_min[bin_idx0] = _mm_min_ps(bin_min[bin_idx0],
                    _mm_load_ps(&aabb_min[idx0].x));
                bin_max[bin_idx0] = _mm_max_ps(bin_max[bin_idx0],
                    _mm_load_ps(&aabb_max[idx0].x));
                bin_min[bin_idx1] = _mm_min_ps(bin_min[bin_idx1],
                    _mm_load_ps(&aabb_min[idx1].x));
                bin_max[bin_idx1] = _mm_max_ps(bin_max[bin_idx1],
                    _mm_load_ps(&aabb_max[idx1].x));
                bin_min[bin_idx2] = _mm_min_ps(bin_min[bin_idx2],
                    _mm_load_ps(&aabb_min[idx2].x));
                bin_max[bin_idx2] = _mm_max_ps(bin_max[bin_idx2],
                    _mm_load_ps(&aabb_max[idx2].x));
                bin_min[bin_idx3] = _mm_min_ps(bin_min[bin_idx3],
                    _mm_load_ps(&aabb_min[idx3].x));
                bin_max[bin_idx3] = _mm_max_ps(bin_max[bin_idx3],
                    _mm_load_ps(&aabb_max[idx3].x));
            }

            // Calculate the rest
            auto cm = mm_select(centroid_min, 0u);
            auto cei = mm_select(centroid_extent_inv, 0u);
            for (auto i = request.start_index + full4;
                i < request.start_index + request.num_refs;
                ++i)
            {
                auto idx = refs[i];
                auto bin_idx = std::min(static_cast<uint32_t>(
                    kNumBins * (aabb_centroid[idx][axis] - cm) * cei),
                    kNumBins - 1);
                ++bin_count[bin_idx];
                bin_min[bin_idx] = _mm_min_ps(bin_min[bin_idx],
                    _mm_load_ps(&aabb_min[idx].x));
                bin_max[bin_idx] = _mm_max_ps(bin_max[bin_idx],
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

            // Perform best SAH search (linear complexity)
            __m128 right_min[kNumBins - 1];
            __m128 right_max[kNumBins - 1];
            auto tmp_min = _mm_set_ps(inf, inf, inf, inf);
            auto tmp_max = _mm_set_ps(-inf, -inf, -inf, -inf);

            // Precalculate right AABBs
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

            // Sweep from the left to right
            auto split_idx = -1;
            for (auto i = 0u; i < kNumBins - 1; ++i)
            {
                tmp_min = _mm_min_ps(tmp_min, bin_min[i]);
                tmp_max = _mm_max_ps(tmp_max, bin_max[i]);
                lc += bin_count[i];
                rc -= bin_count[i];

                auto lsa = mm_select(aabb_surface_area(tmp_min, tmp_max), 0);
                auto rsa = mm_select(aabb_surface_area(right_min[i], right_max[i]), 0);
                auto s = static_cast<float>(NodeTraits::kTraversalCost) +
                    (lc * lsa + rc * rsa) * area_inv;

                // Update SAH if it is better than the current one
                if (s < sah)
                {
                    split_idx = i;
                    sah = s;
                }
            }

            // Caclulate splitting value
            return mm_select(centroid_min, 0u) +
                (split_idx + 1) * (mm_select(centroid_extent, 0u) / kNumBins);
        }

        // BVH nodes
        Node* nodes_ = nullptr;
        // Number of BVH nodes
        std::size_t num_nodes_ = 0;
    };
}
