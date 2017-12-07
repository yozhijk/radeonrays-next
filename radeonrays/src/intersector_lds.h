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
#include <vulkan/vulkan.hpp>
#include "vk_memory_allocator.h"
#include "intersector.h"
#include "bvh.h"
#include "bvh_encoder.h"

namespace RadeonRays {
    class IntersectorLDS : public Intersector {
    public:
        IntersectorLDS(
            vk::Device device,
            vk::CommandPool cmdpool,
            vk::DescriptorPool descpool,
            vk::PipelineCache pipeline_cache,
            VkMemoryAlloc& alloc
        );

        ~IntersectorLDS() override;

        void BindBuffers(vk::Buffer rays, vk::Buffer hits, std::uint32_t num_rays) override;
        vk::CommandBuffer Commit(World const& world) override;
        vk::CommandBuffer TraceRays(std::uint32_t num_rays) override;

        IntersectorLDS(IntersectorLDS const&) = delete;
        IntersectorLDS& operator = (IntersectorLDS const&) = delete;

    private:
        // The function checks if BVH buffers have enough space and
        // reallocates if necessary.
        void CheckAndReallocBVH(std::size_t required_size) {
            if (bvh_staging_.size < required_size) {

                alloc_.deallocate(bvh_staging_);
                alloc_.deallocate(bvh_local_);

                bvh_staging_ = alloc_.allocate(
                    vk::MemoryPropertyFlagBits::eHostVisible,
                    vk::BufferUsageFlagBits::eTransferSrc,
                    required_size,
                    16u);

                bvh_local_ = alloc_.allocate(
                    vk::MemoryPropertyFlagBits::eDeviceLocal,
                    vk::BufferUsageFlagBits::eStorageBuffer |
                    vk::BufferUsageFlagBits::eTransferDst,
                    required_size,
                    16u);
            }
        }

        static std::uint32_t constexpr kInitialWorkBufferSize = 1024u * 1024u;
        static std::uint32_t constexpr kGlobalStackSize = 32u;

        // Intersection device
        vk::Device device_;
        // Allocate command buffers from here
        vk::CommandPool cmdpool_;
        // Allocate descriptors from here
        vk::DescriptorPool descpool_;
        // Pipeline cache, not used for now
        vk::PipelineCache pipeline_cache_;
        // Device memory allocator
        VkMemoryAlloc& alloc_;

        // Intersector compute pipeline layout
        vk::PipelineLayout pipeline_layout_;
        // Intersector compute pipeline
        vk::Pipeline pipeline_;
        // Descriptors layout
        vk::DescriptorSetLayout desc_layout_;
        // Shader module
        vk::ShaderModule shader_;
        // Descriptor sets
        std::vector<vk::DescriptorSet> descsets_;

        // Device local stack buffer
        VkMemoryAlloc::StorageBlock stack_;
        VkMemoryAlloc::StorageBlock bvh_staging_;
        VkMemoryAlloc::StorageBlock bvh_local_;
    };
}
