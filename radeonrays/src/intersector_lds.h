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

#include "vk_utils.h"
#include "vk_memory_allocator.h"

#include "utils.h"
#include "intersector.h"
#include "bvh.h"
#include "bvh_encoder.h"

namespace RadeonRays {
    template <typename BVH, typename BVHTraits> 
    class IntersectorLDS : public Intersector {
        static std::uint32_t constexpr kNumBindings = 4u;
        static std::uint32_t constexpr kWorgGroupSize = 64u;

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

        // Check if we have enough memory in the stack buffer
        void CheckAndReallocStackBuffer(std::size_t required_size) {
            if (stack_.size < required_size) {
                alloc_.deallocate(stack_);
                stack_ = alloc_.allocate(
                    vk::MemoryPropertyFlagBits::eDeviceLocal,
                    vk::BufferUsageFlagBits::eStorageBuffer,
                    required_size,
                    16u);
            }
        }

        static std::uint32_t constexpr kInitialWorkBufferSize = 1920u * 1080u;
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

    template <typename BVH, typename BVHTraits>
    inline IntersectorLDS<BVH, BVHTraits>::IntersectorLDS(
        vk::Device device,
        vk::CommandPool cmdpool,
        vk::DescriptorPool descpool,
        vk::PipelineCache pipeline_cache,
        VkMemoryAlloc& alloc
    )
        : device_(device)
        , cmdpool_(cmdpool)
        , descpool_(descpool)
        , pipeline_cache_(pipeline_cache)
        , alloc_(alloc) {
        // Create bindings:
        // (0) Ray buffer 
        // (1) Hit buffer 
        // (2) BVH buffer
        // (3) Stack buffer
        vk::DescriptorSetLayoutBinding layout_binding[kNumBindings];

        // Initialize bindings
        for (auto i = 0u; i < kNumBindings; ++i) {
            layout_binding[i]
                .setBinding(i)
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setStageFlags(vk::ShaderStageFlagBits::eCompute);
        }

        // Create descriptor set layout
        vk::DescriptorSetLayoutCreateInfo desc_layout_create_info;
        desc_layout_create_info
            .setBindingCount(kNumBindings)
            .setPBindings(layout_binding);
        desc_layout_ = device_
            .createDescriptorSetLayout(desc_layout_create_info);

        // Allocate descriptors
        vk::DescriptorSetAllocateInfo desc_alloc_info;
        desc_alloc_info.setDescriptorPool(descpool_);
        desc_alloc_info.setDescriptorSetCount(1);
        desc_alloc_info.setPSetLayouts(&desc_layout_);
        descsets_ = device_.allocateDescriptorSets(desc_alloc_info);

        // Ray count is a push constant, so create a range for it
        vk::PushConstantRange push_constant_range;
        push_constant_range.setOffset(0)
            .setSize(sizeof(std::uint32_t))
            .setStageFlags(vk::ShaderStageFlagBits::eCompute);

        // Create pipeline layout
        vk::PipelineLayoutCreateInfo pipeline_layout_create_info;
        pipeline_layout_create_info
            .setSetLayoutCount(1)
            .setPSetLayouts(&desc_layout_)
            .setPushConstantRangeCount(1)
            .setPPushConstantRanges(&push_constant_range);
        pipeline_layout_ = device_
            .createPipelineLayout(pipeline_layout_create_info);

        // Load intersection shader module
        std::string path = "../../shaders/";
        path.append(BVHTraits::GetGPUTraversalFileName());
        shader_ = LoadShaderModule(device_, path);

        // Create pipeline 
        vk::PipelineShaderStageCreateInfo shader_stage_create_info;
        shader_stage_create_info
            .setStage(vk::ShaderStageFlagBits::eCompute)
            .setModule(shader_)
            .setPName("main");

        vk::ComputePipelineCreateInfo pipeline_create_info;
        pipeline_create_info
            .setLayout(pipeline_layout_)
            .setStage(shader_stage_create_info);

        pipeline_ = device_.createComputePipeline(
            pipeline_cache_,
            pipeline_create_info);

        stack_ = alloc_.allocate(
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vk::BufferUsageFlagBits::eStorageBuffer,
            kInitialWorkBufferSize * kGlobalStackSize * sizeof(std::uint32_t),
            16u);
    }

    template <typename BVH, typename BVHTraits>
    inline IntersectorLDS<BVH, BVHTraits>::~IntersectorLDS() {
        alloc_.deallocate(stack_);
        alloc_.deallocate(bvh_local_);
        alloc_.deallocate(bvh_staging_);
        device_.destroyPipelineLayout(pipeline_layout_);
        device_.destroyPipeline(pipeline_);
        device_.destroyDescriptorSetLayout(desc_layout_);
        device_.destroyShaderModule(shader_);
    }

    template <typename BVH, typename BVHTraits>
    inline
    vk::CommandBuffer IntersectorLDS<BVH, BVHTraits>::Commit(World const& world) {
        BVH bvh;

        // Build BVH
        bvh.Build(world.cbegin(), world.cend());

        // Calculate BVH size
        auto bvh_size_in_bytes = BVHTraits::GetSizeInBytes(bvh);

#ifdef TEST
        std::cout << "BVH size is " << bvh_size_in_bytes / 1024.f / 1024.f << "MB";
#endif
        // Check if we have enough space for BVH 
        // and realloc buffers if necessary
        CheckAndReallocBVH(bvh_size_in_bytes);

        // Map staging buffer
        auto ptr = device_.mapMemory(
                bvh_staging_.memory,
                bvh_staging_.offset,
                bvh_size_in_bytes);

        // Copy BVH data
        BVHTraits::StreamBVH(bvh, ptr);

        auto mapped_range = vk::MappedMemoryRange{}
            .setMemory(bvh_staging_.memory)
            .setOffset(bvh_staging_.offset)
            .setSize(bvh_size_in_bytes);

        // Flush range
        device_.flushMappedMemoryRanges(mapped_range);
        device_.unmapMemory(bvh_staging_.memory);

        // Allocate command buffer
        vk::CommandBufferAllocateInfo cmdbuffer_alloc_info;
        cmdbuffer_alloc_info
            .setCommandBufferCount(1)
            .setCommandPool(cmdpool_)
            .setLevel(vk::CommandBufferLevel::ePrimary);

        auto cmdbuffers
            = device_.allocateCommandBuffers(cmdbuffer_alloc_info);

        // Begin command buffer
        vk::CommandBufferBeginInfo cmdbuffer_buffer_begin_info;
        cmdbuffer_buffer_begin_info
            .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmdbuffers[0].begin(cmdbuffer_buffer_begin_info);

        // Copy BVH data from staging to local
        vk::BufferCopy cmd_copy;
        cmd_copy.setSize(bvh_size_in_bytes);
        cmdbuffers[0].copyBuffer(
            bvh_staging_.buffer,
            bvh_local_.buffer,
            cmd_copy);

        // Issue memory barrier for BVH data
        vk::BufferMemoryBarrier memory_barrier;
        memory_barrier
            .setBuffer(bvh_local_.buffer)
            .setOffset(0)
            .setSize(bvh_size_in_bytes)
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

        cmdbuffers[0].pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlags{},
            nullptr,
            memory_barrier,
            nullptr
        );

        // End command buffer
        cmdbuffers[0].end();

        return cmdbuffers[0];
    }

    template <typename BVH, typename BVHTraits>
    inline
    void IntersectorLDS<BVH, BVHTraits>::BindBuffers(
        vk::Buffer rays,
        vk::Buffer hits,
        std::uint32_t num_rays) {

        // Check if we have enough stack memory
        auto stack_size_in_bytes = 
            num_rays * kGlobalStackSize * sizeof(std::uint32_t);
        CheckAndReallocStackBuffer(stack_size_in_bytes);

        vk::DescriptorBufferInfo desc_buffer_info[kNumBindings];
        desc_buffer_info[0]
            .setBuffer(rays)
            .setOffset(0)
            .setRange(num_rays * sizeof(Ray));
        desc_buffer_info[1]
            .setBuffer(hits)
            .setOffset(0)
            .setRange(num_rays * sizeof(Hit));
        desc_buffer_info[2]
            .setBuffer(bvh_local_.buffer)
            .setOffset(0)
            // TODO: should be exact bvh_size_in_bytes here,
            // but we do not keep it around.
            .setRange(bvh_local_.size);
        desc_buffer_info[3]
            .setBuffer(stack_.buffer)
            .setOffset(0)
            .setRange(stack_.size);

        vk::WriteDescriptorSet desc_writes;
        desc_writes
            .setDescriptorCount(kNumBindings)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDstSet(descsets_[0])
            .setDstBinding(0)
            .setPBufferInfo(&desc_buffer_info[0]);

        device_.updateDescriptorSets(desc_writes, nullptr);
    }

    template <typename BVH, typename BVHTraits>
    inline
    vk::CommandBuffer IntersectorLDS<BVH, BVHTraits>::TraceRays(std::uint32_t num_rays) {
        // Allocate command buffer
        vk::CommandBufferAllocateInfo cmdbuffer_alloc_info;
        cmdbuffer_alloc_info
            .setCommandBufferCount(1)
            .setCommandPool(cmdpool_)
            .setLevel(vk::CommandBufferLevel::ePrimary);
        auto cmdbuffers = device_.allocateCommandBuffers(cmdbuffer_alloc_info);

        // Begin command buffer recording
        vk::CommandBufferBeginInfo cmdbuffer_begin_info;
        cmdbuffer_begin_info
            .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmdbuffers[0].begin(cmdbuffer_begin_info);

        // Bind intersection pipeline
        cmdbuffers[0].bindPipeline(
            vk::PipelineBindPoint::eCompute,
            pipeline_);

        // Bind descriptor sets
        cmdbuffers[0].bindDescriptorSets(
            vk::PipelineBindPoint::eCompute,
            pipeline_layout_,
            0,
            descsets_,
            nullptr);

        // Push constants
        auto N = static_cast<std::uint32_t>(num_rays);
        cmdbuffers[0].pushConstants(
            pipeline_layout_,
            vk::ShaderStageFlagBits::eCompute,
            0u,
            sizeof(std::uint32_t),
            &N);

        // Dispatch intersection shader
        auto num_groups = (num_rays + kWorgGroupSize - 1) / kWorgGroupSize;
        cmdbuffers[0].dispatch(num_groups, 1, 1);

        // End command buffer
        cmdbuffers[0].end();

        return cmdbuffers[0];
    }
}
