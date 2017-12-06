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
#include "vk_mem_manager.h"
#include "world.h"
#include "bvh.h"
#include "bvh_encoder.h"

namespace RadeonRays {
    struct Instance {
        // Vulkan device to run queries on
        vk::Device device_ = nullptr;
        // Command pool to allocate command buffers
        vk::CommandPool command_pool_ = nullptr;
        //
        vk::PipelineCache pipeline_cache_ = nullptr;
        // Pipeline layout (currently shared)
        vk::PipelineLayout pipeline_layout_ = nullptr;
        // Intersect pipeline
        vk::Pipeline intersect_pipeline_ = nullptr;
        // Descriptor pool for RR descriptor sets
        vk::DescriptorPool descriptor_pool_ = nullptr;
        // Descriptor layouts (currently shared)
        vk::DescriptorSetLayout descriptor_set_layout_ = nullptr;
        // Intersect shader module
        vk::ShaderModule isect_shader_module_ = nullptr;
        // Descriptor sets
        std::vector<vk::DescriptorSet> descriptor_sets_;

        // RR buffers
        // Staging buffer to copy BVH data
        VulkanMemoryManager::Buffer staging_bvh_buffer_;
        // Local buffer for BVH data
        VulkanMemoryManager::Buffer local_bvh_buffer_;
        // Buffer for local memory stacks
        VulkanMemoryManager::Buffer local_stack_buffer_;

        // World keeps set of shapes currently bound
        World world_;
        // Current BVH
        Bvh<BVHNode, BVHNodeTraits> bvh_;

        // Staging & local memory managers
        std::unique_ptr<VulkanMemoryManager> staging_memory_mgr_;
        std::unique_ptr<VulkanMemoryManager> local_memory_mgr_;
    };
}

