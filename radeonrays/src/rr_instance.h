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
#include "world.h"

namespace RadeonRays
{
    class Intersector;
    struct Instance
    {
        // Vulkan device to run queries on
        vk::Device device = nullptr;
        // Command pool to allocate command buffers
        vk::CommandPool cmd_pool = nullptr;
        // Pipeline cache
        vk::PipelineCache pipeline_cache = nullptr;
        // Descriptor pool for RR descriptor sets
        vk::DescriptorPool desc_pool = nullptr;
        // Allocator
        std::unique_ptr<VkMemoryAlloc> alloc = nullptr;
        // Intersector
        std::unique_ptr<Intersector> intersector = nullptr;

        // World keeps set of shapes currently bound
        World world;
    };
}

