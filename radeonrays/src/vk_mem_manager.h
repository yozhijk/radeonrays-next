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

class VulkanMemoryManager {
public:

    struct Buffer {
        vk::Buffer buffer = nullptr;
        vk::DeviceMemory memory = nullptr;
        std::size_t offset = 0;
        std::size_t size = 0;
    };

    VulkanMemoryManager(
        vk::Device device,
        vk::PhysicalDevice physical_device,
        vk::MemoryPropertyFlags property_flags,
        std::size_t pool_size)
        : device_(device)
        , physical_device_(physical_device)
        , pool_size_(pool_size) {
        auto mem_props = physical_device_.getMemoryProperties();

        int mem_type_index = -1;
        for (auto i = 0u; i < mem_props.memoryTypeCount; i++) {
            auto& memory_type = mem_props.memoryTypes[i];
            if ((memory_type.propertyFlags & property_flags) == property_flags) {
                if (pool_size <= mem_props.memoryHeaps[memory_type.heapIndex].size) {
                    mem_type_index = i;
                    break;
                }
            }
        }

        if (mem_type_index < 0) {
            throw std::runtime_error("Not enough device memory");
        }

        auto alloc_info = vk::MemoryAllocateInfo{}
            .setAllocationSize(pool_size_)
            .setMemoryTypeIndex(mem_type_index);

        memory_ = device_.allocateMemory(alloc_info);
    }

    ~VulkanMemoryManager() {
        Release();
    }

    Buffer CreateBuffer(std::size_t size, vk::BufferUsageFlags usage_flags) {

        if (pool_size_ - next_free_index_ < size) {
            throw std::runtime_error("Not enough device memory in the pool");
        }

        Buffer wrapper;

        auto buffer_create_info = vk::BufferCreateInfo{}
            .setUsage(usage_flags)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setSize(size);

        auto buffer = device_.createBuffer(buffer_create_info);

        device_.bindBufferMemory(buffer, memory_, next_free_index_);

        wrapper.buffer = buffer;
        wrapper.memory = memory_;
        wrapper.offset = next_free_index_;
        wrapper.size = size;

        next_free_index_ += size;

        return wrapper;
    }

    void Release() {
        device_.freeMemory(memory_);
    }

    vk::Device device_;
    vk::PhysicalDevice physical_device_;
    vk::DeviceMemory memory_;
    std::size_t pool_size_;
    std::size_t next_free_index_ = 0;
};