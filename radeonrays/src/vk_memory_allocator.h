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
#include <unordered_map>

struct VkMemoryAlloc {
    static std::size_t constexpr kChunkSize = 256 * 1024 * 1024;
    static std::size_t align(std::size_t value, std::size_t alignment) {
        return (value + (alignment - 1)) / alignment * alignment;
    }

    struct StorageBlock {
        vk::DeviceMemory memory;
        mutable vk::Buffer buffer;
        vk::DeviceSize offset;
        vk::DeviceSize size;
        int memory_type_index;

        StorageBlock(
            vk::DeviceMemory m = nullptr,
            vk::Buffer b = nullptr,
            vk::DeviceSize o = 0u,
            vk::DeviceSize s = 0u,
            int midx = -1)
            : memory(m)
            , buffer(b)
            , offset(o)
            , size(s)
            , memory_type_index(midx) {}
    };

    VkMemoryAlloc(
        vk::Device device,
        vk::PhysicalDevice physical_device)
        : device_(device)
        , physical_device_(physical_device)
        , memory_props_(physical_device.getMemoryProperties()) {}

    ~VkMemoryAlloc() {
        ReleaseMemory();
    }

    StorageBlock allocate(
        vk::MemoryPropertyFlags type,
        vk::BufferUsageFlags usage,
        std::size_t size,
        std::size_t alignment);

    void deallocate(StorageBlock const& block);

private:
    // Find memory type index corresponding to specified flags
    int FindMemoryTypeIndex(vk::MemoryPropertyFlags type) {
        int index = -1;
        for (auto i = 0u; i < memory_props_.memoryTypeCount; i++) {
            auto& memory_type = memory_props_.memoryTypes[i];
            if ((memory_type.propertyFlags & type) == type) {
                index = i;
                break;
            }
        }

        return index;
    }

    // Release all the memory
    void ReleaseMemory() {
        for (auto& h : alloc_headers_) {
            for (auto& m : h.second.memories_) {
                device_.freeMemory(m);
            }
        }
    }

    struct AllocHeader {
        int mem_type_index;
        std::list<StorageBlock> free_blocks_;
        std::list<vk::DeviceMemory> memories_;
    };

    vk::Device device_;
    vk::PhysicalDevice physical_device_;
    vk::PhysicalDeviceMemoryProperties memory_props_;
    std::unordered_map<int, AllocHeader> alloc_headers_;
};

inline VkMemoryAlloc::StorageBlock VkMemoryAlloc::allocate(
    vk::MemoryPropertyFlags type,
    vk::BufferUsageFlags usage,
    std::size_t size,
    std::size_t alignment) {
    // Try to find existing header for a given memory type flags
    auto memory_type_index = FindMemoryTypeIndex(type);
    auto iter = alloc_headers_.find(memory_type_index);

    // Insert new header if we could find existing one
    if (iter == alloc_headers_.cend()) {
        AllocHeader header;
        // Find corresponding memory type index
        header.mem_type_index = memory_type_index;

        if (header.mem_type_index == -1) {
            throw std::runtime_error("Cannot find specified memory type");
        }

        // Emplace header
        auto emp = alloc_headers_.emplace(memory_type_index, header);
        iter = emp.first;
    }

    // Here we have a valid header
    auto& header = iter->second;
    // Try to find free block in the list of free blocks
    // taking requested alignment into account
    auto free_block_iter = std::find_if(
        header.free_blocks_.cbegin(),
        header.free_blocks_.cend(),
        [alignment, size](StorageBlock const& block) ->bool {
        // Aligned offset
        auto aligned_offset = align(block.offset, alignment);
        // Size adjusted according w/ alignmnet difference
        auto aligned_size = block.size - (aligned_offset - block.offset);
        return aligned_size >= size;
    });

    // If we have not found a free block, we 
    // allocate new vk::DeviceMemory and create free block
    // out of it.
    if (free_block_iter == header.free_blocks_.cend()) {
        // We round up to the size of minimum chunk
        auto memory_size = align(size, kChunkSize);
        auto alloc_info = vk::MemoryAllocateInfo{}
            .setAllocationSize(memory_size)
            .setMemoryTypeIndex(header.mem_type_index);

        auto memory = device_.allocateMemory(alloc_info);
        // Keep the memory in the list of memories
        header.memories_.push_back(memory);
        // Create free block covering the whole memory
        header.free_blocks_.emplace_back(
            memory,
            nullptr,
            (vk::DeviceSize)0u,
            (vk::DeviceSize)memory_size,
            header.mem_type_index);
        // Recover out iterator
        free_block_iter = header.free_blocks_.cend();
        --free_block_iter;
    }

    // Here we have guaranteed free block
    auto free_block = *free_block_iter;
    // Copy it since we might have to split this block
    // (in case requested memory size is less, than the size of the block)
    auto rest_block = free_block;
    // Remove from the list of free blocks
    header.free_blocks_.erase(free_block_iter);

    // Adjust block offset w/ alignment
    free_block.offset = align(free_block.offset, alignment);
    // Set block size
    free_block.size = size;

    // We have rest_block.size - free_block.size memory left in the block
    // so we put it into new block and insert.
    auto memory_left_in_block = rest_block.size - free_block.size;
    if (memory_left_in_block > 0) {
        rest_block.offset = free_block.offset + free_block.size;
        rest_block.size = memory_left_in_block;
        header.free_blocks_.push_back(rest_block);
    }

    auto buffer_create_info = vk::BufferCreateInfo{}
        .setUsage(usage)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setSize(free_block.size);
    free_block.buffer = device_.createBuffer(buffer_create_info);
    device_.bindBufferMemory(
        free_block.buffer,
        free_block.memory,
        free_block.offset);

    return free_block;
}

inline void VkMemoryAlloc::deallocate(StorageBlock const& block) {
    if (block.size == 0) {
        return;
    }

    auto iter = alloc_headers_.find(block.memory_type_index);

    // Insert new header if we could find existing one
    if (iter == alloc_headers_.cend()) {
        AllocHeader header;
        // Find corresponding memory type index
        header.mem_type_index = block.memory_type_index;
        // Emplace header
        auto emp = alloc_headers_.emplace(block.memory_type_index, header);
        iter = emp.first;
    }

    // Here we have a valid header
    auto& header = iter->second;

    // Release buffer
    device_.destroyBuffer(block.buffer);
    block.buffer = nullptr;

    // Return block to the list of free blocks
    header.free_blocks_.push_back(block);
}