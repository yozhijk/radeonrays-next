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

#include "gtest/gtest.h"

#include <radeonrays.h>

#include <vulkan/vulkan.hpp>

class VulkanMemoryManager {
public:

    struct Buffer {
        vk::Buffer buffer;
        vk::DeviceMemory memory;
        std::size_t offset;
        std::size_t size;
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

class LibTest : public ::testing::Test {
public:

    void SetUp() override {
        auto app_info = vk::ApplicationInfo()
            .setPApplicationName("RadeonRays UnitTest")
            .setApplicationVersion(1)
            .setPEngineName("RadeonRays UnitTest")
            .setEngineVersion(1)
            .setApiVersion(VK_API_VERSION_1_0);

        const char* validation_layers[] = { "VK_LAYER_LUNARG_standard_validation" };
        const char* validation_exts[] = { VK_EXT_DEBUG_REPORT_EXTENSION_NAME };

        auto inst_info = vk::InstanceCreateInfo()
            .setFlags(vk::InstanceCreateFlags())
            .setPApplicationInfo(&app_info)
            .setEnabledExtensionCount(1)
            .setPpEnabledExtensionNames(validation_exts)
            .setEnabledLayerCount(1)
            .setPpEnabledLayerNames(validation_layers);

        instance_ = vk::createInstance(inst_info);
        auto physical_devices = instance_.enumeratePhysicalDevices();
        ASSERT_GT(physical_devices.size(), 0);

        physical_device_ = physical_devices[0];
        auto queue_properties = physical_device_.getQueueFamilyProperties();

        int idx = -1;
        for (auto i = 0u; i < queue_properties.size(); ++i) {
            if (queue_properties[i].queueFlags & vk::QueueFlagBits::eCompute) {
                idx = i;
                break;
            }
        }

        float default_priority = 1.f;
        auto queue_create_info = vk::DeviceQueueCreateInfo()
            .setQueueFamilyIndex(idx)
            .setQueueCount(1)
            .setPQueuePriorities(&default_priority);
        auto device_create_info = vk::DeviceCreateInfo()
            .setQueueCreateInfoCount(1)
            .setPQueueCreateInfos(&queue_create_info);
        device_ = physical_device_.createDevice(device_create_info);
        queue_ = device_.getQueue(idx, 0);

        auto cmd_pool_info = vk::CommandPoolCreateInfo{}
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
            .setQueueFamilyIndex(idx);
        command_pool_ = device_.createCommandPool(cmd_pool_info);


        auto status = rrInitInstance(device_, command_pool_, &rr_instance_);
        ASSERT_EQ(status, RR_SUCCESS);

        m_staging_mgr = std::make_unique<VulkanMemoryManager>(
            device_,
            physical_device_,
            vk::MemoryPropertyFlagBits::eHostVisible,
            128 * 1024 * 1024);

        m_local_mgr = std::make_unique<VulkanMemoryManager>(
            device_,
            physical_device_,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            512 * 1024 * 1024);
    }

    void TearDown() override {
        auto status = rrShutdownInstance(rr_instance_);
        ASSERT_EQ(status, RR_SUCCESS);

        m_local_mgr.reset(nullptr);
        m_staging_mgr.reset(nullptr);

        device_.destroyCommandPool(command_pool_);
        device_.destroy();
        instance_.destroy();
    }

    rr_instance rr_instance_;
    vk::Instance instance_;
    vk::PhysicalDevice physical_device_;
    vk::Device device_;
    vk::CommandPool command_pool_;
    vk::Queue queue_;

    std::unique_ptr<VulkanMemoryManager> m_staging_mgr;
    std::unique_ptr<VulkanMemoryManager> m_local_mgr;
};

TEST_F(LibTest, Init) {
    auto staging_buffer = m_staging_mgr->CreateBuffer(
        128 * 1024 * 1024,
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eTransferSrc);

    auto device_buffer = m_local_mgr->CreateBuffer(
        128 * 1024 * 1024,
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eTransferSrc |
        vk::BufferUsageFlagBits::eStorageBuffer);

    device_.destroyBuffer(staging_buffer.buffer);
    device_.destroyBuffer(device_buffer.buffer);
}

TEST_F(LibTest, InitBuffers) {
    std::vector<int> data(512);
    std::vector<int> result(512);
    std::iota(data.begin(), data.end(), 0);

    auto staging_buffer = m_staging_mgr->CreateBuffer(
        512 * sizeof(int),
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eTransferSrc);

    auto device_buffer = m_local_mgr->CreateBuffer(
        512 * sizeof(int),
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eTransferSrc |
        vk::BufferUsageFlagBits::eStorageBuffer);

    auto ptr = reinterpret_cast<int*>(
        device_.mapMemory(
            staging_buffer.memory,
            staging_buffer.offset,
            staging_buffer.size));

    ASSERT_NE(ptr, nullptr);

    for (int i = 0; i < 512; ++i) {
        ptr[i] = data[i];
    }

    auto mapped_range = vk::MappedMemoryRange{}
        .setMemory(staging_buffer.memory)
        .setOffset(staging_buffer.offset)
        .setSize(staging_buffer.size);

    device_.flushMappedMemoryRanges(mapped_range);
    device_.unmapMemory(staging_buffer.memory);

    auto cmd_allocate_info = vk::CommandBufferAllocateInfo{}
        .setCommandBufferCount(2)
        .setCommandPool(command_pool_)
        .setLevel(vk::CommandBufferLevel::ePrimary);

    auto cmd_buffers = device_.allocateCommandBuffers(cmd_allocate_info);
    auto fence_create_info = vk::FenceCreateInfo{};
    auto fence = device_.createFence(fence_create_info);

    {
        auto cmd_buffer_begin_info = vk::CommandBufferBeginInfo{}
        .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd_buffers[0].begin(cmd_buffer_begin_info);
        auto cmd_copy = vk::BufferCopy{}
        .setSize(staging_buffer.size);
        cmd_buffers[0].copyBuffer(staging_buffer.buffer, device_buffer.buffer, cmd_copy);
        cmd_buffers[0].end();

        auto queue_submit_info = vk::SubmitInfo{}
            .setCommandBufferCount(1)
            .setPCommandBuffers(&cmd_buffers[0]);

        queue_.submit(queue_submit_info, nullptr);
    }


    VkCommandBuffer temp;
    auto status = rrIntersect(rr_instance_, device_buffer.buffer, device_buffer.buffer, 512, &temp);
    ASSERT_EQ(status, RR_SUCCESS);
    ASSERT_NE(temp, nullptr);
    vk::CommandBuffer rt_buffer(temp);

    {
        auto queue_submit_info = vk::SubmitInfo{}
            .setCommandBufferCount(1)
            .setPCommandBuffers(&rt_buffer);
        queue_.submit(queue_submit_info, nullptr);
    }

    {
        auto cmd_buffer_begin_info = vk::CommandBufferBeginInfo{}
        .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd_buffers[1].begin(cmd_buffer_begin_info);
        auto cmd_copy = vk::BufferCopy{}
        .setSize(device_buffer.size);
        cmd_buffers[1].copyBuffer(
            device_buffer.buffer,
            staging_buffer.buffer, cmd_copy);

        vk::BufferMemoryBarrier barrier = vk::BufferMemoryBarrier{}
            .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
            .setDstAccessMask(vk::AccessFlagBits::eHostRead)
            .setBuffer(device_buffer.buffer)
            .setSize(device_buffer.size)
            .setOffset(0);

        cmd_buffers[1].pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eHost,
            vk::DependencyFlags{},
            0,
            barrier,
            0);

        cmd_buffers[1].end();

        auto queue_submit_info = vk::SubmitInfo{}
            .setCommandBufferCount(1)
            .setPCommandBuffers(&cmd_buffers[1]);
        queue_.submit(queue_submit_info, fence);
        device_.waitForFences(fence, true, std::numeric_limits<std::uint32_t>::max());
        device_.resetFences(fence);
    }

    {
        auto ptr = reinterpret_cast<int*>(
            device_.mapMemory(
                staging_buffer.memory,
                staging_buffer.offset,
                staging_buffer.size));

        auto mapped_range = vk::MappedMemoryRange{}
            .setMemory(staging_buffer.memory)
            .setOffset(staging_buffer.offset)
            .setSize(staging_buffer.size);

        device_.invalidateMappedMemoryRanges(mapped_range);

        ASSERT_NE(ptr, nullptr);

        for (int i = 0; i < 512; ++i) {
            result[i] = ptr[i];
        }

        device_.unmapMemory(staging_buffer.memory);
    }

    for (int i = 0; i < 512; ++i) {
        ASSERT_EQ(result[i], data[i] + 1);
    }

    device_.freeCommandBuffers(command_pool_, rt_buffer);
    device_.destroyFence(fence);
    device_.freeCommandBuffers(command_pool_, cmd_buffers);
    device_.destroyBuffer(staging_buffer.buffer);
    device_.destroyBuffer(device_buffer.buffer);
}

