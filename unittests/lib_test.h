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

#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
        if (memory_) {
            device_.freeMemory(memory_);
            memory_ = nullptr;
        }
    }

    vk::Device device_ = nullptr;
    vk::PhysicalDevice physical_device_ = nullptr;
    vk::DeviceMemory memory_ = nullptr;
    std::size_t pool_size_ = 0;
    std::size_t next_free_index_ = 0;
};


class LibTest : public ::testing::Test {
public:
    void TraceRays(std::vector<Ray> const& data, std::vector<Hit>& result);

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


        auto status = rrInitInstance(device_, physical_device_, command_pool_, &rr_instance_);
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

        for (auto& m : meshes_) {
            rrDeleteShape(rr_instance_, m);
        }

        auto status = rrShutdownInstance(rr_instance_);
        ASSERT_EQ(status, RR_SUCCESS);

        m_local_mgr.reset(nullptr);
        m_staging_mgr.reset(nullptr);

        device_.destroyCommandPool(command_pool_);
        device_.destroy();
        instance_.destroy();
    }

    void LoadScene(std::string const& file) {
        std::string err;
        auto ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, file.c_str());
        ASSERT_TRUE(ret);
        ASSERT_GT(shapes.size(), 0u);

        vertices.resize(attrib.vertices.size() / 3);
        for (auto i = 0u; i < attrib.vertices.size() / 3; ++i) {
            vertices[i].x = attrib.vertices[3 * i];
            vertices[i].y = attrib.vertices[3 * i + 1];
            vertices[i].z = attrib.vertices[3 * i + 2];
            vertices[i].w = 1.f;
        }

        attrib.vertices.clear();
        std::uint32_t id = 1u;
        for (auto& shape : shapes) {

            rr_shape mesh = nullptr;
            auto status = rrCreateTriangleMesh(
                rr_instance_, 
                &vertices[0].x,
                (std::uint32_t)vertices.size(),
                sizeof(RadeonRays::float3),
                (std::uint32_t*)&shape.mesh.indices[0].vertex_index,
                (std::uint32_t)sizeof(tinyobj::index_t),
                (std::uint32_t)(shape.mesh.indices.size() / 3),
                id++,
                &mesh);

            ASSERT_EQ(status, RR_SUCCESS);
            meshes_.push_back(mesh);

            status = rrAttachShape(rr_instance_, mesh);
            ASSERT_EQ(status, RR_SUCCESS);
        }
    }

    rr_instance rr_instance_;
    vk::Instance instance_;
    vk::PhysicalDevice physical_device_;
    vk::Device device_;
    vk::CommandPool command_pool_;
    vk::Queue queue_;

    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::vector<RadeonRays::float3> vertices;
    std::vector<rr_shape> meshes_;
    tinyobj::attrib_t attrib;

    std::unique_ptr<VulkanMemoryManager> m_staging_mgr;
    std::unique_ptr<VulkanMemoryManager> m_local_mgr;
};

TEST_F(LibTest, Init) {
    auto rays_staging = m_staging_mgr->CreateBuffer(
        128 * 1024 * 1024,
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eTransferSrc);

    auto rays_local = m_local_mgr->CreateBuffer(
        128 * 1024 * 1024,
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eTransferSrc |
        vk::BufferUsageFlagBits::eStorageBuffer);

    device_.destroyBuffer(rays_staging.buffer);
    device_.destroyBuffer(rays_local.buffer);
}

void LibTest::TraceRays(std::vector<Ray> const& data, std::vector<Hit>& result) {
    auto num_rays = static_cast<std::uint32_t>(data.size());
    // Allocate rays and hits buffer
    auto rays_staging = m_staging_mgr->CreateBuffer(
        num_rays * sizeof(Ray),
        vk::BufferUsageFlagBits::eTransferSrc);
    auto hits_staging = m_staging_mgr->CreateBuffer(
        num_rays * sizeof(Hit),
        vk::BufferUsageFlagBits::eTransferDst);
    auto rays_local = m_local_mgr->CreateBuffer(
        num_rays * sizeof(Ray),
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eStorageBuffer);
    auto hits_local = m_local_mgr->CreateBuffer(
        num_rays * sizeof(Hit),
        vk::BufferUsageFlagBits::eTransferSrc |
        vk::BufferUsageFlagBits::eStorageBuffer);

    // Map rays buffer and fill rays data
    auto ptr = reinterpret_cast<Ray*>(
        device_.mapMemory(
            rays_staging.memory,
            rays_staging.offset,
            rays_staging.size));
    ASSERT_NE(ptr, nullptr);

    for (auto i = 0u; i < num_rays; ++i) {
        ptr[i] = data[i];
    }

    vk::MappedMemoryRange mapped_range;
    mapped_range
        .setMemory(rays_staging.memory)
        .setOffset(rays_staging.offset)
        .setSize(rays_staging.size);
    device_.flushMappedMemoryRanges(mapped_range);
    device_.unmapMemory(rays_staging.memory);


    // Allocate 2 command buffers (for data copy src->dst and back)
    vk::CommandBufferAllocateInfo cmdbuf_allocate_info;
    cmdbuf_allocate_info
        .setCommandBufferCount(2)
        .setCommandPool(command_pool_)
        .setLevel(vk::CommandBufferLevel::ePrimary);
    auto cmd_buffers
        = device_.allocateCommandBuffers(cmdbuf_allocate_info);

    // Create a fence
    vk::FenceCreateInfo fence_create_info;
    auto fence = device_.createFence(fence_create_info);

    {
        // Begin command buffer
        vk::CommandBufferBeginInfo cmdbuffer_begin_info;
        cmdbuffer_begin_info
            .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd_buffers[0].begin(cmdbuffer_begin_info);
        // Issue copy command
        vk::BufferCopy cmd_copy;
        cmd_copy
            .setSize(rays_staging.size);
        cmd_buffers[0].
            copyBuffer(rays_staging.buffer, rays_local.buffer, cmd_copy);

        // Issue barrier for rays buffer host->RR (compute)
        vk::BufferMemoryBarrier memory_barrier;
        memory_barrier
            .setBuffer(rays_local.buffer)
            .setOffset(0)
            .setSize(rays_local.size)
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        cmd_buffers[0].pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlags{},
            nullptr,
            memory_barrier,
            nullptr
        );

        // End command buffer
        cmd_buffers[0].end();

        // Submit to command queue
        vk::SubmitInfo queue_submit_info;
        queue_submit_info
            .setCommandBufferCount(1)
            .setPCommandBuffers(&cmd_buffers[0]);
        queue_.submit(queue_submit_info, nullptr);
    }

    VkCommandBuffer temp0, temp1;
    // Commit geometry to RR
    {
        auto status = rrCommit(rr_instance_, &temp0);
        ASSERT_EQ(status, RR_SUCCESS);
        ASSERT_NE(temp0, nullptr);
        vk::CommandBuffer commit_buffer(temp0);
        vk::SubmitInfo queue_submit_info;
        queue_submit_info
            .setCommandBufferCount(1)
            .setPCommandBuffers(&commit_buffer);
        queue_.submit(queue_submit_info, nullptr);
    }

    // Execute intersection query
    {
        device_.waitIdle();
        using namespace std::chrono;
        auto status = rrIntersect(
            rr_instance_,
            rays_local.buffer,
            hits_local.buffer,
            num_rays,
            &temp1);

        ASSERT_EQ(status, RR_SUCCESS);
        ASSERT_NE(temp1, nullptr);
        vk::CommandBuffer rt_buffer(temp1);

        auto start = high_resolution_clock::now();
        vk::SubmitInfo queue_submit_info;
        queue_submit_info
            .setCommandBufferCount(1)
            .setPCommandBuffers(&rt_buffer);
        queue_.submit(queue_submit_info, fence);
        device_.waitForFences(
            fence,
            true,
            std::numeric_limits<std::uint32_t>::max());
        auto delta = high_resolution_clock::now() - start;
        auto ms = static_cast<float>(duration_cast<milliseconds>(delta).count());
        std::cout << "Ray query time " << ms << "ms\n";
        device_.resetFences(fence);
    }

    // Read hit data back to the host
    {
        vk::CommandBufferBeginInfo cmdbuffer_begin_info;
        cmdbuffer_begin_info
            .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

        // Begin command buffer
        cmd_buffers[1].begin(cmdbuffer_begin_info);

        // Issue barrier
        vk::BufferMemoryBarrier memory_barrier;

        memory_barrier
            .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead)
            .setBuffer(hits_local.buffer)
            .setSize(hits_local.size)
            .setOffset(0);

        cmd_buffers[1].pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eTransfer,
            vk::DependencyFlags{},
            0,
            memory_barrier,
            0);

        // Issue copy command
        vk::BufferCopy cmd_copy;
        cmd_copy
            .setSize(hits_local.size);
        cmd_buffers[1].copyBuffer(
            hits_local.buffer,
            hits_staging.buffer,
            cmd_copy);

        // End command buffer recording
        cmd_buffers[1].end();

        // Submit to the queue
        vk::SubmitInfo queue_submit_info;
        queue_submit_info
            .setCommandBufferCount(1)
            .setPCommandBuffers(&cmd_buffers[1]);
        queue_.submit(queue_submit_info, fence);

        // Wait for the fence
        device_.waitForFences(
            fence,
            true,
            std::numeric_limits<std::uint32_t>::max());
    }

    {
        // Map hits data
        auto ptr = reinterpret_cast<Hit*>(
            device_.mapMemory(
                hits_staging.memory,
                hits_staging.offset,
                hits_staging.size));
        vk::MappedMemoryRange mapped_range;
        mapped_range
            .setMemory(hits_staging.memory)
            .setOffset(hits_staging.offset)
            .setSize(hits_staging.size);
        device_.invalidateMappedMemoryRanges(mapped_range);
        ASSERT_NE(ptr, nullptr);

        for (auto i = 0u; i < num_rays; ++i) {
            result[i] = ptr[i];
        }

        device_.unmapMemory(hits_staging.memory);
    }

    device_.destroyFence(fence);
    device_.freeCommandBuffers(command_pool_, vk::CommandBuffer{ temp0 });
    device_.freeCommandBuffers(command_pool_, vk::CommandBuffer{ temp1 });
    device_.freeCommandBuffers(command_pool_, cmd_buffers);
    device_.destroyBuffer(rays_staging.buffer);
    device_.destroyBuffer(rays_local.buffer);
    device_.destroyBuffer(hits_staging.buffer);
    device_.destroyBuffer(hits_local.buffer);

}

TEST_F(LibTest, CornellBox) {
    LoadScene(CORNELL_BOX);
    auto constexpr kResolution = 1024;

    std::vector<Ray> data(kResolution * kResolution);
    std::vector<Hit> result(kResolution * kResolution);

    for (int x = 0; x < kResolution; ++x) {
        for (int y = 0; y < kResolution; ++y) {
            auto i = kResolution * y + x;

            data[i].origin[0] = 0.f;
            data[i].origin[1] = 1.f;
            data[i].origin[2] = 3.f;

            data[i].direction[0] = -1.f + (2.f / kResolution) * x;
            data[i].direction[1] = -1.f + (2.f / kResolution) * y;
            data[i].direction[2] = -1.f;

            data[i].max_t = 100000.f;
        }
    }

    TraceRays(data, result);

    std::vector<std::uint8_t> imgdata(kResolution * kResolution * 3);

    std::fill(imgdata.begin(), imgdata.end(), 20u);

    for (auto x = 0u; x < kResolution; ++x) {
        for (auto y = 0u; y < kResolution; ++y) {
            auto i = kResolution * y + x;
            auto j = kResolution * (kResolution - 1 - y) + x;

            if (result[i].shape_id != RR_INVALID_ID) {
                imgdata[3 * j] = (std::uint8_t)(200u * result[i].uv[0]);
                imgdata[3 * j + 1] = (std::uint8_t)(200u * result[i].uv[1]);
                imgdata[3 * j + 2] = 200u;
            }
        }
    }

    stbi_write_jpg("CornellBox.jpg", kResolution, kResolution, 3, &imgdata[0], 10);
}

TEST_F(LibTest, Sponza) {

    LoadScene(CRYTEK_SPONZA);
    auto constexpr kResolution = 1024;

    std::vector<Ray> data(kResolution * kResolution);
    std::vector<Hit> result(kResolution * kResolution);

    for (int x = 0; x < kResolution; ++x) {
        for (int y = 0; y < kResolution; ++y) {
            auto i = kResolution * y + x;

            data[i].origin[0] = 0.f;
            data[i].origin[1] = 200.f;
            data[i].origin[2] = 0.f;

            data[i].direction[0] = -1.f;
            data[i].direction[1] = -1.f + (2.f / kResolution) * y;
            data[i].direction[2] = -1.f + (2.f / kResolution) * x;

            data[i].max_t = 100000.f;
        }
    }

    TraceRays(data, result);

    std::vector<std::uint8_t> imgdata(kResolution * kResolution * 3);

    std::fill(imgdata.begin(), imgdata.end(), 20u);

    for (auto x = 0u; x < kResolution; ++x) {
        for (auto y = 0u; y < kResolution; ++y) {
            auto i = kResolution * y + x;
            auto j = kResolution * (kResolution - 1 - y) + x;

            if (result[i].shape_id != RR_INVALID_ID) {
                imgdata[3 * j] = (std::uint8_t)(200u * result[i].uv[0]);
                imgdata[3 * j + 1] = (std::uint8_t)(200u * result[i].uv[1]);
                imgdata[3 * j + 2] = 200u;
            }
        }
    }

    stbi_write_jpg("Sponza.jpg", kResolution, kResolution, 3, &imgdata[0], 10);
}

