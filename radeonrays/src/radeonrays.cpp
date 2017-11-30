#include <radeonrays.h>
#include <vulkan/vulkan.hpp>

#include <vector>

#include "vk_mem_manager.h"
#include "world.h"
#include "mesh.h"
#include "bvh.h"
#include "bvh_encoder.h"
#include "utils.h"
#include "vk_utils.h"

#include "qbvh_encoder.h"
//#define FP16

using namespace RadeonRays;

struct Instance {
    // Vulkan entities
    vk::Device device_ = nullptr;
    vk::CommandPool command_pool_ = nullptr;
    vk::PipelineCache pipeline_cache_ = nullptr;
    vk::PipelineLayout pipeline_layout_ = nullptr;
    vk::Pipeline intersect_pipeline_ = nullptr;
    vk::DescriptorPool descriptor_pool_ = nullptr;
    vk::DescriptorSetLayout descriptor_set_layout_ = nullptr;
    vk::ShaderModule isect_shader_module_ = nullptr;
    std::vector<vk::DescriptorSet> descriptor_sets_;
    VulkanMemoryManager::Buffer staging_bvh_buffer_;
    VulkanMemoryManager::Buffer local_bvh_buffer_;
    VulkanMemoryManager::Buffer local_stack_buffer_;

    World world_;
    Bvh<BVHNode, BVHNodeTraits> bvh_;

    // Staging & local memory managers
    std::unique_ptr<VulkanMemoryManager> staging_memory_mgr_;
    std::unique_ptr<VulkanMemoryManager> local_memory_mgr_;
};


static void InitVulkan(
    Instance* instance,
    VkDevice device,
    VkPhysicalDevice physical_device,
    VkCommandPool command_pool) {

    instance->device_ = device;
    instance->command_pool_ = command_pool;

    auto& dev = instance->device_;

    // Allocate memory pools
    instance->staging_memory_mgr_.reset(
        new VulkanMemoryManager(
            dev,
            physical_device,
            vk::MemoryPropertyFlagBits::eHostVisible,
            128 * 1024 * 1024));
    instance->local_memory_mgr_.reset(
        new VulkanMemoryManager(
            dev,
            physical_device,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            512 * 1024 * 1024));

    // Create pipeline cache
    vk::PipelineCacheCreateInfo cache_create_info;
    instance->pipeline_cache_ = dev.createPipelineCache(cache_create_info);

    // Create descriptor pool
    vk::DescriptorPoolSize pool_sizes;
    pool_sizes
        .setType(vk::DescriptorType::eStorageBuffer)
        .setDescriptorCount(4);
    vk::DescriptorPoolCreateInfo descriptor_pool_create_info;
    descriptor_pool_create_info
        .setMaxSets(2)
        .setPoolSizeCount(1)
        .setPPoolSizes(&pool_sizes);
    instance->descriptor_pool_ =
        dev.createDescriptorPool(descriptor_pool_create_info);

    // Create bindings
    // - Ray buffer 
    // - Hit buffer 
    // - BVH buffer
    // - Stack buffer
    vk::DescriptorSetLayoutBinding layout_binding[4];
    layout_binding[0]
        .setBinding(0)
        .setDescriptorCount(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    layout_binding[1]
        .setBinding(1)
        .setDescriptorCount(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    layout_binding[2]
        .setBinding(2)
        .setDescriptorCount(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    layout_binding[3]
        .setBinding(3)
        .setDescriptorCount(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setStageFlags(vk::ShaderStageFlagBits::eCompute);

    // Create descriptor set layout
    vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info;
    descriptor_set_layout_create_info
        .setBindingCount(4)
        .setPBindings(layout_binding);
    instance->descriptor_set_layout_
        = dev.createDescriptorSetLayout(descriptor_set_layout_create_info);

    // Allocate descriptors
    vk::DescriptorSetAllocateInfo desc_alloc_info;
    desc_alloc_info.setDescriptorPool(instance->descriptor_pool_);
    desc_alloc_info.setDescriptorSetCount(1);
    desc_alloc_info.setPSetLayouts(&instance->descriptor_set_layout_);
    instance->descriptor_sets_ = dev.allocateDescriptorSets(desc_alloc_info);

    // Ray count is a push constant, so create range for it
    vk::PushConstantRange push_constant_range;
    push_constant_range.setOffset(0)
        .setSize(4u)
        .setStageFlags(vk::ShaderStageFlagBits::eCompute);

    // Create pipeline layout
    vk::PipelineLayoutCreateInfo pipeline_layout_create_info;
    pipeline_layout_create_info
        .setSetLayoutCount(1)
        .setPSetLayouts(&instance->descriptor_set_layout_)
        .setPushConstantRangeCount(1)
        .setPPushConstantRanges(&push_constant_range);
    instance->pipeline_layout_ =
        dev.createPipelineLayout(pipeline_layout_create_info);

    // Load intersection shader module

#ifdef FP16
    instance->isect_shader_module_ =
        LoadShaderModule(dev, "../../shaders/isect_fp16.comp.spv");
#else
    instance->isect_shader_module_ =
        LoadShaderModule(dev, "../../shaders/isect.comp.spv");
#endif

    // Create pipeline 
    vk::PipelineShaderStageCreateInfo shader_stage_create_info;
    shader_stage_create_info
        .setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(instance->isect_shader_module_)
        .setPName("main");
    vk::ComputePipelineCreateInfo pipeline_create_info;
    pipeline_create_info
        .setLayout(instance->pipeline_layout_)
        .setStage(shader_stage_create_info);
    instance->intersect_pipeline_ = 
        instance->device_.createComputePipeline(
            instance->pipeline_cache_,
            pipeline_create_info);

    instance->local_stack_buffer_ =
        instance->local_memory_mgr_->CreateBuffer(
            1024u * 1024u * 32u * sizeof(std::uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer
        );
}


static void ShutdownVulkan(Instance* instance) {
    auto& dev = instance->device_;
    dev.destroyShaderModule(instance->isect_shader_module_);
    dev.destroyPipeline(instance->intersect_pipeline_);
    dev.destroyPipelineCache(instance->pipeline_cache_);
    dev.destroyDescriptorPool(instance->descriptor_pool_);
    dev.destroyDescriptorSetLayout(instance->descriptor_set_layout_);
    dev.destroyPipelineLayout(instance->pipeline_layout_);
}

rr_status rrInitInstance(
    VkDevice device,
    VkPhysicalDevice physical_device,
    VkCommandPool command_pool,
    rr_instance* out_instance) {
    if (!device || !command_pool) {
        *out_instance = nullptr;
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = new Instance {};
    InitVulkan(instance, device, physical_device, command_pool);

    *out_instance = reinterpret_cast<rr_instance>(instance);
    return RR_SUCCESS;
}

rr_status rrIntersect(
    rr_instance inst,
    uint32_t num_rays,
    VkCommandBuffer* out_command_buffer) {
    auto instance = reinterpret_cast<Instance*>(inst);
    auto& dev = instance->device_;

    // Allocate command buffer
    vk::CommandBufferAllocateInfo cmdbuffer_alloc_info;
    cmdbuffer_alloc_info
        .setCommandBufferCount(1)
        .setCommandPool(instance->command_pool_)
        .setLevel(vk::CommandBufferLevel::ePrimary);
    auto cmd_buffers = dev.allocateCommandBuffers(cmdbuffer_alloc_info);

    // Begin command buffer recording
    vk::CommandBufferBeginInfo cmdbuffer_begin_info;
    cmdbuffer_begin_info
        .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmd_buffers[0].begin(cmdbuffer_begin_info);

    // Bind intersection pipeline
    cmd_buffers[0].bindPipeline(
        vk::PipelineBindPoint::eCompute,
        instance->intersect_pipeline_);

    // Bind descriptor sets
    cmd_buffers[0].bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        instance->pipeline_layout_,
        0,
        instance->descriptor_sets_,
        nullptr);

    // Push constants
    auto N = static_cast<std::uint32_t>(num_rays);
    cmd_buffers[0].pushConstants(
        instance->pipeline_layout_, 
        vk::ShaderStageFlagBits::eCompute, 
        0u,
        sizeof(std::uint32_t),
        &N);

    // Dispatch intersection shader
    auto num_groups = (num_rays + 63) / 64;
    cmd_buffers[0].dispatch(num_groups, 1, 1);

    // End command buffer
    cmd_buffers[0].end();

    *out_command_buffer = cmd_buffers[0];
    return RR_SUCCESS;
}

rr_status rrOccluded(
    rr_instance instance,
    VkBuffer ray_buffer,
    VkBuffer hit_buffer,
    uint32_t num_rays,
    VkCommandBuffer* out_command_buffer) {
    return RR_ERROR_NOT_IMPLEMENTED;
}

rr_status rrShutdownInstance(rr_instance inst) {
    if (!inst) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);

    if (instance->local_bvh_buffer_.buffer) {
        instance->device_.destroyBuffer(instance->local_bvh_buffer_.buffer);
        instance->device_.destroyBuffer(instance->staging_bvh_buffer_.buffer);
    }

    instance->device_.destroyBuffer(instance->local_stack_buffer_.buffer);

    ShutdownVulkan(instance);

    delete instance;
    return RR_SUCCESS;
}

rr_status rrCreateTriangleMesh(
    rr_instance inst,
    float const* vertices,
    std::uint32_t num_vertices,
    std::uint32_t vertex_stride,
    std::uint32_t const* indices,
    std::uint32_t index_stride,
    std::uint32_t num_faces,
    std::uint32_t id,
    rr_shape* out_shape
) {
    if (!inst || !out_shape || !indices || !out_shape) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);
    auto mesh = new Mesh(
        vertices,
        num_vertices,
        vertex_stride,
        indices,
        index_stride,
        num_faces);

    mesh->SetId(id);

    *out_shape = reinterpret_cast<rr_shape>(mesh);

    return RR_SUCCESS;
}

rr_status rrAttachShape(rr_instance inst, rr_shape s) {
    if (!inst || !s) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);
    auto shape = reinterpret_cast<Shape*>(s);
    instance->world_.AttachShape(shape);

    return RR_SUCCESS;
}

rr_status rrDetachShape(rr_instance inst, rr_shape s) {
    if (!inst || !s) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);
    auto shape = reinterpret_cast<Shape*>(s);
    instance->world_.DetachShape(shape);

    return RR_SUCCESS;
}

rr_status rrDetachAllShapes(rr_instance inst) {
    if (!inst) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);
    instance->world_.DetachAll();

    return RR_SUCCESS;
}

rr_status rrCommit(rr_instance inst, VkCommandBuffer* out_command_buffer) {
    if (!inst | !out_command_buffer) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);

    // Build BVH
    instance->bvh_.Clear();
    instance->bvh_.Build(instance->world_.cbegin(), instance->world_.cend());
    BVHNodeTraits::PropagateBounds(instance->bvh_);

#ifdef FP16
    RadeonRays::QBvh qbvh = qbvh.Create(instance->bvh_);
    auto bvh_size_in_bytes
        = qbvh.nodes_.size() * sizeof(QBVHNode);
#else
    auto bvh_size_in_bytes
        = instance->bvh_.num_nodes() * sizeof(BVHNode);
#endif

#ifdef TEST
    std::cout << "BVH size is " << bvh_size_in_bytes / 1024.f / 1024.f << "MB";
#endif

    // Reallocate buffers if needed
    if (bvh_size_in_bytes > instance->local_bvh_buffer_.size) {
        if (instance->local_bvh_buffer_.buffer) {
            instance->device_.destroyBuffer(instance->local_bvh_buffer_.buffer);
            instance->device_.destroyBuffer(instance->staging_bvh_buffer_.buffer);
        }

        instance->staging_bvh_buffer_ = 
            instance->staging_memory_mgr_->CreateBuffer(
                bvh_size_in_bytes,
                vk::BufferUsageFlagBits::eTransferSrc);

        instance->local_bvh_buffer_ =
            instance->local_memory_mgr_->CreateBuffer(
                bvh_size_in_bytes,
                vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eStorageBuffer
            );
    }

    // Map staging buffer

#ifdef FP16
    auto ptr = reinterpret_cast<QBVHNode*>(
        instance->device_.mapMemory(
            instance->staging_bvh_buffer_.memory,
            instance->staging_bvh_buffer_.offset,
            instance->staging_bvh_buffer_.size));

    auto mapped_range = vk::MappedMemoryRange{}
        .setMemory(instance->staging_bvh_buffer_.memory)
        .setOffset(instance->staging_bvh_buffer_.offset)
        .setSize(instance->staging_bvh_buffer_.size);

    // Copy BVH data
    for (auto i = 0u; i < qbvh.nodes_.size(); ++i) {
        ptr[i] = qbvh.nodes_[i];
    }
#else
    auto ptr = reinterpret_cast<BVHNode*>(
        instance->device_.mapMemory(
            instance->staging_bvh_buffer_.memory,
            instance->staging_bvh_buffer_.offset,
            instance->staging_bvh_buffer_.size));

    auto mapped_range = vk::MappedMemoryRange{}
        .setMemory(instance->staging_bvh_buffer_.memory)
        .setOffset(instance->staging_bvh_buffer_.offset)
        .setSize(instance->staging_bvh_buffer_.size);

    // Copy BVH data
    for (auto i = 0u; i < instance->bvh_.num_nodes(); ++i) {
        ptr[i] = *instance->bvh_.GetNode(i);
    }
#endif

    instance->device_.flushMappedMemoryRanges(mapped_range);
    instance->device_.unmapMemory(instance->staging_bvh_buffer_.memory);

    // Allocate command buffer
    vk::CommandBufferAllocateInfo cmdbuffer_alloc_info;
    cmdbuffer_alloc_info
        .setCommandBufferCount(1)
        .setCommandPool(instance->command_pool_)
        .setLevel(vk::CommandBufferLevel::ePrimary);

    auto cmd_buffers
        = instance->device_.allocateCommandBuffers(cmdbuffer_alloc_info);

    // Begin command buffer
    vk::CommandBufferBeginInfo cmdbuffer_buffer_begin_info;
    cmdbuffer_buffer_begin_info
        .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmd_buffers[0].begin(cmdbuffer_buffer_begin_info);

    // Copy BVH data from staging to local
    vk::BufferCopy cmd_copy;
    cmd_copy
        .setSize(instance->staging_bvh_buffer_.size);
    cmd_buffers[0].copyBuffer(
        instance->staging_bvh_buffer_.buffer,
        instance->local_bvh_buffer_.buffer,
        cmd_copy);

    // Issue memory barrier for BVH buffer
    vk::BufferMemoryBarrier memory_barrier;
    memory_barrier
        .setBuffer(instance->local_bvh_buffer_.buffer)
        .setOffset(0)
        .setSize(instance->local_bvh_buffer_.size)
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

    *out_command_buffer = cmd_buffers[0];
    return RR_SUCCESS;
}

rr_status rrDeleteShape(rr_instance inst, rr_shape s) {
    if (!inst || !s) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);
    auto shape = reinterpret_cast<Shape*>(s);

    delete shape;

    return RR_SUCCESS;
}

rr_status rrSetBuffers(
    rr_instance inst,
    VkBuffer ray_buffer,
    VkBuffer hit_buffer,
    uint32_t num_rays
) {
    if (!inst) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);

    // Update descriptor sets
    vk::DescriptorBufferInfo desc_buffer_info[4];
    desc_buffer_info[0]
        .setBuffer(ray_buffer)
        .setOffset(0)
        .setRange(num_rays * sizeof(Ray));
    desc_buffer_info[1]
        .setBuffer(hit_buffer)
        .setOffset(0)
        .setRange(num_rays * sizeof(Hit));
    desc_buffer_info[2]
        .setBuffer(instance->local_bvh_buffer_.buffer)
        .setOffset(0)
        .setRange(instance->local_bvh_buffer_.size);
    desc_buffer_info[3]
        .setBuffer(instance->local_stack_buffer_.buffer)
        .setOffset(0)
        .setRange(instance->local_stack_buffer_.size);

    vk::WriteDescriptorSet desc_writes;
    desc_writes
        .setDescriptorCount(4)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setDstSet(instance->descriptor_sets_[0])
        .setDstBinding(0)
        .setPBufferInfo(&desc_buffer_info[0]);
    instance->device_.updateDescriptorSets(desc_writes, nullptr);

    return RR_SUCCESS;
}