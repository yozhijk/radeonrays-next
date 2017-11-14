#include <radeonrays.h>
#include <vulkan/vulkan.hpp>
#include "world.h"
#include "mesh.h"

#include <fstream>

using namespace RadeonRays;

struct Instance {
    vk::Device device_ = nullptr;
    vk::CommandPool command_pool_ = nullptr;
    vk::PipelineCache pipeline_cache_ = nullptr;
    vk::PipelineLayout pipeline_layout_ = nullptr;
    vk::Pipeline intersect_pipeline_ = nullptr;
    vk::DescriptorPool desc_pool_ = nullptr;
    vk::DescriptorSetLayout desc_layout_ = nullptr;
    vk::ShaderModule isect_module_ = nullptr;
    std::vector<vk::DescriptorSet> desc_sets_;
    World world_;
};

static
void LoadFileContents(std::string const& name, std::vector<char>& contents, bool binary = false)
{
    std::ifstream in(name, std::ios::in | (std::ios_base::openmode)(binary ? std::ios::binary : 0));

    if (in)
    {
        contents.clear();
        std::streamoff beg = in.tellg();
        in.seekg(0, std::ios::end);
        std::streamoff fileSize = in.tellg() - beg;
        in.seekg(0, std::ios::beg);
        contents.resize(static_cast<unsigned>(fileSize));
        in.read(&contents[0], fileSize);
    }
    else
    {
        throw std::runtime_error("Cannot read the contents of a file");
    }
}

static vk::ShaderModule LoadShaderModule(vk::Device device, std::string const& file_name)
{
    std::vector<char> bytecode;
    LoadFileContents(file_name, bytecode, true);

    vk::ShaderModuleCreateInfo info;
    info.setCodeSize(bytecode.size())
        .setPCode(
            reinterpret_cast<std::uint32_t*>(&bytecode[0]));

    return device.createShaderModule(info);
}

static void InitVulkan(
    Instance* instance,
    VkDevice device,
    VkCommandPool command_pool) {

    instance->device_ = device;
    instance->command_pool_ = command_pool;

    auto& dev = instance->device_;

    vk::PipelineCacheCreateInfo cache_create_info;
    instance->pipeline_cache_ = dev.createPipelineCache(cache_create_info);

    vk::DescriptorPoolSize pool_sizes;
    pool_sizes
        .setType(vk::DescriptorType::eStorageBuffer)
        .setDescriptorCount(10);
    vk::DescriptorPoolCreateInfo descriptor_pool_create_info;
    descriptor_pool_create_info
        .setMaxSets(5)
        .setPoolSizeCount(1)
        .setPPoolSizes(&pool_sizes);
    instance->desc_pool_ = dev.createDescriptorPool(descriptor_pool_create_info);

    vk::DescriptorSetLayoutBinding layout_binding;
    layout_binding
        .setBinding(0)
        .setDescriptorCount(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setStageFlags(vk::ShaderStageFlagBits::eCompute);

    vk::DescriptorSetLayoutCreateInfo desc_layout_create_info;
    desc_layout_create_info
        .setBindingCount(1)
        .setPBindings(&layout_binding);
    instance->desc_layout_ = dev.createDescriptorSetLayout(desc_layout_create_info);

    vk::DescriptorSetAllocateInfo desc_alloc_info;
    desc_alloc_info.setDescriptorPool(instance->desc_pool_);
    desc_alloc_info.setDescriptorSetCount(1);
    desc_alloc_info.setPSetLayouts(&instance->desc_layout_);
    instance->desc_sets_ = dev.allocateDescriptorSets(desc_alloc_info);

    vk::PushConstantRange push_constant_range;
    push_constant_range.setOffset(0)
        .setSize(4u)
        .setStageFlags(vk::ShaderStageFlagBits::eCompute);

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info;
    pipeline_layout_create_info
        .setSetLayoutCount(1)
        .setPSetLayouts(&instance->desc_layout_)
        .setPushConstantRangeCount(1)
        .setPPushConstantRanges(&push_constant_range);
    instance->pipeline_layout_ = dev.createPipelineLayout(pipeline_layout_create_info);
    instance->isect_module_ = LoadShaderModule(dev, "../../shaders/isect.comp.spv");

    vk::PipelineShaderStageCreateInfo shader_stage_create_info;
    shader_stage_create_info
        .setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(instance->isect_module_)
        .setPName("main");

    vk::ComputePipelineCreateInfo pipeline_create_info;
    pipeline_create_info
        .setLayout(instance->pipeline_layout_)
        .setStage(shader_stage_create_info);

    instance->intersect_pipeline_ = 
        instance->device_.createComputePipeline(
            instance->pipeline_cache_,
            pipeline_create_info);
}

static void ShutdownVulkan(Instance* instance) {
    auto& dev = instance->device_;
    dev.destroyShaderModule(instance->isect_module_);
    dev.destroyPipeline(instance->intersect_pipeline_);
    dev.destroyPipelineCache(instance->pipeline_cache_);
    dev.destroyDescriptorPool(instance->desc_pool_);
    dev.destroyDescriptorSetLayout(instance->desc_layout_);
    dev.destroyPipelineLayout(instance->pipeline_layout_);
}


rr_status rrInitInstance(VkDevice device, VkCommandPool command_pool, rr_instance* out_instance) {

    if (!device || !command_pool) {
        *out_instance = nullptr;
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = new Instance {};

    InitVulkan(instance, device, command_pool);


    *out_instance = reinterpret_cast<rr_instance>(instance);
    return RR_SUCCESS;
}

rr_status rrIntersect(rr_instance inst, VkBuffer ray_buffer, VkBuffer hit_buffer, unsigned int num_rays, VkCommandBuffer* out_command_buffer) {
    // Update descriptor sets
    auto instance = reinterpret_cast<Instance*>(inst);
    auto& dev = instance->device_;

    vk::DescriptorBufferInfo desc_buffer_info;
    desc_buffer_info.setBuffer(ray_buffer);
    desc_buffer_info.setOffset(0);
    desc_buffer_info.setRange(vk::DeviceSize{ num_rays * sizeof(Ray) });

    vk::WriteDescriptorSet desc_writes;
    desc_writes.setDescriptorCount(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setDstSet(instance->desc_sets_[0])
        .setDstBinding(0)
        .setDescriptorCount(1)
        .setPBufferInfo(&desc_buffer_info);

    dev.updateDescriptorSets(desc_writes, nullptr);

    // Allocate command buffer
    vk::CommandBufferAllocateInfo cmd_info;
    cmd_info.setCommandBufferCount(1)
        .setCommandPool(instance->command_pool_)
        .setLevel(vk::CommandBufferLevel::ePrimary);

    auto cmd_buffers = dev.allocateCommandBuffers(cmd_info);

    vk::CommandBufferBeginInfo cmd_begin_info;
    cmd_begin_info.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmd_buffers[0].begin(cmd_begin_info);
    cmd_buffers[0].bindPipeline(
        vk::PipelineBindPoint::eCompute,
        instance->intersect_pipeline_);
    cmd_buffers[0].bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        instance->pipeline_layout_,
        0,
        instance->desc_sets_,
        nullptr);

    vk::BufferMemoryBarrier memory_barrier;
    memory_barrier.setBuffer(ray_buffer)
        .setOffset(0)
        .setSize(num_rays * sizeof(Ray))
        .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

    cmd_buffers[0].pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlags{},
        nullptr,
        memory_barrier,
        nullptr);

    auto N = static_cast<std::uint32_t>(num_rays);
    cmd_buffers[0].pushConstants(
        instance->pipeline_layout_, 
        vk::ShaderStageFlagBits::eCompute, 
        0u,
        sizeof(std::uint32_t),
        &N);

    cmd_buffers[0].dispatch(2, 1, 1);
    cmd_buffers[0].end();

    *out_command_buffer = cmd_buffers[0];

    return RR_SUCCESS;
}

rr_status rrOccluded(rr_instance instance, VkBuffer ray_buffer, VkBuffer hit_buffer, unsigned int num_rays, VkCommandBuffer* out_command_buffer) {
    return RR_ERROR_NOT_IMPLEMENTED;
}

rr_status rrShutdownInstance(rr_instance instance) {
    if (instance) {
        ShutdownVulkan(reinterpret_cast<Instance*>(instance));
        delete instance;
        return RR_SUCCESS;
    } else {
        return RR_ERROR_INVALID_VALUE;
    }
}

rr_status rrCreateTriangleMesh(
    rr_instance* inst,
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
    auto mesh = new RadeonRays::Mesh(
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

rr_status rrAttachShape(rr_instance* inst, rr_shape s) {
    if (!inst || !s) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);
    auto shape = reinterpret_cast<Shape*>(s);
    instance->world_.AttachShape(shape);

    return RR_SUCCESS;
}

RR_API rr_status rrDetachShape(rr_instance* inst, rr_shape s) {
    if (!inst || !s) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);
    auto shape = reinterpret_cast<Shape*>(s);
    instance->world_.DetachShape(shape);

    return RR_SUCCESS;
}

RR_API rr_status rrDetachAllShapes(rr_instance* inst) {
    if (!inst) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);
    instance->world_.DetachAll();

    return RR_SUCCESS;
}

RR_API rr_status rrCommit(rr_instance* inst) {
    if (!inst) {
        return RR_ERROR_INVALID_VALUE;
    }

    return RR_SUCCESS;
}

RR_API rr_status rrDeleteShape(rr_instance* inst, rr_shape s) {
    if (!inst || !s) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);
    auto shape = reinterpret_cast<Shape*>(s);

    delete shape;

    return RR_SUCCESS;
}