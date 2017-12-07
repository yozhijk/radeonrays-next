#include <radeonrays.h>
#include <vulkan/vulkan.hpp>

#include <vector>

#include "rr_instance.h"

#include "mesh.h"
#include "utils.h"
#include "vk_memory_allocator.h"
#include "intersector_lds.h"


using namespace RadeonRays;

static std::uint32_t constexpr kMaxDescriptors = 4u;
static std::uint32_t constexpr kMaxDescriptorSets = 2u;

static void InitVulkan(
    Instance* instance,
    VkDevice device,
    VkPhysicalDevice physical_device,
    VkCommandPool command_pool) {

    instance->device = device;
    instance->cmd_pool = command_pool;

    auto& dev = instance->device;

    // Create pipeline cache
    vk::PipelineCacheCreateInfo cache_create_info;
    instance->pipeline_cache = dev.createPipelineCache(cache_create_info);

    // Create descriptor pool
    vk::DescriptorPoolSize pool_sizes;
    pool_sizes
        .setType(vk::DescriptorType::eStorageBuffer)
        .setDescriptorCount(kMaxDescriptors);
    vk::DescriptorPoolCreateInfo descpool_create_info;
    descpool_create_info
        .setMaxSets(kMaxDescriptorSets)
        .setPoolSizeCount(1)
        .setPPoolSizes(&pool_sizes);
    instance->desc_pool =
        dev.createDescriptorPool(descpool_create_info);

    instance->alloc.reset(new VkMemoryAlloc(device, physical_device));
    instance->intersector.reset(
        new IntersectorLDS(
            device,
            command_pool,
            instance->desc_pool,
            instance->pipeline_cache,
            *instance->alloc
        ));
}


static void ShutdownVulkan(Instance* instance) {
    auto& dev = instance->device;
    dev.destroyPipelineCache(instance->pipeline_cache);
    dev.destroyDescriptorPool(instance->desc_pool);
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

rr_status rrTraceRays(
    rr_instance inst,
    rr_query_type query_type,
    uint32_t num_rays,
    VkCommandBuffer* out_command_buffer) {
    auto instance = reinterpret_cast<Instance*>(inst);
    auto cmdbuffer = instance->intersector->TraceRays(num_rays);
    *out_command_buffer = cmdbuffer;
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
    instance->world.AttachShape(shape);
    return RR_SUCCESS;
}

rr_status rrDetachShape(rr_instance inst, rr_shape s) {
    if (!inst || !s) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);
    auto shape = reinterpret_cast<Shape*>(s);
    instance->world.DetachShape(shape);
    return RR_SUCCESS;
}

rr_status rrDetachAllShapes(rr_instance inst) {
    if (!inst) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);
    instance->world.DetachAll();
    return RR_SUCCESS;
}

rr_status rrCommit(rr_instance inst, VkCommandBuffer* out_command_buffer) {
    if (!inst | !out_command_buffer) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);
    auto cmdbuffer = instance->intersector->Commit(instance->world);
    *out_command_buffer = cmdbuffer;
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

rr_status rrBindBuffers(
    rr_instance inst,
    VkBuffer ray_buffer,
    VkBuffer hit_buffer,
    uint32_t num_rays
) {
    if (!inst) {
        return RR_ERROR_INVALID_VALUE;
    }

    auto instance = reinterpret_cast<Instance*>(inst);
    instance->intersector->BindBuffers(ray_buffer, hit_buffer, num_rays);
    return RR_SUCCESS;
}

rr_status rrShapeSetTransform(
    rr_instance instance,
    rr_shape shape,
    float const* transform
) {
    return RR_ERROR_NOT_IMPLEMENTED;
}