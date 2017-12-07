#include "intersector_lds.h"
#include "vk_utils.h"
#include "vk_memory_allocator.h"

namespace RadeonRays {
    std::uint32_t constexpr kNumBindings = 4u;
    std::uint32_t constexpr kWorgGroupSize = 64u;

    IntersectorLDS::IntersectorLDS(
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
        shader_ = LoadShaderModule(device_, "../../shaders/isect.comp.spv");

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
            kInitialWorkBufferSize * kGlobalStackSize,
            16u);
    }

    IntersectorLDS::~IntersectorLDS() {
        alloc_.deallocate(stack_);
        alloc_.deallocate(bvh_local_);
        alloc_.deallocate(bvh_staging_);
        device_.destroyPipelineLayout(pipeline_layout_);
        device_.destroyPipeline(pipeline_);
        device_.destroyDescriptorSetLayout(desc_layout_);
        device_.destroyShaderModule(shader_);
    }

    vk::CommandBuffer IntersectorLDS::Commit(World const& world) {
        Bvh<BVHNode, BVHNodeTraits> bvh;

        // Build BVH
        bvh.Build(world.cbegin(), world.cend());

        // Transform BVH from 1 AABB per node to 2 AABBs
        BVHNodeTraits::PropagateBounds(bvh);

        // Calculate BVH size
        auto bvh_size_in_bytes = bvh.num_nodes() * sizeof(BVHNode);

#ifdef TEST
        std::cout << "BVH size is " << bvh_size_in_bytes / 1024.f / 1024.f << "MB";
#endif
        // Check if we have enough space for BVH 
        // and realloc buffers if necessary
        CheckAndReallocBVH(bvh_size_in_bytes);

        // Map staging buffer
        auto ptr = reinterpret_cast<BVHNode*>(
            device_.mapMemory(
                bvh_staging_.memory,
                bvh_staging_.offset,
                bvh_size_in_bytes));

        // Copy BVH data
        for (auto i = 0u; i < bvh.num_nodes(); ++i) {
            ptr[i] = *bvh.GetNode(i);
        }

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
            vk::DependencyFlags {},
            nullptr,
            memory_barrier,
            nullptr
        );

        // End command buffer
        cmdbuffers[0].end();

        return cmdbuffers[0];
    }

    void IntersectorLDS::BindBuffers(vk::Buffer rays, vk::Buffer hits, std::uint32_t num_rays) {
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

    vk::CommandBuffer IntersectorLDS::TraceRays(std::uint32_t num_rays) {
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
