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

#include <vulkan/vulkan.h>
#include <stdint.h>

#define RR_STATIC_LIBRARY 1
#if !RR_STATIC_LIBRARY
#ifdef WIN32
#ifdef EXPORT_API
#define RR_API __declspec(dllexport)
#else
#define RR_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#ifdef EXPORT_API
#define RR_API __attribute__((visibility ("default")))
#else
#define RR_API
#endif
#endif
#else
#define RR_API
#endif

// Error codes
#define RR_SUCCESS 0
#define RR_ERROR_INVALID_VALUE -1
#define RR_ERROR_NOT_IMPLEMENTED -2
#define RR_ERROR_OUT_OF_SYSTEM_MEMORY -3
#define RR_ERROR_OUT_OF_VIDEO_MEMORY -4

// Invalid index marker
#define RR_INVALID_ID 0xffffffffu

// Data types
typedef int rr_status;
typedef int rr_init_flags;
typedef struct{}* rr_instance;
typedef struct {}* rr_shape;

// Ray payload structure
struct Ray {
    // Ray direction
    float direction[3];
    // Not used for now
    float time;
    // Ray origin
    float origin[3];
    // Ray maximum distance
    float max_t;
};

// Hit payload structure
struct Hit {
    // ID of shape (RR_INVALID_ID in case of a miss)
    uint32_t shape_id;
    // ID of a primitive (undefined in case of a miss)
    uint32_t prim_id;
    // Barycentric coordinates of a hit
    // (undefined in case of a miss)
    float uv[2];
};

// API functions
#ifdef __cplusplus
extern "C" {
#endif
    // Initialize and instance of library.
    RR_API rr_status rrInitInstance(
        // GPU to run ray queries on
        VkDevice device,
        VkPhysicalDevice physical_device,
        // Command pool to allocate command buffers
        VkCommandPool command_pool,
        // Resulting instance
        rr_instance* out_instance);

    // Attach shape to the current scene maintained by an API.
    RR_API rr_status rrAttachShape(rr_instance instance, rr_shape shape);

    // Detach shape from the current scene maintained by an API.
    RR_API rr_status rrDetachShape(rr_instance instance, rr_shape shape);

    // Detach all shapes from the current scene maintained by an API.
    RR_API rr_status rrDetachAllShapes(rr_instance instance);

    // Commit changes for the current scene maintained by an API.
    RR_API rr_status rrCommit(rr_instance instance, VkCommandBuffer* out_command_buffer);

    // Delete shape object.
    RR_API rr_status rrDeleteShape(rr_instance instance, rr_shape shape);

    // Bind buffers for the query
    RR_API rr_status rrBindBuffers(
        rr_instance instance,
        VkBuffer ray_buffer,
        VkBuffer hit_buffer,
        uint32_t num_rays
    );

    RR_API rr_status rrIntersect(
        rr_instance instance,
        uint32_t num_rays,
        VkCommandBuffer* out_command_buffer
    );

    RR_API rr_status rrCreateTriangleMesh(
        rr_instance instance,
        float const* vertices,
        uint32_t num_vertices,
        uint32_t vertex_stride,
        uint32_t const* indices,
        uint32_t index_stride,
        uint32_t num_faces,
        uint32_t id,
        rr_shape* out_shape
    );

    RR_API rr_status rrOccluded(rr_instance instance,
        VkBuffer ray_buffer,
        VkBuffer hit_buffer,
        uint32_t num_rays,
        VkCommandBuffer* out_command_buffer
    );

    RR_API rr_status rrShutdownInstance(rr_instance instance);
#ifdef __cplusplus
}
#endif
