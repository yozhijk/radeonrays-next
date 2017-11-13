#pragma once

#include <vulkan/vulkan.h>

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

#define RR_SUCCESS 0
#define RR_ERROR_INVALID_VALUE -1
#define RR_ERROR_NOT_IMPLEMENTED -2

typedef int rr_status;
typedef int rr_init_flags;
typedef struct{}* rr_instance;

#ifdef __cplusplus
extern "C" {
#endif
    RR_API rr_status rrInitInstance(VkDevice device, VkCommandPool command_pool, rr_instance* out_instance);
    RR_API rr_status rrIntersect(rr_instance instance, VkBuffer ray_buffer, VkBuffer hit_buffer, unsigned int num_rays, VkCommandBuffer* out_command_buffer);
    RR_API rr_status rrOccluded(rr_instance instance, VkBuffer ray_buffer, VkBuffer hit_buffer, unsigned int num_rays, VkCommandBuffer* out_command_buffer);
    RR_API rr_status rrShutdownInstance(rr_instance instance);
#ifdef __cplusplus
}
#endif
