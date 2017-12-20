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
#include "world.h"

namespace RadeonRays
{
    // Base class for all the intersectors, capable of:
    //  * Handling the scene
    //  * Binding ray buffers
    //  * Doing ray queries
    class Intersector
    {
    public:
        Intersector() = default;
        virtual ~Intersector() = default;

        // Set ray buffer to read from and hit buffer to write to
        virtual void BindBuffers(vk::Buffer rays,
            vk::Buffer hits,
            std::uint32_t num_rays) = 0;

        // Commit scene changes
        virtual vk::CommandBuffer Commit(World const& world) = 0;
        // Trace num_rays rays
        virtual vk::CommandBuffer TraceRays(std::uint32_t num_rays) = 0;

        Intersector(Intersector const&) = delete;
        Intersector& operator = (Intersector const&) = delete;
    };
}