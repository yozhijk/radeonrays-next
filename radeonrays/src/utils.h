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

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>

#include <xmmintrin.h>
#include <smmintrin.h>

namespace RadeonRays {

    struct aligned_allocator {
#ifdef WIN32
        static
        void* allocate(std::size_t size, std::size_t alignement) {
            return _aligned_malloc(size, alignement);
        }

        static
        void deallocate(void* ptr) {
            return _aligned_free(ptr);
        }
#else
        static void* allocate(std::size_t size, std::size_t) {
            return malloc(size);
        }

        static void deallocate(void* ptr) {
            return free(ptr);
        }
#endif
    };

#ifdef __GNUC__
#define clz(x) __builtin_clz(x)
#define ctz(x) __builtin_ctz(x)
#else
    inline std::uint32_t popcnt(std::uint32_t x) {
        x -= ((x >> 1) & 0x55555555);
        x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
        x = (((x >> 4) + x) & 0x0f0f0f0f);
        x += (x >> 8);
        x += (x >> 16);
        return x & 0x0000003f;
    }

    inline std::uint32_t clz(std::uint32_t x) {
        x |= (x >> 1);
        x |= (x >> 2);
        x |= (x >> 4);
        x |= (x >> 8);
        x |= (x >> 16);
        return 32 - popcnt(x);
    }

    inline std::uint32_t ctz(std::uint32_t x) {
        return popcnt((std::uint32_t)(x & -(int)x) - 1);
    }
#endif

    inline
    void LoadFileContents(
        std::string const& name,
        std::vector<char>& contents,
        bool binary = false) {
        std::ifstream in(name, std::ios::in | (std::ios_base::openmode)(binary ? std::ios::binary : 0));
        if (in) {
            contents.clear();
            std::streamoff beg = in.tellg();
            in.seekg(0, std::ios::end);
            std::streamoff fileSize = in.tellg() - beg;
            in.seekg(0, std::ios::beg);
            contents.resize(static_cast<unsigned>(fileSize));
            in.read(&contents[0], fileSize);
        } else {
            throw std::runtime_error("Cannot read the contents of a file");
        }
    }
}
