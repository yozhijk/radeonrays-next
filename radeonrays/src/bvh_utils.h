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

#include <xmmintrin.h>
#include <smmintrin.h>

namespace RadeonRays
{

    inline __m128 aabb_surface_area(__m128 pmin, __m128 pmax)
    {
        auto ext = _mm_sub_ps(pmax, pmin);
        auto xxy = _mm_shuffle_ps(ext, ext, _MM_SHUFFLE(3, 1, 0, 0));
        auto yzz = _mm_shuffle_ps(ext, ext, _MM_SHUFFLE(3, 2, 2, 1));
        return _mm_mul_ps(_mm_dp_ps(xxy, yzz, 0xff),
            _mm_set_ps(2.f, 2.f, 2.f, 2.f));
    }

    inline __m128 aabb_extents(__m128 pmin, __m128 pmax)
    {
        return _mm_sub_ps(pmax, pmin);
    }

    inline std::uint32_t aabb_max_extent_axis(__m128 pmin, __m128 pmax)
    {
        auto xyz = _mm_sub_ps(pmax, pmin);
        auto yzx = _mm_shuffle_ps(xyz, xyz, _MM_SHUFFLE(3, 0, 2, 1));
        auto m0 = _mm_max_ps(xyz, yzx);
        auto m1 = _mm_shuffle_ps(m0, m0, _MM_SHUFFLE(3, 0, 2, 1));
        auto m2 = _mm_max_ps(m0, m1);
        auto cmp = _mm_cmpeq_ps(xyz, m2);
        return ctz(_mm_movemask_ps(cmp));
    }

    inline float mm_select(__m128 v, std::uint32_t index)
    {
        _MM_ALIGN16 float temp[4];
        _mm_store_ps(temp, v);
        return temp[index];
    }

    inline bool aabb_contains_point(float const* aabb_min,
        float const* aabb_max,
        float const* point)
    {
        return point[0] >= aabb_min[0] &&
            point[0] <= aabb_max[0] &&
            point[1] >= aabb_min[1] &&
            point[1] <= aabb_max[1] &&
            point[2] >= aabb_min[2] &&
            point[2] <= aabb_max[2];
    }
}
