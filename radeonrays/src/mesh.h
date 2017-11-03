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

#include <vector>
#include <memory>
#include <cassert>

#include "shape.h"
#include "math/bbox.h"
#include "math/float3.h"
#include "math/float2.h"
#include "math/mathutils.h"

namespace RadeonRays {
    class Mesh : public Shape {
    public:
        Mesh(float const* vertices,
            std::uint32_t num_vertices,
            std::uint32_t vertex_stride,
            std::uint32_t const* indices,
            std::uint32_t index_stride,
            std::uint32_t num_faces)
            : vertices_(reinterpret_cast<float3 const*>(vertices))
            , num_vertices_(num_vertices)
            , vertex_stride_(vertex_stride == 0 ? (sizeof(float) * 4) : vertex_stride)
            , indices_(indices)
            , num_indices_(num_faces * 3)
            , index_stride_(index_stride == 0 ? sizeof(std::uint32_t) : index_stride) {
        }

        auto num_faces() const { return num_indices_ / 3; }
        auto num_vertices() const { return num_vertices_; }
        auto GetFaceBounds(int face_index, bool object_space) const {
            float3 vertices[3];
            GetTransformedFace(face_index, object_space ? matrix() : GetTransform(), vertices);
            bbox bounds{ vertices[0], vertices[1] };
            bounds.grow(vertices[2]);
            return bounds;
        }

        auto GetVertexData() const { return vertices_; }
        auto GetVertexData(std::size_t face_index) const {
            return reinterpret_cast<float3 const*>(
                reinterpret_cast<char const*>(vertices_) + face_index * vertex_stride_);
        }

        auto GetIndexData() const { return indices_; }
        auto GetIndexData(std::size_t face_index) const {
            return reinterpret_cast<std::uint32_t const*>(
                reinterpret_cast<char const*>(indices_) + 3 * face_index * index_stride_);
        }

        Mesh(Mesh const&) = delete;
        Mesh& operator =(Mesh const&) = delete;

    private:
         std::uint32_t GetTransformedFace(int face_index, matrix const& transform, float3* out_vertices) const {
            auto face_indices = GetIndexData(face_index);
            out_vertices[0] = transform_point(*GetVertexData(face_indices[0]), transform);
            out_vertices[1] = transform_point(*GetVertexData(face_indices[1]), transform);
            out_vertices[2] = transform_point(*GetVertexData(face_indices[2]), transform);
            return 3u;
        }

        float3 const* vertices_;
        std::uint32_t vertex_stride_;
        std::size_t num_vertices_;

        std::uint32_t const* indices_;
        std::uint32_t index_stride_;
        std::size_t num_indices_;
    };
}
