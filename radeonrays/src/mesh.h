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
#include <array>
#include <cassert>

#include "shape.h"
#include "math/bbox.h"
#include "math/float3.h"
#include "math/float2.h"
#include "math/mathutils.h"

namespace RadeonRays {
    class Mesh : public Shape {
    public:
        enum class CoordinateSpace {
            kLocal,
            kWorld
        };

        struct Face {
            int idx[3];
        };

        Mesh(float const* vertices,
            std::uint32_t num_vertices,
            std::uint32_t vertex_stride,
            std::uint32_t const* indices,
            std::uint32_t index_stride,
            std::uint32_t num_faces)
            : vertices_(reinterpret_cast<float3 const*>(vertices))
            , num_vertices_(num_vertices)
            , vertex_stride_(vertex_stride == 0
                ? (sizeof(float) * 4) : vertex_stride)
            , indices_(indices)
            , num_indices_(num_faces * 3)
            , index_stride_(index_stride == 0
                ? sizeof(std::uint32_t) : index_stride) {
        }

        auto num_faces() const { return num_indices_ / 3; }
        auto num_vertices() const { return num_vertices_; }
        auto vertex_stride() const { return vertex_stride_; }
        auto index_stride() const { return index_stride_; }

        auto GetFaceBounds(
            int face_index,
            CoordinateSpace space = CoordinateSpace::kWorld) const {
            auto vertices = GetTransformedVertices(
                face_index, 
                space == CoordinateSpace::kLocal ?
                matrix() : GetTransform());
            bbox bounds{ vertices[0], vertices[1] };
            bounds.grow(vertices[2]);
            return bounds;
        }

        auto GetVertexData() const { return vertices_; }
        auto GetVertexData(std::size_t vertex_index) const {
            return *reinterpret_cast<float3 const*>(
                reinterpret_cast<char const*>(vertices_) +
                vertex_index * vertex_stride_);
        }
        auto GetVertexDataPtr(std::size_t vertex_index) const {
            return reinterpret_cast<float3 const*>(
                reinterpret_cast<char const*>(vertices_) +
                vertex_index * vertex_stride_);
        }
        auto GetFaceVertexData(std::size_t face_index) const {
            return GetTransformedVertices(face_index, matrix());
        }

        auto GetIndexData() const { return indices_; }
        auto GetIndexData(std::size_t face_index) const {
            Face face;
            face.idx[0] = *reinterpret_cast<std::uint32_t const*>(
                reinterpret_cast<char const*>(indices_) +
                3 * face_index * index_stride_);
            face.idx[1] = *reinterpret_cast<std::uint32_t const*>(
                reinterpret_cast<char const*>(indices_) +
                (3 * face_index + 1) * index_stride_);
            face.idx[2] = *reinterpret_cast<std::uint32_t const*>(
                reinterpret_cast<char const*>(indices_) +
                (3 * face_index + 2) * index_stride_);
            return face;
        }

        auto GetBounds(CoordinateSpace space = CoordinateSpace::kWorld) const {
            bbox bounds;
            for (std::uint32_t i = 0; i < num_faces(); ++i) {
                bounds.grow(GetFaceBounds(i, space));
            }
            return bounds;
        }

        Mesh(Mesh const&) = default;
        Mesh& operator =(Mesh const&) = default;

    private:
        std::array<float3, 3> GetTransformedVertices(
            std::size_t face_index,
            matrix const& transform) const {
            auto face = GetIndexData(face_index);
            std::array<float3, 3> vertices;
            vertices[0] = transform_point(GetVertexData(face.idx[0]),transform);
            vertices[1] = transform_point(GetVertexData(face.idx[1]), transform);
            vertices[2] = transform_point(GetVertexData(face.idx[2]), transform);
            return vertices;
        }

        std::array<float3, 3> GetFaceVertices(
            std::size_t face_index) const {
             auto face = GetIndexData(face_index);
             std::array<float3, 3> vertices;
             vertices[0] = GetVertexData(face.idx[0]);
             vertices[1] = GetVertexData(face.idx[1]);
             vertices[2] = GetVertexData(face.idx[2]);
             return vertices;
         }

        float3 const* vertices_;
        std::size_t num_vertices_;
        std::uint32_t vertex_stride_;

        std::uint32_t const* indices_;
        std::size_t num_indices_;
        std::uint32_t index_stride_;
    };
}
