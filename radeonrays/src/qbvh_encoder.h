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

#include <radeonrays.h>
#include <cstdint>
#include <bvh_encoder.h>
#include <vector>

namespace RadeonRays {

    union FP32 {
        std::uint32_t u;
        float f;
        struct {
            std::uint32_t Mantissa : 23;
            std::uint32_t Exponent : 8;
            std::uint32_t Sign : 1;
        };
    };

    union FP16 {
        unsigned short u;
        struct {
            std::uint32_t Mantissa : 10;
            std::uint32_t Exponent : 5;
            std::uint32_t Sign : 1;
        };
    };

    inline std::uint16_t float_to_half(float value, bool min) {
        FP32 f; f.f = { value };
        FP16 o = { 0 };
        if (f.Exponent == 0)
            o.Exponent = 0;
        else if (f.Exponent == 255) {
            o.Exponent = 31;
            o.Mantissa = f.Mantissa ? 0x200 : 0;
        } else {
            int newexp = f.Exponent - 127 + 15;
            if (newexp >= 31)
                o.Exponent = 31;
            else if (newexp <= 0) {
                if ((14 - newexp) <= 24) {
                    std::uint32_t mant = f.Mantissa | 0x800000;
                    o.Mantissa = mant >> (14 - newexp);
                    if (((mant >> (13 - newexp))) & 1 && (!f.Sign && !min || f.Sign && min ))
                        o.u++;
                }
            } else {
                o.Exponent = newexp;
                o.Mantissa = f.Mantissa >> 13;
                if (!f.Sign && !min || f.Sign && min)
                    o.u++;
            }
        }

        o.Sign = f.Sign;
        return o.u;
    }

    inline std::uint16_t float_to_half_min(float value)
    {
        return float_to_half(value, true);
    }

    inline std::uint16_t float_to_half_max(float value)
    {
        return float_to_half(value, false);
    }

    inline float half_to_float(std::uint16_t value)
    {
        FP16 h = { value };
        static const FP32 magic = { 113 << 23 };
        static const std::uint32_t shifted_exp = 0x7c00 << 13; // exponent mask after shift
        FP32 o;

        o.u = (h.u & 0x7fff) << 13;     // exponent/mantissa bits
        std::uint32_t exp = shifted_exp & o.u;   // just the exponent
        o.u += (127 - 15) << 23;        // exponent adjust

                                        // handle exponent special cases
        if (exp == shifted_exp) // Inf/NaN?
            o.u += (128 - 16) << 23;    // extra exp adjust
        else if (exp == 0) // Zero/Denormal?
        {
            o.u += 1 << 23;             // extra exp adjust
            o.f -= magic.f;             // renormalize
        }

        o.u |= (h.u & 0x8000) << 16;    // sign bit
        return o.f;
    }



    inline void copy3(float const* src, float* dst) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
    }

    inline void copy3(float const* src, std::uint32_t* dst) {
        *(reinterpret_cast<float*>(dst)) = src[0];
        *(reinterpret_cast<float*>(dst) + 1) = src[1];
        *(reinterpret_cast<float*>(dst) + 2) = src[2];
    }

    inline void copy3pack_lo_min(float const* src, std::uint32_t* dst) {
        *dst = (*dst & 0xffff0000u) + float_to_half_min(src[0]);
        *(dst + 1) = (*(dst + 1) & 0xffff0000u) + float_to_half_min(src[1]);
        *(dst + 2) = (*(dst + 2) & 0xffff0000u) + float_to_half_min(src[2]);
    }

    inline void copy3pack_hi_min(float const* src, std::uint32_t* dst) {
        *dst = (*dst & 0xffffu) + (float_to_half_min(src[0]) << 16);
        *(dst + 1) = (*(dst + 1) & 0xffffu) + (float_to_half_min(src[1]) << 16);
        *(dst + 2) = (*(dst + 2) & 0xffffu) + (float_to_half_min(src[2]) << 16);
    }
    inline void copy3pack_lo_max(float const* src, std::uint32_t* dst) {
        *dst = (*dst & 0xffff0000u) + float_to_half_max(src[0]);
        *(dst + 1) = (*(dst + 1) & 0xffff0000u) + float_to_half_max(src[1]);
        *(dst + 2) = (*(dst + 2) & 0xffff0000u) + float_to_half_max(src[2]);
    }

    inline void copy3pack_hi_max(float const* src, std::uint32_t* dst) {
        *dst = (*dst & 0xffffu) + (float_to_half_max(src[0]) << 16);
        *(dst + 1) = (*(dst + 1) & 0xffffu) + (float_to_half_max(src[1]) << 16);
        *(dst + 2) = (*(dst + 2) & 0xffffu) + (float_to_half_max(src[2]) << 16);
    }


    inline void copy3unpack_lo(std::uint32_t const* src, float* dst) {
        *dst = half_to_float(src[0] & 0xffffu);
        *(dst + 1) = half_to_float(src[1] & 0xffffu);
        *(dst + 2) = half_to_float(src[2] & 0xffffu);
    }

    inline void copy3unpack_hi(std::uint32_t const* src, float* dst) {
        *dst = half_to_float(src[0] >> 16);
        *(dst + 1) = half_to_float(src[1] >> 16);
        *(dst + 2) = half_to_float(src[2] >> 16);
    }

    // Encoded node format.
    struct QBVHNode {
        std::uint32_t aabb01_min_or_v0[3];
        std::uint32_t addr0 = RR_INVALID_ID;

        std::uint32_t aabb01_max_or_v1[3];
        std::uint32_t addr1_or_mesh_id = RR_INVALID_ID;

        std::uint32_t aabb23_min_or_v2[3];
        std::uint32_t addr2_or_prim_id = RR_INVALID_ID;

        std::uint32_t aabb23_max[3];
        std::uint32_t addr3 = RR_INVALID_ID;
    };

    struct QBvh {
        std::vector<QBVHNode> nodes_;

        static QBvh Create(Bvh<BVHNode, BVHNodeTraits> const& bvh) {

            QBvh qbvh;
            qbvh.nodes_.resize(1);

            struct Elem {
                std::uint32_t bvh_node_index;
                std::uint32_t qbvh_node_index;
            };

            std::stack<Elem> stack;
            stack.push({ 0u, 0u });

            while (!stack.empty()) {
                auto node = bvh.GetNode(stack.top().bvh_node_index);
                auto qbvh_node_index = stack.top().qbvh_node_index;
                stack.pop();

                if (!BVHNodeTraits::IsInternal(*node)) {
                    qbvh.nodes_[qbvh_node_index].addr0 = RR_INVALID_ID;
                    qbvh.nodes_[qbvh_node_index].addr3 = RR_INVALID_ID;

                    copy3(node->aabb_left_min_or_v0, qbvh.nodes_[qbvh_node_index].aabb01_min_or_v0);
                    copy3(node->aabb_left_max_or_v1, qbvh.nodes_[qbvh_node_index].aabb01_max_or_v1);
                    copy3(node->aabb_right_min_or_v2, qbvh.nodes_[qbvh_node_index].aabb23_min_or_v2);

                    qbvh.nodes_[qbvh_node_index].addr1_or_mesh_id = node->mesh_id;
                    qbvh.nodes_[qbvh_node_index].addr2_or_prim_id = node->prim_id;
                    continue;
                }

                auto c0idx = BVHNodeTraits::GetChildIndex(*node, 0);
                auto c1idx = BVHNodeTraits::GetChildIndex(*node, 1);

                auto c0 = bvh.GetNode(c0idx);
                auto c1 = bvh.GetNode(c1idx);

                if (BVHNodeTraits::IsInternal(*c0)) {
                    copy3pack_lo_min(c0->aabb_left_min_or_v0, qbvh.nodes_[qbvh_node_index].aabb01_min_or_v0);
                    copy3pack_lo_max(c0->aabb_left_max_or_v1, qbvh.nodes_[qbvh_node_index].aabb01_max_or_v1);
                    copy3pack_hi_min(c0->aabb_right_min_or_v2, qbvh.nodes_[qbvh_node_index].aabb01_min_or_v0);
                    copy3pack_hi_max(c0->aabb_right_max, qbvh.nodes_[qbvh_node_index].aabb01_max_or_v1);

                    auto idx = (std::uint32_t)qbvh.nodes_.size();
                    qbvh.nodes_.push_back(QBVHNode{});
                    qbvh.nodes_.push_back(QBVHNode{});

                    qbvh.nodes_[qbvh_node_index].addr0 = idx;
                    qbvh.nodes_[qbvh_node_index].addr1_or_mesh_id = idx + 1;

                    stack.push({BVHNodeTraits::GetChildIndex(*c0, 0),(std::uint32_t)idx});
                    stack.push({BVHNodeTraits::GetChildIndex(*c0, 1),(std::uint32_t)idx + 1 });
                } else {
                    copy3pack_lo_min(node->aabb_left_min_or_v0, qbvh.nodes_[qbvh_node_index].aabb01_min_or_v0);
                    copy3pack_lo_max(node->aabb_left_max_or_v1, qbvh.nodes_[qbvh_node_index].aabb01_max_or_v1);

                    auto idx = (std::uint32_t)qbvh.nodes_.size();
                    qbvh.nodes_.push_back(QBVHNode{});

                    qbvh.nodes_[qbvh_node_index].addr0 = idx;
                    qbvh.nodes_[qbvh_node_index].addr1_or_mesh_id = RR_INVALID_ID;

                    auto& child_node = qbvh.nodes_[idx];
                    child_node.addr0 = RR_INVALID_ID;
                    child_node.addr3 = RR_INVALID_ID;
                    copy3(c0->aabb_left_min_or_v0, child_node.aabb01_min_or_v0);
                    copy3(c0->aabb_left_max_or_v1, child_node.aabb01_max_or_v1);
                    copy3(c0->aabb_right_min_or_v2, child_node.aabb23_min_or_v2);
                    child_node.addr1_or_mesh_id = c0->mesh_id;
                    child_node.addr2_or_prim_id = c0->prim_id;
                }

                if (BVHNodeTraits::IsInternal(*c1)) {
                    copy3pack_lo_min(c1->aabb_left_min_or_v0, qbvh.nodes_[qbvh_node_index].aabb23_min_or_v2);
                    copy3pack_lo_max(c1->aabb_left_max_or_v1, qbvh.nodes_[qbvh_node_index].aabb23_max);
                    copy3pack_hi_min(c1->aabb_right_min_or_v2, qbvh.nodes_[qbvh_node_index].aabb23_min_or_v2);
                    copy3pack_hi_max(c1->aabb_right_max, qbvh.nodes_[qbvh_node_index].aabb23_max);

                    auto idx = (std::uint32_t)qbvh.nodes_.size();
                    qbvh.nodes_.push_back(QBVHNode{});
                    qbvh.nodes_.push_back(QBVHNode{});

                    qbvh.nodes_[qbvh_node_index].addr2_or_prim_id = idx;
                    qbvh.nodes_[qbvh_node_index].addr3 = idx + 1;

                    stack.push({ BVHNodeTraits::GetChildIndex(*c1, 0),(std::uint32_t)idx });
                    stack.push({ BVHNodeTraits::GetChildIndex(*c1, 1),(std::uint32_t)idx + 1 });
                }
                else {
                    copy3pack_lo_min(node->aabb_right_min_or_v2, qbvh.nodes_[qbvh_node_index].aabb23_min_or_v2);
                    copy3pack_lo_max(node->aabb_right_max, qbvh.nodes_[qbvh_node_index].aabb23_max);

                    auto idx = (std::uint32_t)qbvh.nodes_.size();
                    qbvh.nodes_.push_back(QBVHNode{});

                    qbvh.nodes_[qbvh_node_index].addr2_or_prim_id = idx;
                    qbvh.nodes_[qbvh_node_index].addr3 = RR_INVALID_ID;

                    auto& child_node = qbvh.nodes_[idx];
                    child_node.addr0 = RR_INVALID_ID;
                    child_node.addr3 = RR_INVALID_ID;
                    copy3(c1->aabb_left_min_or_v0, child_node.aabb01_min_or_v0);
                    copy3(c1->aabb_left_max_or_v1, child_node.aabb01_max_or_v1);
                    copy3(c1->aabb_right_min_or_v2, child_node.aabb23_min_or_v2);
                    child_node.addr1_or_mesh_id = c1->mesh_id;
                    child_node.addr2_or_prim_id = c1->prim_id;
                }
            }

            return qbvh;
        }
    };
}
