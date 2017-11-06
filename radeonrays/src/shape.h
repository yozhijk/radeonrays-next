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

#include "math/float3.h"
#include "math/matrix.h"

namespace RadeonRays {
    class Shape {
    public:
        using Id = std::int32_t;
        using Mask = std::uint32_t;
        using StateChange = std::uint32_t;

        static constexpr Id kInvalidId = -1;

        enum StateChangeFlags {
            kStateChangeNone = 0x0,
            kStateChangeTransform = 0x1,
            kStateChangeMotion = 0x2,
            kStateChangeId = 0x4,
            kStateChangeMask = 0x5
        };

        Shape() = default;
        virtual ~Shape() = 0;
        virtual bool is_instance() const { return false; }

        void SetTransform(matrix const& m) { 
            transform_ = m; 
            state_change_ |= kStateChangeTransform;
        }
        auto GetTransform() const { return transform_; }
        void SetId(Id id) { 
            id_ = id; 
            state_change_ |= kStateChangeId;
        }
        auto GetId() const { return id_; }
        void SetMask(Mask mask) { 
            mask_ = mask;
            state_change_ |= kStateChangeMask;
        }
        auto GetMask() const { return mask_; }
        void OnCommit() const { state_change_ = kStateChangeNone; }

        StateChange state_change() const { return state_change_; }

    private:
        matrix transform_;
        Mask mask_ = 0xFFFFFFFFu;
        Id id_ = kInvalidId;
        mutable StateChange state_change_ = kStateChangeNone;
    };

    inline Shape::~Shape() {}
}
