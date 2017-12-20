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

#include <memory>
#include <vector>
#include <cstdint>

namespace RadeonRays
{
    class Shape;

    class World
    {
        using ShapeList = std::vector<Shape const*>;
    public:
        using StateChange = std::uint32_t;
        using const_iterator = ShapeList::const_iterator;

        World() = default;
        virtual ~World() = default;

        void AttachShape(Shape const* shape);
        void DetachShape(Shape const* shape);
        void DetachAll();
        void OnCommit();

        StateChange state_change() const;
        bool has_changed() const { return has_changed_; }

        auto cbegin() const { return shapes_.cbegin(); }
        auto cend() const { return shapes_.cend(); }

    private:
        void set_changed(bool changed) { has_changed_ = changed; }

    private:
        std::vector<Shape const*> shapes_;
        bool has_changed_ = true;
    };
}