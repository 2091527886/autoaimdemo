#pragma once

#include "api.hpp"

namespace trtyolo {

struct TRTYOLOAPI Image {
    void*  ptr;
    int    width  = 0;
    int    height = 0;
    size_t pitch  = 0;

    Image(void* data, int width, int height);
    Image(void* data, int width, int height, size_t pitch);

    friend std::ostream& operator<<(std::ostream& os, const Image& img) {
        os << "Image(width=" << img.width
           << ", height=" << img.height
           << ", pitch=" << img.pitch
           << ", ptr=" << img.ptr << ")";
        return os;
    }
};

struct TRTYOLOAPI Mask {
    std::vector<uint8_t> data;
    int                  width  = 0;
    int                  height = 0;

    Mask(int width, int height);

    friend std::ostream& operator<<(std::ostream& os, const Mask& mask) {
        os << "Mask(width=" << mask.width << ", height=" << mask.height << ", data size=" << mask.data.size() << ")";
        return os;
    }
};

struct TRTYOLOAPI KeyPoint {
    float                x;
    float                y;
    std::optional<float> conf;

    KeyPoint(float x, float y, std::optional<float> conf = std::nullopt)
        : x(x), y(y), conf(conf) {}

    friend std::ostream& operator<<(std::ostream& os, const KeyPoint& kp) {
        os << "KeyPoint(x=" << kp.x << ", y=" << kp.y;
        if (kp.conf) os << ", conf=" << *kp.conf;
        os << ")";
        return os;
    }
};

struct TRTYOLOAPI Box {
    float left;
    float top;
    float right;
    float bottom;

    Box(float left, float top, float right, float bottom)
        : left(left), top(top), right(right), bottom(bottom) {}

    friend std::ostream& operator<<(std::ostream& os, const Box& box) {
        os << "Box(left=" << box.left << ", top=" << box.top << ", right=" << box.right << ", bottom=" << box.bottom << ")";
        return os;
    }
};

struct TRTYOLOAPI RotatedBox : public Box {
    float theta;

    RotatedBox(float left, float top, float right, float bottom, float theta)
        : Box(left, top, right, bottom), theta(theta) {}

    friend std::ostream& operator<<(std::ostream& os, const RotatedBox& rbox) {
        os << "RotatedBox(left=" << rbox.left << ", top=" << rbox.top << ", right=" << rbox.right << ", bottom=" << rbox.bottom << ", theta=" << rbox.theta << ")";
        return os;
    }
};

} // namespace trtyolo
