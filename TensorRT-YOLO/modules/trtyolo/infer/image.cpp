#include "infer/trtyolo.hpp"
#include "infer/pch_infer.hpp"
#include "utils/common.hpp"

namespace trtyolo {

Image::Image(void* data, int width, int height)
    : ptr(data), width(width), height(height), pitch(width * sizeof(uint8_t) * 3) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: width and height must be positive"));
    }
}

Image::Image(void* data, int width, int height, size_t pitch)
    : ptr(data), width(width), height(height), pitch(pitch) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: width and height must be positive"));
    }
    if (pitch < static_cast<size_t>(width * sizeof(uint8_t) * 3)) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: pitch must >= width * 3"));
    }
}

} // namespace trtyolo
