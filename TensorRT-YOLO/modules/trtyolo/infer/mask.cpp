#include "infer/trtyolo.hpp"
#include <stdexcept>
#include "utils/common.hpp"

namespace trtyolo {

Mask::Mask(int width, int height) : width(width), height(height) {
    if (width < 0 || height < 0) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Mask: width and height must be positive"));
    }
    data.resize(width * height);
}

} // namespace trtyolo
