#include "infer/backend.hpp"
#include "letterbox.hpp"
#include "core/core.hpp"
#include <cstring>
#include <cstdint>
#include <stdexcept>

namespace trtyolo {

void TrtBackend::dynamicInfer(const std::vector<Image>& inputs) {
    auto num = inputs.size();

    // validate
    if (num < min_shape.x || num > max_shape.x) {
        throw std::invalid_argument("bad n");
    }

    // update shapes and addresses
    for (auto& tensor_info : tensor_infos) {
        tensor_info.shape.d[0] = num;
        tensor_info.update();
        manager_->setTensorAddress(tensor_info.name.c_str(), tensor_info.buffer->device());
        if (tensor_info.input) {
            manager_->setInputShape(tensor_info.name.c_str(), tensor_info.shape);
        }
    }

    if (infer_config.input_shape.has_value()) {
        if (!infer_config.cuda_mem) {
            for (int idx = 0; idx < num; ++idx) {
                uint8_t* dst = static_cast<uint8_t*>(inputs_buffer_->host()) + idx * input_image_bytes_;
                const uint8_t* src = static_cast<const uint8_t*>(inputs[idx].ptr);
                size_t row_bytes = input_row_bytes_;
                for (int r = 0; r < infer_config.input_shape->x; ++r) {
                    std::memcpy(dst + r * row_bytes, src + r * inputs[idx].pitch, row_bytes);
                }
            }
            inputs_buffer_->hostToDevice(stream);
        }

        for (int idx = 0; idx < num; ++idx) {
            cudaLetterboxHalf(
                infer_config.cuda_mem ? inputs[idx].ptr : static_cast<uint8_t*>(inputs_buffer_->device()) + idx * input_image_bytes_,
                inputs[idx].width,
                inputs[idx].height,
                infer_config.cuda_mem ? inputs[idx].pitch : input_row_bytes_,
                static_cast<uint16_t*>(tensor_infos.front().buffer->device()) + idx * infer_size_,
                max_shape.w,
                max_shape.z,
                affine_transforms.front(),
                infer_config.config,
                stream);
        }
    } else {
        int              total_size = 0;
        std::vector<int> input_sizes(num);
        for (int idx = 0; idx < num; ++idx) {
            input_sizes[idx]  = inputs[idx].height * inputs[idx].pitch;
            total_size       += input_sizes[idx];
            affine_transforms[idx].updateParams(inputs[idx].width, inputs[idx].height, max_shape.w, max_shape.z);
        }

        if (!infer_config.cuda_mem) {
            inputs_buffer_->allocate(total_size);
            uint8_t* input_host = static_cast<uint8_t*>(inputs_buffer_->host());
            for (int idx = 0; idx < num; ++idx) {
                std::memcpy(input_host, inputs[idx].ptr, input_sizes[idx]);
                input_host += input_sizes[idx];
            }
            inputs_buffer_->hostToDevice(stream);

            uint8_t* input_device = static_cast<uint8_t*>(inputs_buffer_->device());
            for (int idx = 0; idx < num; ++idx) {
                cudaLetterboxHalf(
                    input_device,
                    inputs[idx].width,
                    inputs[idx].height,
                    inputs[idx].pitch,
                    static_cast<uint16_t*>(tensor_infos.front().buffer->device()) + idx * infer_size_,
                    max_shape.w,
                    max_shape.z,
                    affine_transforms[idx],
                    infer_config.config,
                    stream);
                input_device += input_sizes[idx];
            }
        } else {
            for (int idx = 0; idx < num; ++idx) {
                cudaLetterboxHalf(
                    inputs[idx].ptr,
                    inputs[idx].width,
                    inputs[idx].height,
                    inputs[idx].pitch,
                    static_cast<uint16_t*>(tensor_infos.front().buffer->device()) + idx * infer_size_,
                    max_shape.w,
                    max_shape.z,
                    affine_transforms[idx],
                    infer_config.config,
                    stream);
            }
        }
    }

    // enqueue
    if (!manager_->enqueueV3(stream)) {
        throw std::runtime_error("infer err");
    }

    // d2h
    for (auto& tensor_info : tensor_infos) {
        if (!tensor_info.input) {
            tensor_info.buffer->deviceToHost(stream);
        }
    }

    // sync
    CHECK(cudaStreamSynchronize(stream));
}

} // namespace trtyolo
