#include "infer/backend.hpp"
#include <cstring>
#include <cstdint>
#include <stdexcept>

namespace trtyolo {

void TrtBackend::staticInfer(const std::vector<Image>& inputs) {
    auto num = inputs.size();

    // validate
    if (num < 1 || num > max_shape.x) {
        throw std::invalid_argument("Number of inputs out of range");
    }

    if (infer_config.input_shape.has_value()) {
        if (infer_config.cuda_mem) {
            for (int idx = 0; idx < num; ++idx) {
                void* infer_device_ptr = static_cast<void*>(static_cast<uint16_t*>(tensor_infos.front().buffer->device()) + idx * infer_size_);
                void* kernelParams[] = {
                    (void*)&inputs[idx].ptr,
                    (void*)&inputs[idx].width,
                    (void*)&inputs[idx].height,
                    (void*)&inputs[idx].pitch,
                    (void*)&infer_device_ptr,
                    (void*)&max_shape.w,
                    (void*)&max_shape.z,
                    (void*)&affine_transforms.front(),
                    (void*)&infer_config.config};
                cuda_graph_.updateKernelNodeParams(idx, kernelParams);
            }
        } else {
            for (int idx = 0; idx < num; ++idx) {
                uint8_t* dst = static_cast<uint8_t*>(inputs_buffer_->host()) + idx * input_image_bytes_;
                const uint8_t* src = static_cast<const uint8_t*>(inputs[idx].ptr);
                size_t row_bytes = input_row_bytes_;
                for (int r = 0; r < infer_config.input_shape->x; ++r) {
                    std::memcpy(dst + r * row_bytes, src + r * inputs[idx].pitch, row_bytes);
                }
            }
        }
    } else {
        if (!infer_config.cuda_mem) {
            int              total_size = 0;
            std::vector<int> input_sizes(num);
            for (int idx = 0; idx < num; ++idx) {
                input_sizes[idx]  = inputs[idx].height * inputs[idx].pitch;
                total_size       += input_sizes[idx];
            }
            inputs_buffer_->allocate(total_size);
            uint8_t* input_ptr = static_cast<uint8_t*>(inputs_buffer_->host());
            for (int idx = 0; idx < num; ++idx) {
                std::memcpy(input_ptr, inputs[idx].ptr, input_sizes[idx]);
                input_ptr += input_sizes[idx];
            }
            if (buffer_type_ == BufferType::Discrete) {
                cuda_graph_.updateMemcpyNodeParams(0, inputs_buffer_->host(), inputs_buffer_->device(), total_size);
            }
        }

        uint8_t* input_ptr = !infer_config.cuda_mem ? static_cast<uint8_t*>(inputs_buffer_->device()) : nullptr;
        for (int idx = 0; idx < num; ++idx) {
            affine_transforms[idx].updateParams(inputs[idx].width, inputs[idx].height, max_shape.w, max_shape.z);
            void* infer_device_ptr = static_cast<void*>(static_cast<uint16_t*>(tensor_infos.front().buffer->device()) + idx * infer_size_);
            void* kernelParams[] = {
                infer_config.cuda_mem ? (void*)&inputs[idx].ptr : (void*)&input_ptr,
                (void*)&inputs[idx].width,
                (void*)&inputs[idx].height,
                (void*)&inputs[idx].pitch,
                (void*)&infer_device_ptr,
                (void*)&max_shape.w,
                (void*)&max_shape.z,
                (void*)&affine_transforms[idx],
                (void*)&infer_config.config};
            int node_idx = (infer_config.cuda_mem || buffer_type_ != BufferType::Discrete) ? idx : idx + 1;
            cuda_graph_.updateKernelNodeParams(node_idx, kernelParams);
            if (!infer_config.cuda_mem) {
                input_ptr += inputs[idx].height * inputs[idx].pitch;
            }
        }
    }

    // launch graph
    cuda_graph_.launch(stream);
}

} // namespace trtyolo
