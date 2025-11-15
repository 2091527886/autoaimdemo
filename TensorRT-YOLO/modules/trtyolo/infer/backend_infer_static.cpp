#include "infer/backend.hpp"
#include "letterbox.hpp"
#include "core/core.hpp"
#include "infer/backend_utils.hpp"
#include "infer/pch_infer.hpp"

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
            copy_fixed_to_staging(inputs_buffer_.get(), inputs, input_image_bytes_, input_row_bytes_, infer_config);
        }
    } else {
        if (!infer_config.cuda_mem) {
            int              total_size = 0;
            std::vector<int> input_sizes;
            compute_sizes_and_affines(inputs, max_shape, input_sizes, total_size, affine_transforms);
            inputs_buffer_->allocate(total_size);
            uint8_t* input_ptr = static_cast<uint8_t*>(inputs_buffer_->host());
            for (int idx = 0; idx < num; ++idx) {
                std::memcpy(input_ptr, inputs[idx].ptr, input_sizes[idx]);
                input_ptr += input_sizes[idx];
            }
            update_memcpy_node_if_needed(cuda_graph_, 0, inputs_buffer_->host(), inputs_buffer_->device(), total_size, buffer_type_);
        }

        uint8_t* input_ptr = !infer_config.cuda_mem ? static_cast<uint8_t*>(inputs_buffer_->device()) : nullptr;
        for (int idx = 0; idx < num; ++idx) {
            affine_transforms[idx].updateParams(inputs[idx].width, inputs[idx].height, max_shape.w, max_shape.z);
            void* infer_device_ptr = static_cast<void*>(static_cast<uint16_t*>(tensor_infos.front().buffer->device()) + idx * infer_size_);
            int node_idx = (infer_config.cuda_mem || buffer_type_ != BufferType::Discrete) ? idx : idx + 1;
            const void* src_ptr = infer_config.cuda_mem ? inputs[idx].ptr : input_ptr;
            update_kernel_params_single(cuda_graph_, node_idx,
                                       src_ptr,
                                       inputs[idx].width, inputs[idx].height, inputs[idx].pitch,
                                       infer_device_ptr,
                                       max_shape.w, max_shape.z,
                                       &affine_transforms[idx], &infer_config.config);
            if (!infer_config.cuda_mem) {
                input_ptr += inputs[idx].height * inputs[idx].pitch;
            }
        }
    }

    // launch graph
    cuda_graph_.launch(stream);
}

} // namespace trtyolo
