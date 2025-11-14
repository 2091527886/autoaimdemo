#include "infer/backend.hpp"
#include "utils/common.hpp"
#include <cstdint>
#include <utility>

namespace trtyolo {

void TrtBackend::captureCudaGraph() {
    // Step 1: Pre-inference execution before graph capture
    {
        for (auto& tensor_info : tensor_infos) {
            manager_->setTensorAddress(tensor_info.name.c_str(), tensor_info.buffer->device());
        }
        if (!manager_->enqueueV3(stream)) {
            throw std::runtime_error("captureCudaGraph: EnqueueV3 failed before graph creation.");
        }
        CHECK(cudaStreamSynchronize(stream));
    }

    // Lambda: Calculate input size and device pointers
    auto calculate_input_size_and_device = [&](int idx, int input_width, int input_height) {
        size_t input_size = (infer_config.input_shape.has_value() && !infer_config.cuda_mem)
                                ? input_image_bytes_
                                : static_cast<size_t>(input_height) * input_width * max_shape.y;
        void*  input_device = static_cast<uint8_t*>(inputs_buffer_->device()) + idx * input_size;
        void*  infer_device = static_cast<void*>(static_cast<uint16_t*>(tensor_infos.front().buffer->device()) + idx * infer_size_);
        return std::make_pair(input_device, infer_device);
    };

    // Lambda: Perform WarpAffine operation
    auto run_letterbox = [&](bool multi, int input_width, int input_height) {
        size_t src_pitch = (infer_config.input_shape.has_value() && !infer_config.cuda_mem)
                               ? input_row_bytes_
                               : static_cast<size_t>(input_width) * max_shape.y; // assume tightly packed CHW when not fixed-host path
        if (multi) {
            cudaMultiLetterboxHalf(
                inputs_buffer_->device(),
                input_width,
                input_height,
                src_pitch,
                tensor_infos.front().buffer->device(),
                max_shape.w,
                max_shape.z,
                affine_transforms.front(),
                infer_config.config,
                max_shape.x,
                stream);
        } else {
            for (int idx = 0; idx < max_shape.x; ++idx) {
                auto [input_device, infer_device] = calculate_input_size_and_device(idx, input_width, input_height);
                cudaLetterboxHalf(
                    input_device,
                    input_width,
                    input_height,
                    src_pitch,
                    infer_device,
                    max_shape.w,
                    max_shape.z,
                    affine_transforms[idx],
                    infer_config.config,
                    stream);
            }
        }
    };

    // Step 2: Begin CUDA Graph Capture
    cuda_graph_.beginCapture(stream);

    // Step 3: Perform memory transfer and WarpAffine based on configuration
    if (infer_config.cuda_mem) {
        int input_width  = infer_config.input_shape ? infer_config.input_shape->y : max_shape.w;
        int input_height = infer_config.input_shape ? infer_config.input_shape->x : max_shape.z;
        run_letterbox(false, input_width, input_height);
    } else {
        inputs_buffer_->hostToDevice(stream);
        int input_width  = infer_config.input_shape ? infer_config.input_shape->y : max_shape.w;
        int input_height = infer_config.input_shape ? infer_config.input_shape->x : max_shape.z;
        run_letterbox(infer_config.input_shape.has_value(), input_width, input_height);
    }

    // Step 4: Enqueue inference
    if (!manager_->enqueueV3(stream)) {
        throw std::runtime_error("captureCudaGraph: EnqueueV3 failed when graph creation.");
    }

    // Step 5: Copy output tensors to host if required
    for (auto& tensor_info : tensor_infos) {
        if (!tensor_info.input) {
            tensor_info.buffer->deviceToHost(stream);
        }
    }

    // Step 6: End CUDA Graph Capture
    cuda_graph_.endCapture(stream);

    // Step 7: Initialize CUDA Graph Nodes
    if (!(infer_config.input_shape.has_value() && !infer_config.cuda_mem)) {
        int num_nodes = max_shape.x + (infer_config.cuda_mem ? 0 : 1);
        cuda_graph_.initializeNodes(num_nodes);
    }
}

} // namespace trtyolo
