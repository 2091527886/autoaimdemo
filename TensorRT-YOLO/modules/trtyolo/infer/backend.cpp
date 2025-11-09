/**
 * @file backend.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief TensorRT 推理后端实现
 * @date 2025-01-15
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#include <algorithm>
#include <cstring>
#include <cstdint>

#include "backend.hpp"

namespace trtyolo {

TrtBackend::TrtBackend(const std::string& trt_engine_file, const InferConfig& infer_config) : infer_config(infer_config) {
    cudaSetDevice(infer_config.device_id);  // < 设置设备
    CHECK(cudaStreamCreate(&stream));       // < 创建 stream

    // 是否支持 Zero Copy
    zero_copy_ = SupportsIntegratedZeroCopy(infer_config.device_id);

    // 创建 TRTManager 实例
    manager_ = std::make_unique<TRTManager>();

    // 获取 Engine Buffer
    std::string engine_buffer;
    ReadBinaryFromFile(trt_engine_file, &engine_buffer);

    // 调用 initialize 方法进行初始化
    manager_->initialize(engine_buffer.data(), engine_buffer.size());

    // 获取 TensorInfo
    getTensorInfo();

    // 初始化相关变量
    initialize();

    // 捕获 Cuda Graph，当模型是静态时
    if (!dynamic) captureCudaGraph();
}

std::unique_ptr<TrtBackend> TrtBackend::clone() {
    auto clone_backend          = std::make_unique<TrtBackend>();
    clone_backend->infer_config = infer_config;

    cudaSetDevice(infer_config.device_id);            // < 设置设备
    CHECK(cudaStreamCreate(&clone_backend->stream));  // < 创建 stream

    // 是否支持 Zero Copy
    clone_backend->zero_copy_ = zero_copy_;

    clone_backend->manager_ = manager_->clone();

    // 获取 TensorInfo
    clone_backend->getTensorInfo();

    // 初始化相关变量
    clone_backend->initialize();

    // 捕获 Cuda Graph，当模型是静态时
    if (!clone_backend->dynamic) clone_backend->captureCudaGraph();

    return clone_backend;
}

TrtBackend::~TrtBackend() {
    std::vector<TensorInfo>().swap(tensor_infos);
    std::vector<LetterboxTransform>().swap(affine_transforms);
    if (!dynamic) cuda_graph_.destroy();
    CHECK(cudaStreamDestroy(stream));
}

void TrtBackend::getTensorInfo() {
    std::vector<TensorInfo>().swap(tensor_infos);
    buffer_type_     = infer_config.enable_managed_memory ? BufferType::Unified : (zero_copy_ ? BufferType::Mapped : BufferType::Discrete);
    auto num_tensors = manager_->getNbIOTensors();
    for (auto i = 0; i < num_tensors; ++i) {
        std::string name  = std::string(manager_->getIOTensorName(i));
        auto        shape = manager_->getTensorShape(name.c_str());
        auto        dtype = manager_->getTensorDataType(name.c_str());
        bool        input = (manager_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT);

        if (input) {
            dynamic = std::any_of(shape.d, shape.d + shape.nbDims, [](int val) { return val == -1; });
            if (dynamic) {
                shape     = manager_->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);
                min_shape = make_int4(shape.d[0], shape.d[1], shape.d[2], shape.d[3]);
                shape     = manager_->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
                // < 打印接受范围
            }
            max_shape = make_int4(shape.d[0], shape.d[1], shape.d[2], shape.d[3]);
        } else if (!input && dynamic) {
            shape.d[0] = max_shape.x;
        }
        tensor_infos.emplace_back(name, shape, dtype, input, input ? BufferType::Device : buffer_type_);
    }
}

void TrtBackend::initialize() {
    // 清空并释放affine_transforms和image_buffers_中的资源
    std::vector<LetterboxTransform>().swap(affine_transforms);
    inputs_buffer_ = BufferFactory::createBuffer(buffer_type_);

    infer_size_ = max_shape.y * max_shape.w * max_shape.z;

    if (infer_config.input_shape.has_value()) {
        // 固定输入尺寸：源为 BGRA（4 字节/像素），按行（pitch）拷贝
        int in_w = infer_config.input_shape->y;
        int in_h = infer_config.input_shape->x;
        input_row_bytes_  = static_cast<size_t>(in_w) * 4;           // BGRA
        input_image_bytes_ = static_cast<size_t>(in_h) * input_row_bytes_;
        input_size_ = static_cast<int>(input_image_bytes_);          // 兼容旧 int 字段

        affine_transforms.emplace_back(LetterboxTransform());
        affine_transforms.front().updateParams(
            in_w,
            in_h,
            max_shape.w,
            max_shape.z);
        inputs_buffer_->allocate(max_shape.x * input_image_bytes_);  // < 按最大情况分配空间（逐行拷贝）
    } else {
        // 输入尺寸不固定时
        affine_transforms.resize(max_shape.x, LetterboxTransform());
        if (!dynamic) inputs_buffer_->allocate(max_shape.x * infer_size_);
    }
}

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
    auto warp_affine = [&](bool multi, int input_width, int input_height) {
        size_t src_pitch = (infer_config.input_shape.has_value() && !infer_config.cuda_mem)
                               ? input_row_bytes_
                               : static_cast<size_t>(input_width) * max_shape.y; // assume tightly packed CHW when not fixed-host path
        if (multi) {
            // Multi-instance WarpAffine
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
            // Single-instance WarpAffine
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
        warp_affine(false, input_width, input_height);
    } else {
        inputs_buffer_->hostToDevice(stream);

        int input_width  = infer_config.input_shape ? infer_config.input_shape->y : max_shape.w;
        int input_height = infer_config.input_shape ? infer_config.input_shape->x : max_shape.z;
        warp_affine(infer_config.input_shape.has_value(), input_width, input_height);
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
    // 如果输入形状存在且不在CUDA内存中，则不需要调用initializeNodes
    if (!(infer_config.input_shape.has_value() && !infer_config.cuda_mem)) {
        int num_nodes = max_shape.x + (infer_config.cuda_mem ? 0 : 1);  // 如果 buffer_type 不是 Dis 也不需要 + 1
        cuda_graph_.initializeNodes(num_nodes);
    }
}

void TrtBackend::staticInfer(const std::vector<Image>& inputs) {
    auto num = inputs.size();

    // 1. 判断输入是否合法，尽早返回
    if (num < 1 || num > max_shape.x) {
        throw std::invalid_argument("Number of inputs out of range");
    }

    if (infer_config.input_shape.has_value()) {
        if (infer_config.cuda_mem) {
            for (int idx = 0; idx < num; ++idx) {
                // 计算 infer_device_ptr，避免重复计算
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

                // 更新 kernel 参数
                cuda_graph_.updateKernelNodeParams(idx, kernelParams);
            }
        } else {
            // 主机拷贝需按行考虑 pitch
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

            // 计算输入大小，并累加总大小
            for (int idx = 0; idx < num; ++idx) {
                input_sizes[idx]  = inputs[idx].height * inputs[idx].pitch;
                total_size       += input_sizes[idx];
            }

            // 在主机内存中分配空间并拷贝数据
            inputs_buffer_->allocate(total_size);
            uint8_t* input_ptr = static_cast<uint8_t*>(inputs_buffer_->host());

            for (int idx = 0; idx < num; ++idx) {
                std::memcpy(input_ptr, inputs[idx].ptr, input_sizes[idx]);
                input_ptr += input_sizes[idx];
            }

            // 更新 Memcpy 节点
            if (buffer_type_ == BufferType::Discrete) {
                cuda_graph_.updateMemcpyNodeParams(0, inputs_buffer_->host(), inputs_buffer_->device(), total_size);
            }
        }

        // 更新 kernel 节点
        uint8_t* input_ptr = !infer_config.cuda_mem ? static_cast<uint8_t*>(inputs_buffer_->device()) : nullptr;
        for (int idx = 0; idx < num; ++idx) {
            affine_transforms[idx].updateParams(inputs[idx].width, inputs[idx].height, max_shape.w, max_shape.z);
            // 计算 infer_device_ptr，避免重复计算
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

            // 判断 idx 更新 kernel 参数
            int node_idx = (infer_config.cuda_mem || buffer_type_ != BufferType::Discrete) ? idx : idx + 1;
            cuda_graph_.updateKernelNodeParams(node_idx, kernelParams);

            // 更新 input_ptr 仅在 cuda_mem 为 false 时
            if (!infer_config.cuda_mem) {
                input_ptr += inputs[idx].height * inputs[idx].pitch;
            }
        }
    }

    // Launch the CUDA graph
    cuda_graph_.launch(stream);
}

void TrtBackend::dynamicInfer(const std::vector<Image>& inputs) {
    auto num = inputs.size();

    // 1. 判断输入是否合法，尽早返回
    if (num < min_shape.x || num > max_shape.x) {
        throw std::invalid_argument("Number of inputs out of range");
    }

    // 更新 tensor_info 的 shape 和设备地址
    for (auto& tensor_info : tensor_infos) {
        tensor_info.shape.d[0] = num;
        tensor_info.update();
        manager_->setTensorAddress(tensor_info.name.c_str(), tensor_info.buffer->device());
        if (tensor_info.input) {
            manager_->setInputShape(tensor_info.name.c_str(), tensor_info.shape);
        }
    }

    if (infer_config.input_shape.has_value()) {
        // 2. 处理静态输入形状
        if (!infer_config.cuda_mem) {
            // 主机拷贝需按行考虑 pitch
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

        // 计算输入大小，并累加总大小
        for (int idx = 0; idx < num; ++idx) {
            input_sizes[idx]  = inputs[idx].height * inputs[idx].pitch;
            total_size       += input_sizes[idx];
            affine_transforms[idx].updateParams(inputs[idx].width, inputs[idx].height, max_shape.w, max_shape.z);
        }

        // 在主机内存或设备内存中分配空间
        if (!infer_config.cuda_mem) {
            // 在主机内存中分配空间并拷贝数据
            inputs_buffer_->allocate(total_size);
            uint8_t* input_host = static_cast<uint8_t*>(inputs_buffer_->host());

            // 拷贝输入数据到主机内存
            for (int idx = 0; idx < num; ++idx) {
                std::memcpy(input_host, inputs[idx].ptr, input_sizes[idx]);
                input_host += input_sizes[idx];
            }

            // 拷贝到设备内存
            inputs_buffer_->hostToDevice(stream);

            // 在设备内存中进行 WarpAffine 操作
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
            // 直接在设备内存上进行 WarpAffine 操作
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

    // 推理
    if (!manager_->enqueueV3(stream)) {
        throw std::runtime_error("Infer Error.");
    }

    // 数据拷贝从设备到主机
    for (auto& tensor_info : tensor_infos) {
        if (!tensor_info.input) {
            tensor_info.buffer->deviceToHost(stream);
        }
    }

    // 同步流，确保所有 CUDA 操作完成
    CHECK(cudaStreamSynchronize(stream));
}

void TrtBackend::infer(const std::vector<Image>& inputs) {
    cudaSetDevice(infer_config.device_id);  // 推理前切换设备

    if (dynamic) {
        dynamicInfer(inputs);
    } else {
        staticInfer(inputs);
    }
}

}  // namespace trtyolo