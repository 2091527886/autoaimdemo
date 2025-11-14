/**
 * @file trtyolo.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief TensorRT YOLO 模型推理相关类和结构体的定义（聚合头，API 不变，仅拆分声明到多个头文件）
 * @date 2025-06-01
 *
 * Copyright (c) 2025
 */

#pragma once

// Aggregated public header. Keep API identical; only split declarations to multiple headers.
// 统一从本目录下的子头导入，外部仍只需包含 trtyolo.hpp。

#include "api.hpp"
#include "types.hpp"
#include "results.hpp"
#include "infer_option.hpp"
#include "base_model.hpp"
#include "model_classify.hpp"
#include "model_detect.hpp"
#include "model_obb.hpp"
#include "model_segment.hpp"
#include "model_pose.hpp"

// 无其它声明，所有对外可见声明已在以上子头文件中。

namespace trtyolo {} // namespace trtyolo
