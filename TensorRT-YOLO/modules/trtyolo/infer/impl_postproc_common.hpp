#pragma once

#include "infer/detail/trtyolo_impl.hpp"

namespace trtyolo {

// 小工具：根据 idx 选择 affine（固定输入尺寸复用 front，否则按 batch）
static inline const LetterboxTransform& pick_affine(const InferConfig& cfg, const std::vector<LetterboxTransform>& affs, int idx) {
    return cfg.input_shape.has_value() ? affs.front() : affs[idx];
}

// 小工具：预分配结果数组容量
static inline void reserve_common(DetectRes& r, int n) { r.num = n; r.boxes.reserve(n); r.scores.reserve(n); r.classes.reserve(n); }
static inline void reserve_common(OBBRes& r,    int n) { r.num = n; r.boxes.reserve(n); r.scores.reserve(n); r.classes.reserve(n); }
static inline void reserve_common(SegmentRes& r,int n) { r.num = n; r.boxes.reserve(n); r.scores.reserve(n); r.classes.reserve(n); r.masks.reserve(n); }
static inline void reserve_common(PoseRes& r,   int n) { r.num = n; r.boxes.reserve(n); r.scores.reserve(n); r.classes.reserve(n); r.kpts.reserve(n); }

// 小工具：从 tensor_infos 提取常用张量指针
struct PostprocPtrs {
    int    num;
    float* boxes;
    float* scores;
    int*   classes;
    int    box_size;
};

static inline PostprocPtrs get_ptrs(const std::vector<TensorInfo>& tins, int idx) {
    auto& num_tensor   = const_cast<TensorInfo&>(tins[1]);
    auto& box_tensor   = const_cast<TensorInfo&>(tins[2]);
    auto& score_tensor = const_cast<TensorInfo&>(tins[3]);
    auto& class_tensor = const_cast<TensorInfo&>(tins[4]);
    PostprocPtrs p;
    p.num      = static_cast<int*>(num_tensor.buffer->host())[idx];
    p.boxes    = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
    p.scores   = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
    p.classes  = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];
    p.box_size = box_tensor.shape.d[2];
    return p;
}

// 公共宏：提取、变换并写入结果，尽量减少每个实现文件的重复行
#define TRTYOLO_EXTRACT_LTRB(P, BASE, L, T, R, B) \
    float L = (P).boxes[(BASE)]; \
    float T = (P).boxes[(BASE) + 1]; \
    float R = (P).boxes[(BASE) + 2]; \
    float B = (P).boxes[(BASE) + 3];

#define TRTYOLO_AFFINE4(AFF, L, T, R, B) do{ \
    (AFF).applyTransform((L), (T), &(L), &(T)); \
    (AFF).applyTransform((R), (B), &(R), &(B)); \
}while(0)

#define TRTYOLO_PUSH_DET(RES, P, I, L, T, R, B) do{ \
    (RES).boxes.emplace_back(Box{(L),(T),(R),(B)}); \
    (RES).scores.push_back((P).scores[(I)]); \
    (RES).classes.push_back((P).classes[(I)]); \
}while(0)

#define TRTYOLO_PUSH_OBB(RES, P, I, L, T, R, B, THETA) do{ \
    (RES).boxes.emplace_back(RotatedBox{(L),(T),(R),(B),(THETA)}); \
    (RES).scores.push_back((P).scores[(I)]); \
    (RES).classes.push_back((P).classes[(I)]); \
}while(0)

} // namespace trtyolo
