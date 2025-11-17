#pragma once
#include <cuda_runtime.h>

namespace trtyolo {

// 将 xyxy 按 letterbox 参数做逆变换：从网络输出(dst)坐标 -> 原图(src)坐标
// xyxy: device pointer, N x 4 (left,top,right,bottom)
void cudaAffineBoxesInverse(float* xyxy, int num, int pad_x, int pad_y, float scale, cudaStream_t stream);

} // namespace trtyolo
