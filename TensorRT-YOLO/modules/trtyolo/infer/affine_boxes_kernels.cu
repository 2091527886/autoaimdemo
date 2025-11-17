#include "affine_boxes_kernels.cuh"

namespace trtyolo {

__global__ void k_affine_boxes(float* xyxy, int num, int pad_x, int pad_y, float scale){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=num) return;
    float* p = xyxy + i*4;
    p[0] = (p[0] - pad_x) / scale;
    p[1] = (p[1] - pad_y) / scale;
    p[2] = (p[2] - pad_x) / scale;
    p[3] = (p[3] - pad_y) / scale;
}

void cudaAffineBoxesInverse(float* xyxy, int num, int pad_x, int pad_y, float scale, cudaStream_t stream){
    int tb=256, gb=(num+tb-1)/tb; k_affine_boxes<<<gb,tb,0,stream>>>(xyxy,num,pad_x,pad_y,scale);
}

} // namespace trtyolo
