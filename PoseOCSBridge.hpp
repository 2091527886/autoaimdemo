#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "Detection.hpp"
#include "OCSort.hpp"

// 将 pose 的 host 检测结果(ds)上传到设备侧，调用 OCSort::update_dev，然后释放临时显存。
// 仅用于将 OCSort 接入“多目标选择”，不参与瞄准追踪。
// 返回 true 表示成功（即使 N==0 也返回 true 并输出空轨迹）。
inline bool ocs_update_from_pose_host(const std::vector<Detection>& ds,
                                      OCSort& ocs,
                                      int img_w,
                                      int img_h,
                                      std::vector<OCSort::Trk>& out_trks,
                                      cudaStream_t stream = 0){
    out_trks.clear();
    const int N = (int)ds.size();
    if (N <= 0){ return true; }

    // 准备 host 打包缓冲
    std::vector<float> h_xyxy(N*4);
    std::vector<float> h_sc(N);
    std::vector<int>   h_cl(N);
    for (int i=0;i<N;++i){
        const auto& d = ds[i];
        h_xyxy[i*4+0] = (float)d.box.x;
        h_xyxy[i*4+1] = (float)d.box.y;
        h_xyxy[i*4+2] = (float)(d.box.x + d.box.width);
        h_xyxy[i*4+3] = (float)(d.box.y + d.box.height);
        h_sc[i] = d.confidence;
        h_cl[i] = d.class_id;
    }

    // 设备缓冲
    float *d_xyxy=nullptr, *d_sc=nullptr; int *d_cl=nullptr;
    size_t sz_xyxy = (size_t)N * 4 * sizeof(float);
    size_t sz_sc   = (size_t)N * sizeof(float);
    size_t sz_cl   = (size_t)N * sizeof(int);
    cudaError_t e1 = cudaMalloc((void**)&d_xyxy, sz_xyxy);
    cudaError_t e2 = cudaMalloc((void**)&d_sc,   sz_sc);
    cudaError_t e3 = cudaMalloc((void**)&d_cl,   sz_cl);
    if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess){
        if (d_xyxy) cudaFree(d_xyxy);
        if (d_sc)   cudaFree(d_sc);
        if (d_cl)   cudaFree(d_cl);
        return false;
    }

    cudaMemcpyAsync(d_xyxy, h_xyxy.data(), sz_xyxy, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sc,   h_sc.data(),   sz_sc,   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_cl,   h_cl.data(),   sz_cl,   cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // 调用 OCSort（设备内核 + D2H 打包输出）
    ocs.update_dev(d_xyxy, d_sc, d_cl, N, out_trks, img_w, img_h, stream);

    // 释放临时显存
    cudaFree(d_xyxy);
    cudaFree(d_sc);
    cudaFree(d_cl);
    return true;
}
