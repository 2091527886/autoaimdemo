#include "OCSort.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include "ocs_state.cuh"   // ocs::Box
#include "ocs_out.cuh"     // ocs::OcsPackedTrk

void OCSort::ensure_capacity(int N, int M, cudaStream_t stream){
    // ensure device counters are allocated
    if (!db_.dev_trk_count) { cudaMalloc(&db_.dev_trk_count, sizeof(int)); }
    if (!db_.dev_next_id)   { cudaMalloc(&db_.dev_next_id,   sizeof(int)); }
    bool needN = N > cap_N_;
    bool needM = M > cap_M_;
    if (!needN && !needM) return;

    int newCapN = cap_N_ > 0 ? cap_N_ : 16;
    int newCapM = cap_M_ > 0 ? cap_M_ : 16;
    if (needN) newCapN = std::max(cap_N_ * 2, N);
    if (needM) newCapM = std::max(cap_M_ * 2, M);

    // M 扩容前，读取当前 T（设备常驻轨迹数）用于迁移
    int Tcur = 0;
    if (dev_counter_inited_ && db_.dev_trk_count) {
        cudaMemcpy(&Tcur, db_.dev_trk_count, sizeof(int), cudaMemcpyDeviceToHost);
    } else {
        Tcur = last_T_;
    }

    // 先处理持久状态的迁移（仅当 M 扩容时）
    if (needM){
        // 分配新持久缓冲
        float *new_kf_x=nullptr, *new_kf_P=nullptr; float4 *new_meta_obs=nullptr; int *new_meta_id=nullptr, *new_meta_cls=nullptr, *new_meta_tsu=nullptr, *new_meta_hit=nullptr; float *new_meta_score=nullptr; float2 *new_obs_prev=nullptr, *new_obs_curr=nullptr; int* new_trk_cls=nullptr;
        cudaMalloc(&new_kf_x, newCapM*7*sizeof(float));
        cudaMalloc(&new_kf_P, newCapM*49*sizeof(float));
        cudaMalloc(&new_meta_obs, newCapM*sizeof(float4));
        cudaMalloc(&new_meta_id, newCapM*sizeof(int));
        cudaMalloc(&new_meta_cls, newCapM*sizeof(int));
        cudaMalloc(&new_meta_score, newCapM*sizeof(float));
        cudaMalloc(&new_meta_tsu, newCapM*sizeof(int));
        cudaMalloc(&new_meta_hit, newCapM*sizeof(int));
        cudaMalloc(&new_obs_prev, newCapM*sizeof(float2));
        cudaMalloc(&new_obs_curr, newCapM*sizeof(float2));
        cudaMalloc(&new_trk_cls, newCapM*sizeof(int));
        // 迁移已有数据
        if (Tcur > 0){
            if (db_.kf_x) cudaMemcpyAsync(new_kf_x, db_.kf_x, Tcur*7*sizeof(float), cudaMemcpyDeviceToDevice, stream);
            if (db_.kf_P) cudaMemcpyAsync(new_kf_P, db_.kf_P, Tcur*49*sizeof(float), cudaMemcpyDeviceToDevice, stream);
            if (db_.meta_obs) cudaMemcpyAsync(new_meta_obs, db_.meta_obs, Tcur*sizeof(float4), cudaMemcpyDeviceToDevice, stream);
            if (db_.meta_id) cudaMemcpyAsync(new_meta_id, db_.meta_id, Tcur*sizeof(int), cudaMemcpyDeviceToDevice, stream);
            if (db_.meta_cls) cudaMemcpyAsync(new_meta_cls, db_.meta_cls, Tcur*sizeof(int), cudaMemcpyDeviceToDevice, stream);
            if (db_.meta_score) cudaMemcpyAsync(new_meta_score, db_.meta_score, Tcur*sizeof(float), cudaMemcpyDeviceToDevice, stream);
            if (db_.meta_tsu) cudaMemcpyAsync(new_meta_tsu, db_.meta_tsu, Tcur*sizeof(int), cudaMemcpyDeviceToDevice, stream);
            if (db_.meta_hit) cudaMemcpyAsync(new_meta_hit, db_.meta_hit, Tcur*sizeof(int), cudaMemcpyDeviceToDevice, stream);
            if (db_.obs_prev) cudaMemcpyAsync(new_obs_prev, db_.obs_prev, Tcur*sizeof(float2), cudaMemcpyDeviceToDevice, stream);
            if (db_.obs_curr) cudaMemcpyAsync(new_obs_curr, db_.obs_curr, Tcur*sizeof(float2), cudaMemcpyDeviceToDevice, stream);
            if (db_.trk_cls) cudaMemcpyAsync(new_trk_cls, db_.trk_cls, Tcur*sizeof(int), cudaMemcpyDeviceToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        // 释放旧持久缓冲并替换
        if (db_.kf_x) cudaFree(db_.kf_x); db_.kf_x = new_kf_x;
        if (db_.kf_P) cudaFree(db_.kf_P); db_.kf_P = new_kf_P;
        if (db_.meta_obs) cudaFree(db_.meta_obs); db_.meta_obs = new_meta_obs;
        if (db_.meta_id) cudaFree(db_.meta_id); db_.meta_id = new_meta_id;
        if (db_.meta_cls) cudaFree(db_.meta_cls); db_.meta_cls = new_meta_cls;
        if (db_.meta_score) cudaFree(db_.meta_score); db_.meta_score = new_meta_score;
        if (db_.meta_tsu) cudaFree(db_.meta_tsu); db_.meta_tsu = new_meta_tsu;
        if (db_.meta_hit) cudaFree(db_.meta_hit); db_.meta_hit = new_meta_hit;
        if (db_.obs_prev) cudaFree(db_.obs_prev); db_.obs_prev = new_obs_prev;
        if (db_.obs_curr) cudaFree(db_.obs_curr); db_.obs_curr = new_obs_curr;
        if (db_.trk_cls) cudaFree(db_.trk_cls); db_.trk_cls = new_trk_cls;
    }

    // 更新 cap 值，随后用新容量重建其它工作缓冲
    if (needN) cap_N_ = newCapN;
    if (needM) cap_M_ = newCapM;

    size_t szMat = (size_t)cap_N_*cap_M_*sizeof(float);
    // 逐个处理非持久缓冲：释放并按新容量分配
    auto freem = [&](void** p){ if (*p){ cudaFree(*p); *p=nullptr; } };
    freem((void**)&db_.iou);  cudaMalloc(&db_.iou,  szMat);
    freem((void**)&db_.ang);  cudaMalloc(&db_.ang,  szMat);
    freem((void**)&db_.cost); cudaMalloc(&db_.cost, szMat);
    freem((void**)&db_.pairs); cudaMalloc(&db_.pairs, cap_N_*2*sizeof(int));
    freem((void**)&db_.k);     cudaMalloc(&db_.k, sizeof(int));

    // auction pooled buffers
    freem((void**)&db_.auc_price);    cudaMalloc(&db_.auc_price, cap_M_*sizeof(float));
    freem((void**)&db_.auc_owner);    cudaMalloc(&db_.auc_owner, cap_M_*sizeof(int));
    freem((void**)&db_.auc_det_match);cudaMalloc(&db_.auc_det_match, cap_N_*sizeof(int));
    freem((void**)&db_.auc_best_j);   cudaMalloc(&db_.auc_best_j, cap_N_*sizeof(int));
    freem((void**)&db_.auc_bids);     cudaMalloc(&db_.auc_bids, cap_N_*sizeof(float));
    freem((void**)&db_.auc_colwin);   cudaMalloc(&db_.auc_colwin, cap_M_*sizeof(unsigned long long));

    // derived track buffers
    freem((void**)&db_.trk_boxes); cudaMalloc(&db_.trk_boxes, cap_M_*sizeof(ocs::Box));
    freem((void**)&db_.trk_valid); cudaMalloc(&db_.trk_valid, cap_M_*sizeof(float));
    freem((void**)&db_.trk_vdc);   cudaMalloc(&db_.trk_vdc,   cap_M_*sizeof(float2));

    // KF per-frame update masks
    freem((void**)&db_.kf_z); cudaMalloc(&db_.kf_z, cap_M_*4*sizeof(float));
    freem((void**)&db_.kf_m); cudaMalloc(&db_.kf_m, cap_M_*sizeof(int));

    // BYTE buffers
    freem((void**)&db_.det_used);    cudaMalloc(&db_.det_used, cap_N_*sizeof(int));
    freem((void**)&db_.trk_matched); cudaMalloc(&db_.trk_matched, cap_M_*sizeof(int));
    freem((void**)&db_.sel_det);     cudaMalloc(&db_.sel_det, cap_M_*sizeof(int));
    freem((void**)&db_.unused_det_idx); cudaMalloc(&db_.unused_det_idx, cap_N_*sizeof(int));
    freem((void**)&db_.unused_det_cnt); cudaMalloc(&db_.unused_det_cnt, sizeof(int));

    // outputs compact buffers (legacy; keep for compatibility)
    freem((void**)&db_.out_trk);   cudaMalloc(&db_.out_trk,   cap_M_*sizeof(float4));
    freem((void**)&db_.out_obs);   cudaMalloc(&db_.out_obs,   cap_M_*sizeof(float4));
    freem((void**)&db_.out_score); cudaMalloc(&db_.out_score, cap_M_*sizeof(float));
    freem((void**)&db_.out_cls);   cudaMalloc(&db_.out_cls,   cap_M_*sizeof(int));
    freem((void**)&db_.out_id);    cudaMalloc(&db_.out_id,    cap_M_*sizeof(int));
    freem((void**)&db_.out_valid); cudaMalloc(&db_.out_valid, cap_M_*sizeof(int));
    // packed AoS buffer
    freem((void**)&db_.out_packed); cudaMalloc(&db_.out_packed, cap_M_*sizeof(ocs::OcsPackedTrk));

    // buffers for new tracks (gathered)
    freem((void**)&db_.new_xyxy);   cudaMalloc(&db_.new_xyxy,   cap_N_*4*sizeof(float));
    freem((void**)&db_.new_scores); cudaMalloc(&db_.new_scores, cap_N_*sizeof(float));
    freem((void**)&db_.new_cls);    cudaMalloc(&db_.new_cls,    cap_N_*sizeof(int));

    // tmp for compaction (size by M)
    freem((void**)&db_.tmp_kf_x); cudaMalloc(&db_.tmp_kf_x, cap_M_*7*sizeof(float));
    freem((void**)&db_.tmp_kf_P); cudaMalloc(&db_.tmp_kf_P, cap_M_*49*sizeof(float));
    freem((void**)&db_.tmp_meta_id);    cudaMalloc(&db_.tmp_meta_id,    cap_M_*sizeof(int));
    freem((void**)&db_.tmp_meta_cls);   cudaMalloc(&db_.tmp_meta_cls,   cap_M_*sizeof(int));
    freem((void**)&db_.tmp_meta_score); cudaMalloc(&db_.tmp_meta_score, cap_M_*sizeof(float));
    freem((void**)&db_.tmp_meta_tsu);   cudaMalloc(&db_.tmp_meta_tsu,   cap_M_*sizeof(int));
    freem((void**)&db_.tmp_meta_hit);   cudaMalloc(&db_.tmp_meta_hit,   cap_M_*sizeof(int));
    freem((void**)&db_.tmp_meta_obs);   cudaMalloc(&db_.tmp_meta_obs,   cap_M_*sizeof(float4));
    freem((void**)&db_.tmp_obs_prev);   cudaMalloc(&db_.tmp_obs_prev,   cap_M_*sizeof(float2));
    freem((void**)&db_.tmp_obs_curr);   cudaMalloc(&db_.tmp_obs_curr,   cap_M_*sizeof(float2));
    freem((void**)&db_.tmp_trk_cls);    cudaMalloc(&db_.tmp_trk_cls,    cap_M_*sizeof(int));
    freem((void**)&db_.comp_newT);      cudaMalloc(&db_.comp_newT, sizeof(int));
}
