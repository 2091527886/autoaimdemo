#include "OCSort.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include "Log.hpp"
#include "ocs_state.cuh"   // ocs::Box
#include "ocs_out.cuh"     // ocs::OcsPackedTrk
#include "Log.hpp"

#include "TensorRT-YOLO/modules/trtyolo/utils/common.hpp"
using trtyolo::checkCudaError;

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
        auto mall = [&](void** p, size_t sz, const char* name){ auto e=cudaMalloc(p, sz); if (e!=cudaSuccess){ LO << P_E "OCS.ensure.malloc fail " << name << " sz=" << (unsigned long long)sz << " err=" << (int)e << END; } };
        mall((void**)&new_kf_x, newCapM*7*sizeof(float), "kf_x");
        mall((void**)&new_kf_P, newCapM*49*sizeof(float), "kf_P");
        mall((void**)&new_meta_obs, newCapM*sizeof(float4), "meta_obs");
        mall((void**)&new_meta_id, newCapM*sizeof(int), "meta_id");
        mall((void**)&new_meta_cls, newCapM*sizeof(int), "meta_cls");
        mall((void**)&new_meta_score, newCapM*sizeof(float), "meta_score");
        mall((void**)&new_meta_tsu, newCapM*sizeof(int), "meta_tsu");
        mall((void**)&new_meta_hit, newCapM*sizeof(int), "meta_hit");
        mall((void**)&new_obs_prev, newCapM*sizeof(float2), "obs_prev");
        mall((void**)&new_obs_curr, newCapM*sizeof(float2), "obs_curr");
        mall((void**)&new_trk_cls, newCapM*sizeof(int), "trk_cls");
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
        // 地址快照（持久缓冲）
        }

    // 更新 cap 值，随后用新容量重建其它工作缓冲
    if (needN) cap_N_ = newCapN;
    if (needM) cap_M_ = newCapM;

    size_t szMat = (size_t)cap_N_*cap_M_*sizeof(float);
    // 逐个处理非持久缓冲：释放并按新容量分配
    auto freem = [&](void** p){ if (*p){ auto e=cudaFree(*p); if(e!=cudaSuccess){ LO << P_E "OCS.ensure.free fail ptr=" << *p << " err=" << (int)e << END; } *p=nullptr; } };
    auto mall = [&](void** p, size_t sz, const char* name){ auto e=cudaMalloc(p, sz); if (e!=cudaSuccess){ LO << P_E "OCS.ensure.malloc fail " << name << " sz=" << (unsigned long long)sz << " err=" << (int)e << END; } };
    freem((void**)&db_.iou);  mall((void**)&db_.iou,  szMat, "iou");
    freem((void**)&db_.ang);  mall((void**)&db_.ang,  szMat, "ang");
    freem((void**)&db_.cost); mall((void**)&db_.cost, szMat, "cost");
    freem((void**)&db_.pairs); mall((void**)&db_.pairs, cap_N_*2*sizeof(int), "pairs");
    freem((void**)&db_.k);     mall((void**)&db_.k, sizeof(int), "k");

    // auction pooled buffers
    freem((void**)&db_.auc_price);    mall((void**)&db_.auc_price, cap_M_*sizeof(float), "auc_price");
    freem((void**)&db_.auc_owner);    mall((void**)&db_.auc_owner, cap_M_*sizeof(int), "auc_owner");
    freem((void**)&db_.auc_det_match);mall((void**)&db_.auc_det_match, cap_N_*sizeof(int), "auc_det_match");
    freem((void**)&db_.auc_best_j);   mall((void**)&db_.auc_best_j, cap_N_*sizeof(int), "auc_best_j");
    freem((void**)&db_.auc_bids);     mall((void**)&db_.auc_bids, cap_N_*sizeof(float), "auc_bids");
    freem((void**)&db_.auc_colwin);   mall((void**)&db_.auc_colwin, cap_M_*sizeof(unsigned long long), "auc_colwin");

    // derived track buffers
    freem((void**)&db_.trk_boxes); mall((void**)&db_.trk_boxes, cap_M_*sizeof(ocs::Box), "trk_boxes");
    freem((void**)&db_.trk_valid); mall((void**)&db_.trk_valid, cap_M_*sizeof(float), "trk_valid");
    freem((void**)&db_.trk_vdc);   mall((void**)&db_.trk_vdc,   cap_M_*sizeof(float2), "trk_vdc");

    // KF per-frame update masks
    freem((void**)&db_.kf_z); mall((void**)&db_.kf_z, cap_M_*4*sizeof(float), "kf_z");
    freem((void**)&db_.kf_m); mall((void**)&db_.kf_m, cap_M_*sizeof(int), "kf_m");

    // BYTE buffers
    freem((void**)&db_.det_used);    mall((void**)&db_.det_used, cap_N_*sizeof(int), "det_used");
    freem((void**)&db_.trk_matched); mall((void**)&db_.trk_matched, cap_M_*sizeof(int), "trk_matched");
    freem((void**)&db_.sel_det);     mall((void**)&db_.sel_det, cap_M_*sizeof(int), "sel_det");
    freem((void**)&db_.unused_det_idx); mall((void**)&db_.unused_det_idx, cap_N_*sizeof(int), "unused_det_idx");
    freem((void**)&db_.unused_det_cnt); mall((void**)&db_.unused_det_cnt, sizeof(int), "unused_det_cnt");

    // optional aligned det buffer
    freem((void**)&db_.det_xyxy_aligned); cudaMalloc(&db_.det_xyxy_aligned, cap_N_*4*sizeof(float));

    // outputs compact buffers (legacy; keep for compatibility)
    freem((void**)&db_.out_trk);   mall((void**)&db_.out_trk,   cap_M_*sizeof(float4), "out_trk");
    freem((void**)&db_.out_obs);   mall((void**)&db_.out_obs,   cap_M_*sizeof(float4), "out_obs");
    freem((void**)&db_.out_score); mall((void**)&db_.out_score, cap_M_*sizeof(float), "out_score");
    freem((void**)&db_.out_cls);   mall((void**)&db_.out_cls,   cap_M_*sizeof(int), "out_cls");
    freem((void**)&db_.out_id);    mall((void**)&db_.out_id,    cap_M_*sizeof(int), "out_id");
    freem((void**)&db_.out_valid); mall((void**)&db_.out_valid, cap_M_*sizeof(int), "out_valid");
    // packed AoS buffer
    freem((void**)&db_.out_packed); mall((void**)&db_.out_packed, cap_M_*sizeof(ocs::OcsPackedTrk), "out_packed");

    // buffers for new tracks (gathered)
    freem((void**)&db_.new_xyxy);   mall((void**)&db_.new_xyxy,   cap_N_*4*sizeof(float), "new_xyxy");
    freem((void**)&db_.new_scores); mall((void**)&db_.new_scores, cap_N_*sizeof(float), "new_scores");
    freem((void**)&db_.new_cls);    mall((void**)&db_.new_cls,    cap_N_*sizeof(int), "new_cls");

    // tmp for compaction (size by M)
    freem((void**)&db_.tmp_kf_x); mall((void**)&db_.tmp_kf_x, cap_M_*7*sizeof(float), "tmp_kf_x");
    freem((void**)&db_.tmp_kf_P); mall((void**)&db_.tmp_kf_P, cap_M_*49*sizeof(float), "tmp_kf_P");
    freem((void**)&db_.tmp_meta_id);    mall((void**)&db_.tmp_meta_id,    cap_M_*sizeof(int), "tmp_meta_id");
    freem((void**)&db_.tmp_meta_cls);   mall((void**)&db_.tmp_meta_cls,   cap_M_*sizeof(int), "tmp_meta_cls");
    freem((void**)&db_.tmp_meta_score); mall((void**)&db_.tmp_meta_score, cap_M_*sizeof(float), "tmp_meta_score");
    freem((void**)&db_.tmp_meta_tsu);   mall((void**)&db_.tmp_meta_tsu,   cap_M_*sizeof(int), "tmp_meta_tsu");
    freem((void**)&db_.tmp_meta_hit);   mall((void**)&db_.tmp_meta_hit,   cap_M_*sizeof(int), "tmp_meta_hit");
    freem((void**)&db_.tmp_meta_obs);   mall((void**)&db_.tmp_meta_obs,   cap_M_*sizeof(float4), "tmp_meta_obs");
    freem((void**)&db_.tmp_obs_prev);   mall((void**)&db_.tmp_obs_prev,   cap_M_*sizeof(float2), "tmp_obs_prev");
    freem((void**)&db_.tmp_obs_curr);   mall((void**)&db_.tmp_obs_curr,   cap_M_*sizeof(float2), "tmp_obs_curr");
    freem((void**)&db_.tmp_trk_cls);    mall((void**)&db_.tmp_trk_cls,    cap_M_*sizeof(int), "tmp_trk_cls");
    freem((void**)&db_.comp_newT);      mall((void**)&db_.comp_newT, sizeof(int), "comp_newT");

    // ensure 内：输出依赖缓冲地址快照 + 只读校验
    // 指针属性审计（设备/主机），帮助定位潜在错误指针来源
    // 强同步：仅当 needM 时，确保所有迁移/分配在本流上完成
    cudaStreamSynchronize(stream);
}
