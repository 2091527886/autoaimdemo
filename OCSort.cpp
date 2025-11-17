#include "OCSort.hpp"
#include "Config.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include "ocs_iou.cuh"
#include "ocs_assign.cuh"
#include "ocs_kf.cuh"
#include "ocs_ocm.cuh"
#include "ocs_byte.cuh"
#include "ocs_out.cuh"
#include "ocs_state.cuh"

// 旧 CPU iou_rect 已删除，避免依赖 OpenCV Rect 类型

OCSort::OCSort(){}

OCSort::~OCSort(){
    // free all device buffers
    auto f=[&](void**p){ if (*p){ cudaFree(*p); *p=nullptr; } };
    f((void**)&db_.iou); f((void**)&db_.ang); f((void**)&db_.cost);
    f((void**)&db_.pairs); f((void**)&db_.k);
    // auction pooled
    f((void**)&db_.auc_price); f((void**)&db_.auc_owner); f((void**)&db_.auc_det_match); f((void**)&db_.auc_best_j); f((void**)&db_.auc_bids); f((void**)&db_.auc_colwin);
    // tracking state
    f((void**)&db_.trk_boxes); f((void**)&db_.trk_valid); f((void**)&db_.trk_vdc); f((void**)&db_.trk_cls);
    f((void**)&db_.kf_x); f((void**)&db_.kf_P); f((void**)&db_.kf_z); f((void**)&db_.kf_m);
    f((void**)&db_.obs_prev); f((void**)&db_.obs_curr);
    f((void**)&db_.det_used); f((void**)&db_.trk_matched); f((void**)&db_.sel_det);
    // outputs
    f((void**)&db_.out_trk); f((void**)&db_.out_obs); f((void**)&db_.out_score); f((void**)&db_.out_cls); f((void**)&db_.out_id); f((void**)&db_.out_valid); f((void**)&db_.out_packed);
    // meta
    f((void**)&db_.meta_id); f((void**)&db_.meta_cls); f((void**)&db_.meta_score); f((void**)&db_.meta_tsu); f((void**)&db_.meta_hit); f((void**)&db_.meta_obs);
    f((void**)&db_.unused_det_idx); f((void**)&db_.unused_det_cnt);
    f((void**)&db_.new_xyxy); f((void**)&db_.new_scores); f((void**)&db_.new_cls);
    // device counters and tmp buffers
    f((void**)&db_.dev_trk_count); f((void**)&db_.dev_next_id);
    f((void**)&db_.tmp_kf_x); f((void**)&db_.tmp_kf_P);
    f((void**)&db_.tmp_meta_id); f((void**)&db_.tmp_meta_cls); f((void**)&db_.tmp_meta_score); f((void**)&db_.tmp_meta_tsu); f((void**)&db_.tmp_meta_hit);
    f((void**)&db_.tmp_meta_obs);
    f((void**)&db_.tmp_obs_prev); f((void**)&db_.tmp_obs_curr);
    f((void**)&db_.tmp_trk_cls);
    f((void**)&db_.comp_newT);
}


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

void OCSort::configure(const Config& c){
    iou_th_  = c.ocs_iou_th;
    max_age_ = c.ocs_max_age;
    min_hits_ = c.ocs_min_hits;
    vdc_weight_ = c.ocs_inertia;
    q_pos_ = c.ocs_kf_q_pos; q_size_ = c.ocs_kf_q_size; r_pos_ = c.ocs_kf_r_pos; r_size_ = c.ocs_kf_r_size;
    if (c.ocs_asso == "iou") afn_ = 0; else if (c.ocs_asso=="giou") afn_ = 1; else if (c.ocs_asso=="ciou") afn_ = 2; else if (c.ocs_asso=="diou") afn_ = 3; else if (c.ocs_asso=="ct_dist") afn_ = 4; else afn_ = 0;
    oru_ = c.ocs_oru; ocm_ = c.ocs_ocm; ocr_ = c.ocs_ocr;
    use_byte_ = c.ocs_use_byte; byte_iou_th_ = c.ocs_byte_iou_th;
    recover_relax_ = c.ocs_recover_relax;
    auction_rounds_ = c.ocs_auction_rounds;
    ocr_slope_ = c.ocs_ocr_slope;
    oru_alpha_ = c.ocs_oru_alpha;

}

void OCSort::reset(){
    next_id_ = 1; kf_inited_=false; last_T_=0;
    // ensure device counters allocated and reset
    if (!db_.dev_trk_count) cudaMalloc(&db_.dev_trk_count, sizeof(int));
    if (!db_.dev_next_id)   cudaMalloc(&db_.dev_next_id,   sizeof(int));
    int zero=0, one=1;
    dev_counter_inited_ = true;
    cudaMemcpy(db_.dev_trk_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(db_.dev_next_id,   &one,  sizeof(int), cudaMemcpyHostToDevice);
}


// 设备侧 det 输入路径（全 GPU：CUDA 计算代价+拍卖分配+KF/OCM/状态更新，零拷贝对接 TRT）
void OCSort::update_dev(const float* det_xyxy, const float* det_scores, const int* det_cls, int det_num,
                        std::vector<Trk>& out,
                        int img_w, int img_h,
                        cudaStream_t stream){
    // 直接走 GPU 代价 + CUDA Auction 路径，避免 Host 回拷（tracks 仍在 CPU，短期小量数据回读）
    (void)img_w; (void)img_h; out.clear();

    // 全 GPU 模式：设备端维护轨迹计数与ID
    ensure_capacity(det_num, std::max(last_T_, 1), stream);
    if (!dev_counter_inited_){
        if (!db_.dev_trk_count) cudaMalloc(&db_.dev_trk_count, sizeof(int));
        if (!db_.dev_next_id)   cudaMalloc(&db_.dev_next_id,   sizeof(int));
        int zero=0, one=1;
        cudaMemcpyAsync(db_.dev_trk_count,&zero,sizeof(int),cudaMemcpyHostToDevice,stream);
        cudaMemcpyAsync(db_.dev_next_id,&one,sizeof(int),cudaMemcpyHostToDevice,stream);
        // no immediate stream sync needed here; subsequent uses are on the same stream
        dev_counter_inited_=true;
    }
    int T = 0; cudaMemcpyAsync(&T, db_.dev_trk_count, sizeof(int), cudaMemcpyDeviceToHost, stream); cudaStreamSynchronize(stream);
    const int D = det_num;
    // 若当前没有轨迹且存在检测，直接在设备端创建所有轨迹（注意：ensure_capacity 会迁移已有状态，不会清空）
    if (T==0 && D>0){
        // Ensure capacity can hold all new tracks on first frame
        ensure_capacity(D, D, stream);
        std::vector<int> h_idx(D); for (int i=0;i<D;++i) h_idx[i]=i;
        cudaMemcpyAsync(db_.unused_det_idx, h_idx.data(), D*sizeof(int), cudaMemcpyHostToDevice, stream);
        ocs::append_new_tracks(det_xyxy, det_scores, det_cls, db_.unused_det_idx, D, cap_M_, db_.dev_trk_count, db_.dev_next_id,
                          db_.kf_x, db_.kf_P, db_.meta_id, db_.meta_cls, db_.meta_score, db_.meta_tsu, db_.meta_hit, db_.meta_obs,
                          db_.obs_prev, db_.obs_curr, db_.trk_cls, stream);
        cudaMemcpyAsync(&T, db_.dev_trk_count, sizeof(int), cudaMemcpyDeviceToHost, stream); cudaStreamSynchronize(stream);
        last_T_ = T; kf_inited_ = (T>0);
    }

    // KF predict on device with persistent state
    if (T>0){
        ensure_capacity(D, T, stream);
        // 直接对当前 T 个轨迹做 KF 预测（状态常驻于设备）
        ocs::kf_predict(db_.kf_x, db_.kf_P, T, kf_dt_, q_pos_, q_size_, stream);
        // tsu aging：无论是否有检测，本帧轨迹均老化一次
        ocs::age_tsu_begin(T, db_.meta_tsu, stream);
    }

    if (T>0 && D>0){
        // 设备缓冲（池化）
        ensure_capacity(D, T, stream);
        // 基于上一帧中心计算 OCM 动量与有效性（设备）
        // 确保 trk_valid 在本帧之前被清零（当 OCM 关闭或首帧），随后再计算 OCM 动量与有效性
        if (T>0) ocs::memset_float(db_.trk_valid, 0.0f, T, stream);
        if (ocm_) ocs::ocm_compute_momentum(db_.obs_prev, db_.obs_curr, T, 1e-3f, db_.trk_vdc, db_.trk_valid, stream);
        // 从 KF 状态生成预测框用于匹配
        ocs::kfx_to_boxes(db_.kf_x, T, reinterpret_cast<ocs::Box*>(db_.trk_boxes), stream);

        // 计算代价
        ocs::Assoc afn = (ocs::Assoc)afn_;
        // 类别掩码：直接使用设备常驻 meta_cls
        iou_cost_matrix(reinterpret_cast<const ocs::Box*>(det_xyxy), D, reinterpret_cast<const ocs::Box*>(db_.trk_boxes), T,
                        det_scores, db_.trk_valid, db_.trk_vdc, vdc_weight_, afn,
                        det_cls, reinterpret_cast<const int*>(db_.meta_cls), 1,
                        db_.iou, db_.ang, db_.cost, stream);

        // CUDA Auction（全设备化输出，使用池化缓冲）
        ocs::auction_assign_dev_pooled(db_.cost, D, T, db_.pairs, db_.k,
                                       db_.auc_price, db_.auc_owner, db_.auc_det_match, db_.auc_best_j, db_.auc_bids, db_.auc_colwin,
                                       stream, auction_rounds_);

        // 设备侧清零 det_used 与 sel_det
        ocs::memset_int(db_.det_used, 0, D, stream);
        ocs::memset_int(db_.sel_det, -1, T, stream);
        // 主匹配选择（tsu aging 已在预测后完成）
        ocs::primary_select_from_pairs(db_.iou, D, T, db_.pairs, db_.k, det_cls, reinterpret_cast<const int*>(db_.meta_cls), db_.meta_tsu, iou_th_, recover_relax_, ocr_?1:0, ocr_slope_, db_.det_used, db_.sel_det, stream);
        // 在设备端直接根据 sel_det 构建 KF z/m 并更新 meta（obs/score/cls）；不在此处回读 sel_det
        // 若保留 trk_cls: 在应用 meta 后将 trk_cls 同步为 meta_cls，避免类别漂移
        int actM = T;
        float oru_alpha = oru_ ? oru_alpha_ : 0.0f;
        ocs::build_z_from_sel(det_xyxy, D, db_.sel_det, actM, db_.obs_prev, oru_alpha, db_.kf_z, db_.kf_m, stream);
        // 直接用设备端选择结果更新 meta（obs/score/cls）
        ocs::apply_sel_to_meta(det_xyxy, det_scores, det_cls, D, db_.sel_det, T, db_.meta_obs, db_.meta_score, db_.meta_cls, stream);
        // 同步 trk_cls <- meta_cls，防止类别漂移
        cudaMemcpyAsync(db_.trk_cls, db_.meta_cls, T*sizeof(int), cudaMemcpyDeviceToDevice, stream);
        // KF update on device + OCM centers更新
        if (T>0){
            ocs::kf_update(db_.kf_x, db_.kf_P, db_.kf_z, db_.kf_m, actM, r_pos_, r_size_, stream);
            // 命中统计：匹配上的轨迹 tsu=0，hit+=1
            ocs::update_hit_tsu_after_match(db_.sel_det, T, db_.meta_hit, db_.meta_tsu, stream);
            // 更新 OCM centers（obs_prev <- obs_curr, obs_curr <- meta_obs centers）
            cudaMemcpyAsync(db_.obs_prev, db_.obs_curr, T*sizeof(float2), cudaMemcpyDeviceToDevice, stream);
            ocs::meta_obs_to_centers(db_.meta_obs, T, db_.obs_curr, stream);
            if (ocm_) ocs::ocm_compute_momentum(db_.obs_prev, db_.obs_curr, T, 1e-3f, db_.trk_vdc, db_.trk_valid, stream);
            else ocs::memset_float(db_.trk_valid, 0.0f, T, stream);
        }
        // 已使用的 det 完全在设备侧通过 det_used 维护；无需回读/host 标记

        // BYTE 二阶段匹配（全设备化）
        if (use_byte_) {
            // 设备侧根据 sel_det 标记 trk_matched
            ocs::memset_int(db_.trk_matched, 0, T, stream);
            ocs::sel_to_trk_matched(db_.sel_det, T, db_.trk_matched, stream);
            // 选择未匹配轨迹的最佳 det（原子占用），进一步占用 det_used
            ocs::byte_select(db_.iou, D, T, byte_iou_th_,
                              det_cls, reinterpret_cast<const int*>(db_.meta_cls),
                              db_.det_used, db_.trk_matched, db_.sel_det, stream);
            // 由选中的 det 构建 z/m；应用 ORU；更新 KF
            float oru_alpha = oru_ ? oru_alpha_ : 0.0f;
            ocs::build_z_from_sel(det_xyxy, D, db_.sel_det, actM, db_.obs_prev, oru_alpha, db_.kf_z, db_.kf_m, stream);
            ocs::kf_update(db_.kf_x, db_.kf_P, db_.kf_z, db_.kf_m, actM, r_pos_, r_size_, stream);
            // 命中统计：BYTE 二阶段新增匹配同样归零 tsu 并累计 hit
            ocs::update_hit_tsu_after_match(db_.sel_det, T, db_.meta_hit, db_.meta_tsu, stream);
            // 将二阶段匹配写入 meta，并更新 OCM 中心/动量（同步 trk_cls <- meta_cls 防止类别漂移）
            ocs::apply_sel_to_meta(det_xyxy, det_scores, det_cls, D, db_.sel_det, T, db_.meta_obs, db_.meta_score, db_.meta_cls, stream);
            // 同步 trk_cls <- meta_cls，防止类别漂移
            cudaMemcpyAsync(db_.trk_cls, db_.meta_cls, T*sizeof(int), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(db_.obs_prev, db_.obs_curr, T*sizeof(float2), cudaMemcpyDeviceToDevice, stream);
            ocs::meta_obs_to_centers(db_.meta_obs, T, db_.obs_curr, stream);
            if (ocm_) ocs::ocm_compute_momentum(db_.obs_prev, db_.obs_curr, T, 1e-3f, db_.trk_vdc, db_.trk_valid, stream);
        }

        // 新增 tracks（仅对未使用的 det），设备端 gather 后一次性回读紧凑数组，减少散拷
        ocs::build_unused_det_list(db_.det_used, D, db_.unused_det_idx, db_.unused_det_cnt, stream);
        int h_unused_cnt=0; cudaMemcpyAsync(&h_unused_cnt, db_.unused_det_cnt, sizeof(int), cudaMemcpyDeviceToHost, stream); cudaStreamSynchronize(stream);
        if (h_unused_cnt>0){
            // ensure capacity for existing tracks + new tracks
            ensure_capacity(D, T + h_unused_cnt, stream);
            ocs::append_new_tracks(det_xyxy, det_scores, det_cls, db_.unused_det_idx, h_unused_cnt, cap_M_, db_.dev_trk_count, db_.dev_next_id,
                              db_.kf_x, db_.kf_P, db_.meta_id, db_.meta_cls, db_.meta_score, db_.meta_tsu, db_.meta_hit, db_.meta_obs,
                              db_.obs_prev, db_.obs_curr, db_.trk_cls, stream);
            cudaMemcpyAsync(&T, db_.dev_trk_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            // sync done
        }

    }
    // 输出：设备端构建紧凑输出并一次性回读（全GPU meta）
    out.clear(); out.reserve(T);
    // 设备端构建输出（直接使用设备常驻 meta），一次性回读
    if (T>0){
        ocs::build_outputs_packed(db_.kf_x,
                                  reinterpret_cast<ocs::OcsPackedTrk*>(db_.out_packed),
                                  db_.meta_id,
                                  db_.meta_cls,
                                  db_.meta_score,
                                  db_.meta_tsu,
                                  db_.meta_hit,
                                  db_.meta_obs,
                                  min_hits_,
                                  T,
                                  stream);
        std::vector<ocs::OcsPackedTrk> h(T);
        cudaMemcpyAsync(h.data(), db_.out_packed, T*sizeof(ocs::OcsPackedTrk), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (int j=0;j<T;++j){ const auto& p=h[j]; if (!p.valid) continue; Trk r; auto tb=p.trk; auto ob=p.obs; r.x1=tb.x; r.y1=tb.y; r.x2=tb.z; r.y2=tb.w; r.ox1=ob.x; r.oy1=ob.y; r.ox2=ob.z; r.oy2=ob.w; r.score=p.score; r.cls=p.cls; r.id=p.id; r.valid=true; out.push_back(r);} 
    }
    // 设备端紧凑搬移：移除超龄轨迹并写回计数
    ocs::compact_tracks(T, max_age_,
                        db_.kf_x, db_.kf_P,
                        db_.meta_id, db_.meta_cls, db_.meta_score, db_.meta_tsu, db_.meta_hit,
                        db_.meta_obs,
                        db_.obs_prev, db_.obs_curr,
                        db_.trk_cls,
                        db_.tmp_kf_x, db_.tmp_kf_P,
                        db_.tmp_meta_id, db_.tmp_meta_cls, db_.tmp_meta_score, db_.tmp_meta_tsu, db_.tmp_meta_hit,
                        db_.tmp_meta_obs,
                        db_.tmp_obs_prev, db_.tmp_obs_curr,
                        db_.tmp_trk_cls,
                        db_.comp_newT,
                        stream);
    int newT=0; cudaMemcpyAsync(&newT, db_.comp_newT, sizeof(int), cudaMemcpyDeviceToHost, stream); cudaStreamSynchronize(stream);
    ocs::swap_compacted_back(db_.kf_x, db_.kf_P,
                             db_.meta_id, db_.meta_cls, db_.meta_score, db_.meta_tsu, db_.meta_hit,
                             db_.meta_obs,
                             db_.obs_prev, db_.obs_curr,
                             db_.trk_cls,
                             db_.tmp_kf_x, db_.tmp_kf_P,
                             db_.tmp_meta_id, db_.tmp_meta_cls, db_.tmp_meta_score, db_.tmp_meta_tsu, db_.tmp_meta_hit,
                             db_.tmp_meta_obs,
                             db_.tmp_obs_prev, db_.tmp_obs_curr,
                             db_.tmp_trk_cls);
    last_T_ = newT; kf_inited_ = (newT>0);
    cudaMemcpyAsync(db_.dev_trk_count, &newT, sizeof(int), cudaMemcpyHostToDevice, stream);

}
