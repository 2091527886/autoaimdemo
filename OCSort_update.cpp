#include "OCSort.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include "TensorRT-YOLO/modules/trtyolo/utils/common.hpp"
using trtyolo::checkCudaError;
#include "ocs_iou.cuh"
#include "ocs_assign.cuh"
#include "ocs_kf.cuh"
#include "ocs_ocm.cuh"
#include "ocs_byte.cuh"
#include "ocs_out.cuh"
#include "ocs_state.cuh"

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
    int T = 0; cudaMemcpyAsync(&T, db_.dev_trk_count, sizeof(int), cudaMemcpyDeviceToHost, stream); cudaStreamSynchronize(stream); if (T < 0) T = 0; if (T > cap_M_) T = cap_M_;
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
        CHECK(cudaPeekAtLastError());
        // tsu aging：无论是否有检测，本帧轨迹均老化一次
        ocs::age_tsu_begin(T, db_.meta_tsu, stream);
        CHECK(cudaPeekAtLastError());
    }

    if (T>0 && D>0){
        // 设备缓冲（池化）
        ensure_capacity(D, T, stream);
        // 基于上一帧中心计算 OCM 动量与有效性（设备）
        // 确保 trk_valid 在本帧之前被清零（当 OCM 关闭或首帧），随后再计算 OCM 动量与有效性
        if (T>0) ocs::memset_float(db_.trk_valid, 0.0f, T, stream);
        if (ocm_) ocs::ocm_compute_momentum(db_.obs_prev, db_.obs_curr, T, 1e-3f, db_.trk_vdc, db_.trk_valid, stream);
       CHECK(cudaPeekAtLastError());
        // 从 KF 状态生成预测框用于匹配
        ocs::kfx_to_boxes(db_.kf_x, T, reinterpret_cast<ocs::Box*>(db_.trk_boxes), stream);
        CHECK(cudaPeekAtLastError());

        // 计算代价
        ocs::Assoc afn = (ocs::Assoc)afn_;
        // 类别掩码：直接使用设备常驻 meta_cls
        iou_cost_matrix(reinterpret_cast<const ocs::Box*>(det_xyxy), D, reinterpret_cast<const ocs::Box*>(db_.trk_boxes), T,
                        det_scores, db_.trk_valid, db_.trk_vdc, vdc_weight_, afn,
                        det_cls, reinterpret_cast<const int*>(db_.meta_cls), 1,
                        db_.iou, db_.ang, db_.cost, stream);
         CHECK(cudaPeekAtLastError());

        // CUDA Auction（全设备化输出，使用池化缓冲）
        ocs::auction_assign_dev_pooled(db_.cost, D, T, db_.pairs, db_.k,
                                       db_.auc_price, db_.auc_owner, db_.auc_det_match, db_.auc_best_j, db_.auc_bids, db_.auc_colwin,
                                       stream, auction_rounds_);
         CHECK(cudaPeekAtLastError());

        // 设备侧清零 det_used 与 sel_det
        ocs::memset_int(db_.det_used, 0, D, stream);
         CHECK(cudaPeekAtLastError());
        ocs::memset_int(db_.sel_det, -1, T, stream);
         CHECK(cudaPeekAtLastError());
        // 主匹配选择（tsu aging 已在预测后完成）
        ocs::primary_select_from_pairs(db_.iou, D, T, db_.pairs, db_.k, det_cls, reinterpret_cast<const int*>(db_.meta_cls), db_.meta_tsu, iou_th_, recover_relax_, ocr_?1:0, ocr_slope_, db_.det_used, db_.sel_det, stream);
         CHECK(cudaPeekAtLastError());
        // 在设备端直接根据 sel_det 构建 KF z/m 并更新 meta（obs/score/cls）；不在此处回读 sel_det
        // 若保留 trk_cls: 在应用 meta 后将 trk_cls 同步为 meta_cls，避免类别漂移
        int actM = T;
        float oru_alpha = oru_ ? oru_alpha_ : 0.0f;
        ocs::build_z_from_sel(det_xyxy, D, db_.sel_det, actM, db_.obs_prev, oru_alpha, db_.kf_z, db_.kf_m, stream);
        CHECK(cudaPeekAtLastError());
        // 直接用设备端选择结果更新 meta（obs/score/cls）
        ocs::apply_sel_to_meta(det_xyxy, det_scores, det_cls, D, db_.sel_det, T, db_.meta_obs, db_.meta_score, db_.meta_cls, stream);
        CHECK(cudaPeekAtLastError());
        // 同步 trk_cls <- meta_cls，防止类别漂移
        cudaMemcpyAsync(db_.trk_cls, db_.meta_cls, T*sizeof(int), cudaMemcpyDeviceToDevice, stream);
        // KF update on device + OCM centers更新
        if (T>0){
            ocs::kf_update(db_.kf_x, db_.kf_P, db_.kf_z, db_.kf_m, actM, r_pos_, r_size_, stream);
            CHECK(cudaPeekAtLastError());
            // 命中统计：匹配上的轨迹 tsu=0，hit+=1
            ocs::update_hit_tsu_after_match(db_.sel_det, T, db_.meta_hit, db_.meta_tsu, stream);
            CHECK(cudaPeekAtLastError());
            // 更新 OCM centers（obs_prev <- obs_curr, obs_curr <- meta_obs centers）
            cudaMemcpyAsync(db_.obs_prev, db_.obs_curr, T*sizeof(float2), cudaMemcpyDeviceToDevice, stream);
            ocs::meta_obs_to_centers(db_.meta_obs, T, db_.obs_curr, stream);
            CHECK(cudaPeekAtLastError());
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
            CHECK(cudaPeekAtLastError());
            // 由选中的 det 构建 z/m；应用 ORU；更新 KF
            oru_alpha = oru_ ? oru_alpha_ : 0.0f;
            ocs::build_z_from_sel(det_xyxy, D, db_.sel_det, actM, db_.obs_prev, oru_alpha, db_.kf_z, db_.kf_m, stream);
             CHECK(cudaPeekAtLastError());
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
