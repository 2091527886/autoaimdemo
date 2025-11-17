#pragma once
#include <vector>
// #include <opencv2/core.hpp> // not needed here
#include <cuda_runtime.h>

class Config;

class OCSort {
public:
    void set_dt(float dt) { kf_dt_ = dt > 0.0f ? dt : 1.0f; }

public:
    struct Det { float x1, y1, x2, y2; float score; int cls; };
    struct Trk {
        float x1, y1, x2, y2;      // track bbox (current state)
        float ox1, oy1, ox2, oy2;  // last matched observation bbox (obs)
        float score; int cls; int id; bool valid;
    };

    OCSort();
    ~OCSort();
    void configure(const Config& c);
    void set_kf_noise(float q_pos, float q_size, float r_pos, float r_size){ q_pos_=q_pos; q_size_=q_size; r_pos_=r_pos; r_size_=r_size; }
    void reset();

    // img_w/h kept for future use (normalization if needed)
    // GPU路径：直接接受设备侧指针（未来与 trtyolo 零拷贝对接）
    // 支持传入 cudaStream（与 trtyolo 串联），默认 0
    void update_dev(const float* det_xyxy, const float* det_scores, const int* det_cls, int det_num,
                    std::vector<Trk>& out,
                    int img_w, int img_h,
                    cudaStream_t stream = 0);

private:
    bool dev_counter_inited_ = false;


private:
    // 移除旧的 CPU 路径：全部状态驻留在设备端
    int next_id_ = 1;

    // device workspace (pooled)
    int cap_N_ = 0, cap_M_ = 0; // capacity for det N and trk M
    // cost & assign
    struct DevBuf {
        float *iou=nullptr, *ang=nullptr, *cost=nullptr;
        int   *pairs=nullptr, *k=nullptr; // device pairs (2*k) and count
        // auction temp buffers (pooled)
        float *auc_price=nullptr; // M
        int   *auc_owner=nullptr; // M
        int   *auc_det_match=nullptr; // N
        int   *auc_best_j=nullptr; // N
        float *auc_bids=nullptr; // N
        unsigned long long *auc_colwin=nullptr; // M
        // trk side attrs
        void *trk_boxes=nullptr; // ocs::Box*
        float *trk_valid=nullptr; float2 *trk_vdc=nullptr;
        int   *trk_cls=nullptr;
        // kf
        float *kf_x=nullptr, *kf_P=nullptr; // kf_x: cap_M_*7, kf_P: cap_M_*49
        float *kf_z=nullptr; // size cap_M_*4
        int   *kf_m=nullptr; // size cap_M_
        // OCM centers
        float2 *obs_prev=nullptr; // size cap_M_
        float2 *obs_curr=nullptr; // size cap_M_
        // BYTE
        int *det_used=nullptr; // size cap_N_
        int *trk_matched=nullptr; // size cap_M_
        int *sel_det=nullptr; // size cap_M_
        int *unused_det_idx=nullptr; // size cap_N_
        int *unused_det_cnt=nullptr; // scalar on device
        // outputs (device-compact)
        float4 *out_trk=nullptr; // M (x1,y1,x2,y2)
        float4 *out_obs=nullptr; // M (x1,y1,x2,y2)
        float  *out_score=nullptr; // M
        int    *out_cls=nullptr; // M
        int    *out_id=nullptr; // M
        int    *out_valid=nullptr; // M (1/0)
        // packed AoS buffer to reduce D2H copies
        void   *out_packed=nullptr; // ocs::OcsPackedTrk* size M
        // buffers for new tracks (gathered unused detections)
        float *new_xyxy=nullptr;   // N x 4
        float *new_scores=nullptr; // N
        int   *new_cls=nullptr;    // N
        // persistent device-side track state (meta)
        int    *meta_id=nullptr; // M
        int    *meta_cls=nullptr; // M
        float  *meta_score=nullptr; // M
        int    *meta_tsu=nullptr; // M (time_since_update)
        int    *meta_hit=nullptr; // M (hit_streak)
        float4 *meta_obs=nullptr; // M (obs box)
        // device counters for tracks
        int *dev_trk_count=nullptr; // scalar
        int *dev_next_id=nullptr;   // scalar
        // temp buffers for compaction
        float *tmp_kf_x=nullptr, *tmp_kf_P=nullptr;
        int   *tmp_meta_id=nullptr, *tmp_meta_cls=nullptr; float *tmp_meta_score=nullptr; int *tmp_meta_tsu=nullptr, *tmp_meta_hit=nullptr;
        float4 *tmp_meta_obs=nullptr; float2 *tmp_obs_prev=nullptr, *tmp_obs_curr=nullptr; int *tmp_trk_cls=nullptr;
        int *comp_newT=nullptr; // scalar
    } db_;

    // OCM centers device buffers
    
    void ensure_capacity(int N, int M, cudaStream_t stream);

    // persistent KF/OCM state trackers
    bool kf_inited_ = false;
    int  last_T_ = 0;
    // 设备端统一管理轨迹，不再维护主机侧 free_slots_


    // params
    float iou_th_ = 0.3f;
    int   max_age_ = 30;
    int   min_hits_ = 3;
    float vdc_weight_ = 0.0f;
    int   afn_ = 0; // 0 IOU, 1 GIOU, 2 CIOU, 3 DIOU, 4 CT
    float kf_dt_ = 1.0f;
    bool  oru_ = true;
    bool  ocm_ = true;
    bool  ocr_ = true;
    // KF noise params (defaults; later can be exposed)
    float q_pos_ = 0.01f, q_size_ = 0.01f;
    float r_pos_ = 1.0f,  r_size_ = 1.0f;
    // OCR/BYTE
    float recover_relax_ = 1.0f; // 1.0=no relax; <1.0 relax threshold
    bool  use_byte_ = false; float byte_iou_th_ = 0.15f;
    // tuning
    int   auction_rounds_ = 64;
    float ocr_slope_ = 0.05f;
    float oru_alpha_ = 0.85f; // ORU 融合系数，来自 Config
};
