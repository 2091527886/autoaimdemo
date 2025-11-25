#include "OCSort.hpp"
#include <cuda_runtime.h>

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
