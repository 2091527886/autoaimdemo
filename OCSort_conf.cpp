#include "OCSort.hpp"
#include "Config.hpp"
#include <cuda_runtime.h>

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
