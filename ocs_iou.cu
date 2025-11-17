#include "ocs_iou.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace ocs {

static __device__ __forceinline__ float iou1(const Box& a, const Box& b){
    float xx1 = fmaxf(a.x1, b.x1);
    float yy1 = fmaxf(a.y1, b.y1);
    float xx2 = fminf(a.x2, b.x2);
    float yy2 = fminf(a.y2, b.y2);
    float w = fmaxf(0.f, xx2 - xx1); float h = fmaxf(0.f, yy2 - yy1);
    float inter = w*h; float sa=(a.x2-a.x1)*(a.y2-a.y1); float sb=(b.x2-b.x1)*(b.y2-b.y1);
    float uni = sa + sb - inter + 1e-6f; return inter / uni;
}

static __device__ __forceinline__ float2 dir_det_to_trk(const Box& d, const Box& t){
    float cx1 = 0.5f*(d.x1+d.x2), cy1 = 0.5f*(d.y1+d.y2);
    float cx2 = 0.5f*(t.x1+t.x2), cy2 = 0.5f*(t.y1+t.y2);
    float dx = cx1-cx2, dy = cy1-cy2; float n = rsqrtf(dx*dx+dy*dy+1e-6f);
    return make_float2(dy*n, dx*n); // (Y, X)
}

static __device__ __forceinline__ float angle_score(const float2& inertiaYX, const float2& detYX){
    float c = inertiaYX.x*detYX.x + inertiaYX.y*detYX.y;
    c = fmaxf(-1.f, fminf(1.f, c));
    float a = acosf(c); // [0,pi]
    float v = (CUDART_PI_F*0.5f - fabsf(a)) / CUDART_PI_F; // [-0.5,0.5] -> shift to [0,1]
    return v + 0.5f;
}

__global__ void k_cost(const Box* dets,int N,const Box* trks,int M,
                       const float* det_scores,
                       const float* trk_prev_obs_valid,
                       const float2* vdc_dirs,
                       float vdc_weight, int ass_fn,
                       const int* det_cls, const int* trk_cls, int class_mask_enable,
                       float* out_iou,float* out_ang,float* out_cost){
    int di = blockIdx.y*blockDim.y + threadIdx.y;
    int ti = blockIdx.x*blockDim.x + threadIdx.x;
    if (di>=N || ti>=M) return;
    const Box d = dets[di]; const Box t = trks[ti];
    // class mask
    if (class_mask_enable && det_cls && trk_cls){
        int dc = det_cls[di]; int tc = trk_cls[ti];
        if (dc >= 0 && tc >= 0 && dc != tc){
            int idx = di*M + ti; out_iou[idx]=0.f; out_ang[idx]=0.f; out_cost[idx]=1e6f; return;
        }
    }
    float i = 0.f;
    // association function
    if (ass_fn==0){ // IOU
        i = iou1(d,t);
    } else if (ass_fn==4){ // CT_DIST -> map to [0,1] by normalized center distance
        float cx1 = 0.5f*(d.x1+d.x2), cy1 = 0.5f*(d.y1+d.y2);
        float cx2 = 0.5f*(t.x1+t.x2), cy2 = 0.5f*(t.y1+t.y2);
        float dx = cx1-cx2, dy = cy1-cy2;
        float ad = fabsf(dx)+fabsf(dy) + 1e-3f;
        float sw = fmaxf(fabsf(d.x2-d.x1), 1.f) + fmaxf(fabsf(t.x2-t.x1),1.f);
        float sh = fmaxf(fabsf(d.y2-d.y1), 1.f) + fmaxf(fabsf(t.y2-t.y1),1.f);
        float nd = (fabsf(dx)/sw + fabsf(dy)/sh) * 0.5f; // [0, inf)
        i = fmaxf(0.f, 1.0f - nd); // map to [0,1]
    } else if (ass_fn==1){ // GIOU
        float xx1=fmaxf(d.x1,t.x1), yy1=fmaxf(d.y1,t.y1), xx2=fminf(d.x2,t.x2), yy2=fminf(d.y2,t.y2);
        float w=fmaxf(0.f,xx2-xx1), h=fmaxf(0.f,yy2-yy1); float inter=w*h;
        float sa=(d.x2-d.x1)*(d.y2-d.y1), sb=(t.x2-t.x1)*(t.y2-t.y1);
        float uni=sa+sb-inter+1e-6f; float iou=inter/uni;
        float cx1=fminf(d.x1,t.x1), cy1=fminf(d.y1,t.y1), cx2=fmaxf(d.x2,t.x2), cy2=fmaxf(d.y2,t.y2);
        float cw=fmaxf(0.f,cx2-cx1), ch=fmaxf(0.f,cy2-cy1); float ca=cw*ch+1e-6f;
        float giou = iou - (ca-uni)/ca; i = giou;
    } else if (ass_fn==2){ // CIOU
        float iou = iou1(d,t);
        float dcx=0.5f*(d.x1+d.x2)-0.5f*(t.x1+t.x2); float dcy=0.5f*(d.y1+d.y2)-0.5f*(t.y1+t.y2);
        float cx1=fminf(d.x1,t.x1), cy1=fminf(d.y1,t.y1), cx2=fmaxf(d.x2,t.x2), cy2=fmaxf(d.y2,t.y2);
        float cw=fmaxf(0.f,cx2-cx1), ch=fmaxf(0.f,cy2-cy1); float c2=cw*cw+ch*ch+1e-6f;
        float u= (dcx*dcx + dcy*dcy)/c2;
        float w1=d.x2-d.x1, h1=d.y2-d.y1, w2=t.x2-t.x1, h2=t.y2-t.y1;
        float v = 4.0f/(CUDART_PI_F*CUDART_PI_F) * powf(atanf(w2/h2)-atanf(w1/h1),2);
        float alpha = v/(1 - iou + v + 1e-6f);
        i = iou - (u + alpha*v);
    } else if (ass_fn==3){ // DIOU
        float iou = iou1(d,t);
        float dcx=0.5f*(d.x1+d.x2)-0.5f*(t.x1+t.x2); float dcy=0.5f*(d.y1+d.y2)-0.5f*(t.y1+t.y2);
        float cx1=fminf(d.x1,t.x1), cy1=fminf(d.y1,t.y1), cx2=fmaxf(d.x2,t.x2), cy2=fmaxf(d.y2,t.y2);
        float cw=fmaxf(0.f,cx2-cx1), ch=fmaxf(0.f,cy2-cy1); float c2=cw*cw+ch*ch+1e-6f;
        float u= (dcx*dcx + dcy*dcy)/c2; i = iou - u;
    } else {
        i = iou1(d,t);
    }
    float ang = 0.f;
    if (trk_prev_obs_valid[ti] > 0.5f){
        float2 detyx = dir_det_to_trk(d,t);
        ang = angle_score(vdc_dirs[ti], detyx);
    }
    float s = det_scores[di];
    float cost = -(i + vdc_weight*ang) * s;
    int idx = di*M + ti;
    out_iou[idx] = i; out_ang[idx] = ang; out_cost[idx] = cost;
}

void iou_cost_matrix(const Box* dets, int N,
                     const Box* trks, int M,
                     const float* det_scores,
                     const float* trk_prev_obs_valid,
                     const float2* vdc_dirs,
                     float vdc_weight,
                     Assoc ass_fn,
                     const int* det_cls,
                     const int* trk_cls,
                     int class_mask_enable,
                     float* out_iou, float* out_angle, float* out_cost,
                     cudaStream_t stream){
    dim3 block(16,16); dim3 grid((M+block.x-1)/block.x, (N+block.y-1)/block.y);
    k_cost<<<grid,block,0,stream>>>(dets,N,trks,M,det_scores,trk_prev_obs_valid,vdc_dirs,vdc_weight,(int)ass_fn,det_cls,trk_cls,class_mask_enable,out_iou,out_angle,out_cost);
}

} // namespace ocs
