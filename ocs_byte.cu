#include "ocs_byte.cuh"

namespace ocs {

__global__ void k_zero_int(int* p, int v, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=v; }
void memset_int(int* ptr, int value, int count, cudaStream_t stream){ int tb=256, gb=(count+tb-1)/tb; k_zero_int<<<gb,tb,0,stream>>>(ptr,value,count); }

__global__ void k_zero_float(float* p, float v, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=v; }
void memset_float(float* ptr, float value, int count, cudaStream_t stream){ int tb=256, gb=(count+tb-1)/tb; k_zero_float<<<gb,tb,0,stream>>>(ptr,value,count); }

__global__ void k_byte_select(const float* iou, int N, int M, float th,
                              const int* det_cls, const int* trk_cls,
                              int* det_used, int* trk_matched, int* sel_det){
    int tj = blockIdx.x*blockDim.x + threadIdx.x; if (tj>=M) return;
    if (trk_matched[tj]) { sel_det[tj] = -1; return; }
    float best = th; int best_di = -1;
    for (int di=0; di<N; ++di){
        if (det_used[di]) continue;
        // 类别一致性约束：若类别数组提供且不一致，则跳过
        if (det_cls && trk_cls){ int dc = det_cls[di]; int tc = trk_cls[tj]; if (dc>=0 && tc>=0 && dc!=tc) continue; }
        float u = iou[di*M + tj];
        if (u>best){ best=u; best_di=di; }
    }
    if (best_di>=0) {
        // 占用该 det（乐观原子，若竞争失败则该 tj 选中失败）
        int old = atomicCAS(&det_used[best_di], 0, 1);
        if (old==0){ sel_det[tj] = best_di; trk_matched[tj] = 1; }
        else sel_det[tj] = -1;
    } else sel_det[tj] = -1;
}

void byte_select(const float* iou, int N, int M, float th,
                 const int* det_cls, const int* trk_cls,
                 int* det_used, int* trk_matched, int* sel_det,
                 cudaStream_t stream){
    int tb=256, gb=(M+tb-1)/tb; k_byte_select<<<gb,tb,0,stream>>>(iou,N,M,th,det_cls,trk_cls,det_used,trk_matched,sel_det);
}

__global__ void k_primary_select(const float* iou, int N, int M,
                                   const int* pairs, const int* k_count,
                                   const int* det_cls, const int* trk_cls,
                                   const int* tsu,
                                   float base_th, float recover_relax, int ocr_enable, float ocr_slope,
                                   int* det_used, int* sel_det){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= *k_count) return;
    int di = pairs[2*idx+0];
    int tj = pairs[2*idx+1];
    if (di<0 || di>=N || tj<0 || tj>=M) return;
    float th = base_th;
    if (ocr_enable && tsu){ int t = tsu[tj]; float dyn = 1.0f - ocr_slope * (float)t; float rel = fminf(1.0f, fmaxf(0.5f, dyn)); th = base_th * fmaxf(recover_relax, rel); }
    float u = iou[di*M + tj];
    if (u < th) return;
    if (det_cls && trk_cls){ int dc = det_cls[di], tc = trk_cls[tj]; if (dc>=0 && tc>=0 && dc!=tc) return; }
    // occupy det; if successful and sel_det empty, set
    if (atomicCAS(det_used + di, 0, 1) == 0){ atomicCAS(sel_det + tj, -1, di); }
}

__global__ void k_build_z(const float* det_xyxy, int N, const int* sel_det, int M,
                          const float2* prev, float oru_alpha,
                          float* z, int* m){
    int tj = blockIdx.x*blockDim.x + threadIdx.x; if (tj>=M) return;
    int di = sel_det[tj]; if (di<0 || di>=N){ m[tj]=0; return; }
    const float* d = det_xyxy + di*4;
    float cx = 0.5f*(d[0]+d[2]);
    float cy = 0.5f*(d[1]+d[3]);
    float s  = (d[2]-d[0])*(d[3]-d[1]);
    float r  = (d[2]-d[0]) / ((d[3]-d[1])+1e-3f);
    if (oru_alpha>0){ float pcx=prev[tj].y, pcy=prev[tj].x; cx = oru_alpha*cx + (1-oru_alpha)*pcx; cy = oru_alpha*cy + (1-oru_alpha)*pcy; }
    z[tj*4+0]=cx; z[tj*4+1]=cy; z[tj*4+2]=s; z[tj*4+3]=r; m[tj]=1;
}

void primary_select_from_pairs(const float* iou, int N, int M,
                              const int* pairs, const int* k_count,
                              const int* det_cls, const int* trk_cls,
                              const int* tsu,
                              float base_th, float recover_relax, int ocr_enable, float ocr_slope,
                              int* det_used, int* sel_det,
                              cudaStream_t stream){
    int tb=256;
    // 读取 k_count 到 host，按 K 启动更紧凑的网格（微优化）。
    int Khost = 0; cudaMemcpyAsync(&Khost, k_count, sizeof(int), cudaMemcpyDeviceToHost, stream); cudaStreamSynchronize(stream);
    int gb = (Khost + tb - 1) / tb; if (gb <= 0) return;
    k_primary_select<<<gb,tb,0,stream>>>(iou,N,M,pairs,k_count,det_cls,trk_cls,tsu,base_th,recover_relax,ocr_enable,ocr_slope,det_used,sel_det);
}

void build_z_from_sel(const float* det_xyxy, int N,
                      const int* sel_det, int M,
                      const float2* prev_centers,
                      float oru_alpha,
                      float* z_out, int* m_out,
                      cudaStream_t stream){
    int tb=256, gb=(M+tb-1)/tb; k_build_z<<<gb,tb,0,stream>>>(det_xyxy,N,sel_det,M,prev_centers,oru_alpha,z_out,m_out);
}

__global__ void k_apply_sel_meta(const float* det_xyxy, const float* det_scores, const int* det_cls, int N,
                                 const int* sel_det, int M,
                                 float4* meta_obs, float* meta_score, int* meta_cls){
    int tj = blockIdx.x*blockDim.x + threadIdx.x; if (tj>=M) return;
    int di = sel_det[tj]; if (di<0 || di>=N) return;
    const float* d = det_xyxy + di*4;
    meta_obs[tj] = make_float4(d[0], d[1], d[2], d[3]);
    if (det_scores) meta_score[tj] = det_scores[di];
    if (det_cls)    meta_cls[tj]   = det_cls[di];
}

void apply_sel_to_meta(const float* det_xyxy, const float* det_scores, const int* det_cls, int N,
                      const int* sel_det, int M,
                      float4* meta_obs, float* meta_score, int* meta_cls,
                      cudaStream_t stream){
   int tb=256, gb=(M+tb-1)/tb; k_apply_sel_meta<<<gb,tb,0,stream>>>(det_xyxy,det_scores,det_cls,N,sel_det,M,meta_obs,meta_score,meta_cls);
}

__global__ void k_sel_to_trk(const int* sel_det, int M, int* trk_matched){ int tj=blockIdx.x*blockDim.x+threadIdx.x; if(tj>=M) return; trk_matched[tj] = (sel_det[tj] >= 0) ? 1 : 0; }
void sel_to_trk_matched(const int* sel_det, int M, int* trk_matched, cudaStream_t stream){ int tb=256, gb=(M+tb-1)/tb; k_sel_to_trk<<<gb,tb,0,stream>>>(sel_det,M,trk_matched);}
__global__ void k_meta_obs_to_centers(const float4* meta_obs, int M, float2* out_curr){
   int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=M) return; float4 b=meta_obs[j]; float cx=0.5f*(b.x+b.z); float cy=0.5f*(b.y+b.w); out_curr[j]=make_float2(cy,cx);
}
void meta_obs_to_centers(const float4* meta_obs, int M, float2* out_curr, cudaStream_t stream){ int tb=256, gb=(M+tb-1)/tb; k_meta_obs_to_centers<<<gb,tb,0,stream>>>(meta_obs,M,out_curr); }

__global__ void k_build_unused(const int* det_used, int N, int* out_indices, int* out_count){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return; if(det_used[i]==0){ int pos=atomicAdd(out_count,1); out_indices[pos]=i; } }
void build_unused_det_list(const int* det_used, int N, int* out_indices, int* out_count, cudaStream_t stream){ cudaMemsetAsync(out_count,0,sizeof(int),stream); int tb=256, gb=(N+tb-1)/tb; k_build_unused<<<gb,tb,0,stream>>>(det_used,N,out_indices,out_count); }

__global__ void k_gather(const float* det_xyxy, const float* det_scores, const int* det_cls, const int* indices, int count, float* out_xyxy4N, float* out_scores, int* out_cls){ int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=count) return; int di=indices[j]; const float* d=det_xyxy+di*4; out_xyxy4N[j*4+0]=d[0]; out_xyxy4N[j*4+1]=d[1]; out_xyxy4N[j*4+2]=d[2]; out_xyxy4N[j*4+3]=d[3]; if(out_scores&&det_scores) out_scores[j]=det_scores[di]; if(out_cls&&det_cls) out_cls[j]=det_cls[di]; }
void gather_dets_by_index(const float* det_xyxy, const float* det_scores, const int* det_cls, const int* indices, int count, float* out_xyxy4N, float* out_scores, int* out_cls, cudaStream_t stream){ int tb=256, gb=(count+tb-1)/tb; k_gather<<<gb,tb,0,stream>>>(det_xyxy,det_scores,det_cls,indices,count,out_xyxy4N,out_scores,out_cls);} 

} // namespace ocs
