#include "ocs_assign.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <stdint.h>

namespace ocs {

// CUDA Auction 实现（并行回合式）。
// 说明：为兼顾实现复杂度与性能，本实现采用回合式：
//  1) det 并行计算基于当前 price 的 best/second best 与 bid
//  2) 以列为单位用原子最大选择当前回合的赢家（按 bid 最大）
//  3) 应用赢家：更新 owner 与 price；上一拥有者释放
//  多回合迭代直到稳定或到达最大回合。

static __device__ inline unsigned long long pack_bid_det(float bid, int det){
    // 将浮点 bid 映射为可比较的无符号整数，保证 bid 大者更大
    // 使用 IEEE754 排序：将负值转换，但这里 bid >= 0，直接按位即可
    unsigned int bid_bits = __float_as_uint(bid);
    return (static_cast<unsigned long long>(bid_bits) << 32) | static_cast<unsigned int>(det);
}

#if 0
static __device__ inline float unpack_bid(unsigned long long p){
    unsigned int b = static_cast<unsigned int>(p >> 32);
    return __uint_as_float(b);
}
#endif
static __device__ inline int unpack_det(unsigned long long p){
    return static_cast<int>(p & 0xFFFFFFFFu);
}

__global__ void k_init_price_owner(float* price, int* owner, int M){
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j < M){ price[j] = 0.f; owner[j] = -1; }
}

__global__ void k_init_detmatch(int* det_match, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N){ det_match[i] = -1; }
}

__global__ void k_compute_bids(const float* cost, int N, int M,
                               const float* price,
                               const int* det_match,
                               int* best_j, float* bids){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (det_match[i] >= 0) { best_j[i] = -1; bids[i] = 0.f; return; }
    float best_v = -1e30f, second_v = -1e30f; int best_col = -1;
    // v = -cost - price
    const float* row = cost + i*M;
    for (int j=0;j<M;++j){
        float v = -row[j] - price[j];
        if (v > best_v){ second_v = best_v; best_v = v; best_col = j; }
        else if (v > second_v){ second_v = v; }
    }
    float eps = 1e-3f;
    float bid = (best_v - second_v) + eps;
    best_j[i] = best_col; bids[i] = bid;
}

__global__ void k_clear_colwin(unsigned long long* colwin, int M){
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j < M) colwin[j] = 0ull; // 0 表示无竞标
}

__global__ void k_vote_columns(const int* best_j, const float* bids, int N, unsigned long long* colwin){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    int j = best_j[i]; if (j < 0) return;
    unsigned long long p = pack_bid_det(bids[i], i);
    atomicMax((unsigned long long*)&colwin[j], p);
}

__global__ void k_apply_winners(const unsigned long long* colwin, int M,
                               float* price, int* owner, int* det_match,
                               const float* bids,
                               int* changed){
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= M) return;
    unsigned long long w = colwin[j];
    if (w == 0ull) return; // no bidder
    int det = unpack_det(w);
    float bid = bids[det];
    // 释放旧 owner（若存在）
    int old = atomicExch(&owner[j], det);
    if (old >= 0) det_match[old] = -1;
    det_match[det] = j;
    atomicAdd(&price[j], bid);
    if (changed) atomicExch(changed, 1);
}

__global__ void k_count_unmatched(const int* det_match, int N, int* out_cnt){
    __shared__ int sm[256];
    int tid = threadIdx.x; sm[tid] = 0; __syncthreads();
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int v = 0; if (i < N && det_match[i] < 0) v = 1;
    sm[tid] = v; __syncthreads();
    // reduce
    for (int s=blockDim.x/2; s>0; s>>=1){ if (tid < s) sm[tid] += sm[tid+s]; __syncthreads(); }
    if (tid==0) atomicAdd(out_cnt, sm[0]);
}

void auction_assign(const float* cost, int N, int M,
                    int* out_pairs_host, int* out_k_host,
                    cudaStream_t stream,
                    int max_rounds){
    if (N<=0 || M<=0){ if (out_k_host) *out_k_host = 0; return; }
    // device buffers
    float *d_price=nullptr; int *d_owner=nullptr; int *d_det_match=nullptr; int *d_best_j=nullptr; float *d_bids=nullptr; unsigned long long *d_colwin=nullptr;
    cudaMalloc(&d_price, M*sizeof(float));
    cudaMalloc(&d_owner, M*sizeof(int));
    cudaMalloc(&d_det_match, N*sizeof(int));
    cudaMalloc(&d_best_j, N*sizeof(int));
    cudaMalloc(&d_bids, N*sizeof(float));
    cudaMalloc(&d_colwin, M*sizeof(unsigned long long));

    int tb = 256;
    int gb_det = (N + tb - 1)/tb;
    int gb_trk = (M + tb - 1)/tb;

    k_init_price_owner<<<gb_trk, tb, 0, stream>>>(d_price, d_owner, M);
    k_init_detmatch<<<gb_det, tb, 0, stream>>>(d_det_match, N);

    const int rounds = (max_rounds>0? max_rounds : 64); // 可配置回合上限
    int* d_changed=nullptr; cudaMalloc(&d_changed, sizeof(int));
    for (int r=0; r<rounds; ++r){
        // 计算 bid
        k_compute_bids<<<gb_det, tb, 0, stream>>>(cost, N, M, d_price, d_det_match, d_best_j, d_bids);
        // 列赢家
        k_clear_colwin<<<gb_trk, tb, 0, stream>>>(d_colwin, M);
        k_vote_columns<<<gb_det, tb, 0, stream>>>(d_best_j, d_bids, N, d_colwin);
        // 应用赢家
        cudaMemsetAsync(d_changed, 0, sizeof(int), stream);
        k_apply_winners<<<gb_trk, tb, 0, stream>>>(d_colwin, M, d_price, d_owner, d_det_match, d_bids, d_changed);
        // 提前收敛：若本轮无变化，则停止
        int hchg=0; cudaMemcpyAsync(&hchg, d_changed, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (hchg==0) break;
    }
    cudaFree(d_changed);

    // 回读 det_match 生成配对
    std::vector<int> h_match(N);
    cudaMemcpyAsync(h_match.data(), d_det_match, N*sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int k=0;
    if (out_pairs_host) {
        for (int i=0;i<N;++i){ int j = h_match[i]; if (j>=0){ out_pairs_host[2*k]=i; out_pairs_host[2*k+1]=j; ++k; } }
    } else {
        for (int i=0;i<N;++i){ int j = h_match[i]; if (j>=0){ ++k; } }
    }
    if (out_k_host) *out_k_host = k;

    cudaFree(d_price); cudaFree(d_owner); cudaFree(d_det_match); cudaFree(d_best_j); cudaFree(d_bids); cudaFree(d_colwin);
}

void auction_assign_dev(const float* cost, int N, int M,
                        int* out_pairs_dev, int* out_k_dev,
                        cudaStream_t stream,
                        int max_rounds){
    // fallback to pooled variant with local temporaries (kept for compatibility)
    float *d_price=nullptr; int *d_owner=nullptr; int *d_det_match=nullptr; int *d_best_j=nullptr; float *d_bids=nullptr; unsigned long long *d_colwin=nullptr;
    if (N<=0 || M<=0){ if (out_k_dev) cudaMemsetAsync(out_k_dev, 0, sizeof(int), stream); return; }
    cudaMalloc(&d_price, M*sizeof(float));
    cudaMalloc(&d_owner, M*sizeof(int));
    cudaMalloc(&d_det_match, N*sizeof(int));
    cudaMalloc(&d_best_j, N*sizeof(int));
    cudaMalloc(&d_bids, N*sizeof(float));
    cudaMalloc(&d_colwin, M*sizeof(unsigned long long));
    // call pooled core
    auction_assign_dev_pooled(cost, N, M, out_pairs_dev, out_k_dev, d_price, d_owner, d_det_match, d_best_j, d_bids, d_colwin, stream, max_rounds);
    cudaFree(d_price); cudaFree(d_owner); cudaFree(d_det_match); cudaFree(d_best_j); cudaFree(d_bids); cudaFree(d_colwin);
}

void auction_assign_dev_pooled(const float* cost, int N, int M,
                               int* out_pairs_dev, int* out_k_dev,
                               float* d_price, int* d_owner, int* d_det_match, int* d_best_j, float* d_bids, unsigned long long* d_colwin,
                               cudaStream_t stream,
                               int max_rounds){
    if (N<=0 || M<=0){ if (out_k_dev) cudaMemsetAsync(out_k_dev, 0, sizeof(int), stream); return; }
    // using pooled device buffers provided by caller

    int tb = 256; int gb_det=(N+tb-1)/tb; int gb_trk=(M+tb-1)/tb;
    k_init_price_owner<<<gb_trk, tb, 0, stream>>>(d_price, d_owner, M);
    k_init_detmatch<<<gb_det, tb, 0, stream>>>(d_det_match, N);
    const int rounds = (max_rounds>0? max_rounds : 64);
    for (int r=0; r<rounds; ++r){
        k_compute_bids<<<gb_det, tb, 0, stream>>>(cost, N, M, d_price, d_det_match, d_best_j, d_bids);
        k_clear_colwin<<<gb_trk, tb, 0, stream>>>(d_colwin, M);
        k_vote_columns<<<gb_det, tb, 0, stream>>>(d_best_j, d_bids, N, d_colwin);
        k_apply_winners<<<gb_trk, tb, 0, stream>>>(d_colwin, M, d_price, d_owner, d_det_match, d_bids, nullptr);
    }
    // out_k_dev
    if (out_k_dev){ cudaMemsetAsync(out_k_dev, 0, sizeof(int), stream); }
    // write out_pairs_dev using det_match
    // 方案：为避免对 extended-lambda 的依赖与编译器差异，这里采用一次 D2H 将 det_match 读回主机，
    // 在主机侧构建紧凑的 (det, trk) 配对数组和配对数量，然后再将结果回写到设备侧 out_pairs_dev / out_k_dev。
    // 该阶段数据量极小，相比 IOU/Auction 等核函数的主开销可忽略，对整体延迟影响很小。
    if (out_pairs_dev && out_k_dev){
        // 读取 det_match 到 host
        std::vector<int> h_match(N);
        cudaMemcpyAsync(h_match.data(), d_det_match, N*sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int kcount=0;
        for (int i=0;i<N;++i){ int j=h_match[i]; if(j>=0){
            cudaMemcpyAsync(out_pairs_dev+2*kcount, &i, sizeof(int), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(out_pairs_dev+2*kcount+1, &j, sizeof(int), cudaMemcpyHostToDevice, stream);
            ++kcount;
        }}
        cudaMemcpyAsync(out_k_dev, &kcount, sizeof(int), cudaMemcpyHostToDevice, stream);
    }
}

} // namespace ocs
