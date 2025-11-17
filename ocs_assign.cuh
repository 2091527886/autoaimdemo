#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace ocs {

// Auction-based assignment on GPU (legacy host-output variant)
// cost: N*M (det-major). Output matched det->trk indices count K (K<=min(N,M)).
void auction_assign(const float* cost, int N, int M,
                    int* out_pairs_host, int* out_k_host,
                    cudaStream_t stream,
                    int max_rounds);

// Pure device-output variant (recommended):
// out_pairs_dev: device int buffer with capacity N*2
// out_k_dev: device int scalar
void auction_assign_dev(const float* cost, int N, int M,
                        int* out_pairs_dev, int* out_k_dev,
                        cudaStream_t stream,
                        int max_rounds);

// Device-output variant using pooled temporary buffers (recommended in hot loop)
void auction_assign_dev_pooled(const float* cost, int N, int M,
                               int* out_pairs_dev, int* out_k_dev,
                               float* price, int* owner, int* det_match, int* best_j, float* bids, unsigned long long* colwin,
                               cudaStream_t stream,
                               int max_rounds);


}
