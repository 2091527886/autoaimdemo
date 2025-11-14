#pragma once

#include "letterbox_kernels_decl.cuh"
#include "letterbox_common.cuh"

namespace trtyolo {

static inline dim3 pickBlock(const ProcessConfig& cfg, int /*dst_cols*/) {
    int bx = cfg.block_x > 0 ? cfg.block_x : 32;
    int by = cfg.block_y > 0 ? cfg.block_y : 8;
    if (cfg.use_occupancy) {
        const int cand[][2] = {{32,8},{16,16},{64,4},{32,16},{64,8},{48,8},{24,16},{40,8},{20,16}};
        int best_bx = bx, best_by = by; int best_act = -1;
        for (auto& c : cand) {
            int cbx = c[0], cby = c[1];
            int threads = cbx * cby;
            int active = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active,
                (const void*)gpuLetterboxF16Vec4_generic,
                threads,
                0);
            if (active > best_act) { best_act = active; best_bx = cbx; best_by = cby; }
        }
        bx = best_bx; by = best_by;
    }
    return dim3(bx, by, 1);
}

} // namespace trtyolo
