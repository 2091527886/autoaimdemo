#include "letterbox.hpp"
#include "letterbox_kernels.cuh"
#include "letterbox_common.cuh"

namespace trtyolo {

// 选择块大小（占用推荐或配置）；若 use_occupancy=1，将在候选集中估算 active blocks 选择最优
static inline dim3 pickBlock(const ProcessConfig& cfg, int dst_cols) {
    int bx = cfg.block_x > 0 ? cfg.block_x : 32;
    int by = cfg.block_y > 0 ? cfg.block_y : 8;
    if (cfg.use_occupancy) {
        // 候选配置集合（针对3060 SM86）
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
    // 对齐到2像素/线程：以像素对为单位
    return dim3(bx, by, 1);
}

// Half path
void cudaLetterboxHalf(const void* src, int src_cols, int src_rows, size_t src_pitch,
                       void* dst, int dst_cols, int dst_rows,
                       const LetterboxTransform& params, const ProcessConfig config, cudaStream_t stream) {
    dim3 block = pickBlock(config, dst_cols);
    dim3 grid(iDivUp((dst_cols + 3) / 4, block.x), iDivUp(dst_rows, block.y));
    if (config.use_launch_bounds) {
        // 模板分派（常见配置），否则退回通用
        int MINB = config.min_blocks_per_sm; MINB = (MINB < 1 ? 1 : (MINB > 3 ? 3 : MINB));
        if (block.x == 32 && block.y == 8) {
            if (MINB<=1)      gpuLetterboxF16Vec4_lb<32,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec4_lb<32,8,2><<<grid, block, 0, stream>>> (src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec4_lb<32,8,3><<<grid, block, 0, stream>>> (src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 16 && block.y == 16) {
            if (MINB<=1)      gpuLetterboxF16Vec4_lb<16,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec4_lb<16,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec4_lb<16,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 64 && block.y == 4) {
            if (MINB<=1)      gpuLetterboxF16Vec4_lb<64,4,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec4_lb<64,4,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec4_lb<64,4,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 32 && block.y == 16) {
            if (MINB<=1)      gpuLetterboxF16Vec4_lb<32,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec4_lb<32,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec4_lb<32,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 64 && block.y == 8) {
            if (MINB<=1)      gpuLetterboxF16Vec4_lb<64,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec4_lb<64,8,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec4_lb<64,8,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 48 && block.y == 8) {
            if (MINB<=1)      gpuLetterboxF16Vec4_lb<48,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec4_lb<48,8,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec4_lb<48,8,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 24 && block.y == 16) {
            if (MINB<=1)      gpuLetterboxF16Vec4_lb<24,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec4_lb<24,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec4_lb<24,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 40 && block.y == 8) {
            if (MINB<=1)      gpuLetterboxF16Vec4_lb<40,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec4_lb<40,8,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec4_lb<40,8,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 20 && block.y == 16) {
            if (MINB<=1)      gpuLetterboxF16Vec4_lb<20,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec4_lb<20,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec4_lb<20,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else {
            gpuLetterboxF16Vec4_generic<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        }
    } else {
        gpuLetterboxF16Vec4_generic<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
    }
}

void cudaMultiLetterboxHalf(const void* src, int src_cols, int src_rows, size_t src_pitch,
                            void* dst, int dst_cols, int dst_rows,
                            const LetterboxTransform& params, const ProcessConfig config,
                            int num_images, cudaStream_t stream) {
    dim3 block = pickBlock(config, dst_cols);
    dim3 grid(iDivUp((dst_cols + 3) / 4, block.x), iDivUp(dst_rows, block.y), num_images);
    if (config.use_launch_bounds) {
        int MINB = config.min_blocks_per_sm; MINB = (MINB < 1 ? 1 : (MINB > 3 ? 3 : MINB));
        if (block.x == 32 && block.y == 8) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<32,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<32,8,2><<<grid, block, 0, stream>>> (src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<32,8,3><<<grid, block, 0, stream>>> (src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 16 && block.y == 16) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<16,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<16,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<16,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 64 && block.y == 4) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<64,4,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<64,4,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<64,4,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 32 && block.y == 16) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<32,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<32,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<32,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 64 && block.y == 8) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<64,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<64,8,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<64,8,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 48 && block.y == 8) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<48,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<48,8,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<48,8,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 24 && block.y == 16) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<24,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<24,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<24,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 40 && block.y == 8) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<40,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<40,8,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<40,8,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 20 && block.y == 16) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<20,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<20,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<20,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else {
            gpuMultiLetterboxHalfVec4_generic<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        }
    } else {
        gpuMultiLetterboxHalfVec4_generic<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
    }
}

} // namespace trtyolo
