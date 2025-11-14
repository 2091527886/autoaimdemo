#include "Aim.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>

void Aim::mv(int mx, int my, double& ms) {
    if (mx || my) {
        auto t0 = (c.timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{});
        mouse.Move(mx, my);
        auto t1 = (c.timing ? std::chrono::high_resolution_clock::now() : t0);
        ms = c.timing ? std::chrono::duration<double, std::milli>(t1 - t0).count() : 0.0;
    }
}

void Aim::mv_wind(int mx, int my, double& ms) {
    auto t0 = (c.timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{});
    cv::Point2d want{ (double)mx, (double)my };
    want.x += wind_.residual.x; want.y += wind_.residual.y;

    double seg_scale = 1.0;
    if (c.wind_use_seg_scale) {
        if (last_seg_ == 2) seg_scale = c.wind_near_scale;
        else if (last_seg_ == 1) seg_scale = c.wind_mid_scale;
        else seg_scale = c.wind_far_scale;
    }
    double step_min = std::max(0.0, c.wind_step_min_px * seg_scale);
    double step_max = std::max(step_min, c.wind_step_max_px * seg_scale);
    double wind_k   = c.wind_wind_k * seg_scale;
    double grav_k   = c.wind_gravity_k * seg_scale;
    double jitter_s = c.wind_jitter_std_px * seg_scale;
    double end_slow = std::max(0.0, c.wind_end_slow_px * seg_scale);
    double out_lim  = std::max(0.0, c.wind_out_limit_px);
    double dead_px  = std::max(0.0, c.wind_dead_px * seg_scale);

    auto budget_end = std::chrono::high_resolution_clock::now()
        + std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(
              std::chrono::duration<double, std::milli>(std::max(0.0, c.wind_time_budget_ms)));
    int steps = 0;
    double sent = 0.0;
    const int steps_cap = std::max(1, c.wind_max_steps_per_frame);

    while (steps < steps_cap) {
        auto now = std::chrono::high_resolution_clock::now();
        if (now >= budget_end) break;
        double dist = std::hypot(want.x, want.y);
        if (dist <= dead_px) break;

        double ux = (dist > 1e-6) ? (want.x / dist) : 0.0;
        double uy = (dist > 1e-6) ? (want.y / dist) : 0.0;
        double tx = -uy, ty = ux;

        double t = std::clamp(1.0 - dist / std::max(1.0, end_slow * 4.0), 0.0, 1.0);
        double curve = std::clamp(c.wind_speed_curve, 0.0, 1.0);
        double bell = (1.0 - curve) * t + curve * (1.0 - std::abs(2.0 * t - 1.0));
        bell = std::clamp(bell, 0.1, 1.0);
        double step_len = std::clamp(dist * 0.25 * bell, step_min, step_max);

        double jx = wind_.n01(wind_.rng) * jitter_s;
        double jy = wind_.n01(wind_.rng) * jitter_s;

        double sx = grav_k * ux + wind_k * tx + jx;
        double sy = grav_k * uy + wind_k * ty + jy;
        double s_norm = std::hypot(sx, sy);
        if (s_norm < 1e-6) { sx = ux; sy = uy; s_norm = 1.0; }
        sx = sx / s_norm * step_len;
        sy = sy / s_norm * step_len;

        if (dist < end_slow) {
            sx -= c.wind_backtrack_k * (want.x - sx);
            sy -= c.wind_backtrack_k * (want.y - sy);
        }

        int dx = (int)std::lround(sx);
        int dy = (int)std::lround(sy);
        if (dx == 0 && dy == 0) {
            dx = (want.x > 0) ? 1 : ((want.x < 0) ? -1 : 0);
            dy = (want.y > 0) ? 1 : ((want.y < 0) ? -1 : 0);
            if (dx == 0 && dy == 0) break;
        }

        mouse.Move(dx, dy);
        want.x -= dx; want.y -= dy;
        sent += std::hypot((double)dx, (double)dy);
        ++steps;
        if (sent >= out_lim) break;
    }

    if (c.wind_target_noise_enable) {
        if (std::hypot(want.x, want.y) <= dead_px && c.wind_target_noise_px > 0.0) {
            int ndx = (int)std::lround(wind_.n01(wind_.rng) * c.wind_target_noise_px * 0.5);
            int ndy = (int)std::lround(wind_.n01(wind_.rng) * c.wind_target_noise_px * 0.5);
            if (ndx || ndy) mouse.Move(ndx, ndy);
        }
    }

    wind_.residual.x = want.x * c.wind_residual_carry;
    wind_.residual.y = want.y * c.wind_residual_carry;

    auto t1 = (c.timing ? std::chrono::high_resolution_clock::now() : t0);
    ms = c.timing ? std::chrono::duration<double, std::milli>(t1 - t0).count() : 0.0;
}
