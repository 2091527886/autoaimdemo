#include "PidController.hpp"
#include <cmath>
#include <algorithm>

PidController::PidController(double kp_in, double ki_in, double kd_in, double out_limit, double integ_limit)
    : kp(kp_in), ki(ki_in), kd(kd_in), out_lim(std::abs(out_limit)), i_lim(std::abs(integ_limit)) {}

void PidController::setK(double kp_in, double ki_in, double kd_in) { this->kp = kp_in; this->ki = ki_in; this->kd = kd_in; }

void PidController::setLim(double out_limit, double integ_limit) {
    out_lim = std::abs(out_limit);
    i_lim = std::abs(integ_limit);
}

void PidController::setDtClamp(bool enable, double min_ms, double max_ms) {
    clamp_dt = enable;
    dt_min = std::max(0.000001, min_ms / 1000.0);
    dt_max = std::max(dt_min, max_ms / 1000.0);
}

void PidController::setAntiWindupFreeze(bool enable) { aw_freeze = enable; }

void PidController::reset() { prev_err = 0.0; i_sum = 0.0; init = false; }

double PidController::step(double err, double dt) {
    // dt 夹紧（可配置）
    if (!(dt > 0.0)) dt = 1.0 / 60.0; // fallback
    if (clamp_dt) {
        if (dt < dt_min) dt = dt_min;
        if (dt > dt_max) dt = dt_max;
    }

    if (!init) { prev_err = err; init = true; }

    // 误差导数
    double d = (err - prev_err) / dt;

    // 积分项：先暂存，待输出是否饱和后再决定是否采纳
    double i_sum_new = i_sum + err * dt;
    // 预计算积分限幅
    if (i_lim > 0.0) {
        if (i_sum_new > i_lim) i_sum_new = i_lim;
        else if (i_sum_new < -i_lim) i_sum_new = -i_lim;
    }

    // 用新积分计算输出
    double out = kp * err + ki * i_sum_new + kd * d;

    // 输出限幅
    bool saturated = false;
    if (out_lim > 0.0) {
        if (out > out_lim) { out = out_lim; saturated = true; }
        else if (out < -out_lim) { out = -out_lim; saturated = true; }
    }

    // 抗风up：冻结积分（可配置）
    if (!saturated || !aw_freeze) {
        i_sum = i_sum_new;
    }

    prev_err = err;
    return out;
}
