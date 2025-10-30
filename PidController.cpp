#include "PidController.hpp"
#include <cmath>

PidController::PidController(double kp, double ki, double kd, double out_limit, double integ_limit)
    : kp(kp), ki(ki), kd(kd), out_lim(std::abs(out_limit)), i_lim(std::abs(integ_limit)) {}

void PidController::setK(double kp, double ki, double kd) { this->kp = kp; this->ki = ki; this->kd = kd; }

void PidController::setLim(double out_limit, double integ_limit) {
    out_lim = std::abs(out_limit);
    i_lim = std::abs(integ_limit);
}

void PidController::reset() { prev_err = 0.0; i_sum = 0.0; init = false; }

double PidController::step(double err, double dt) {
    if (dt <= 0.0) dt = 1.0 / 60.0; // fallback
    if (!init) { prev_err = err; init = true; }

    // 误差导数
    double d = (err - prev_err) / dt;

    // 积分项
    i_sum += err * dt;
    // 积分限幅
    if (i_lim > 0.0) {
        if (i_sum > i_lim) i_sum = i_lim;
        else if (i_sum < -i_lim) i_sum = -i_lim;
    }

    double out = kp * err + ki * i_sum + kd * d;

    // 输出限幅 + 抗饱和：若饱和则在下次积分前自然恢复（简单策略）
    if (out_lim > 0.0) {
        if (out > out_lim) out = out_lim;
        else if (out < -out_lim) out = -out_lim;
    }

    prev_err = err;
    return out;
}
