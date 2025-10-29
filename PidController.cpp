#include "PidController.hpp"
#include <cmath>

PidController::PidController(double kp, double ki, double kd, double out_limit, double integ_limit)
    : kp_(kp), ki_(ki), kd_(kd), out_limit_(std::abs(out_limit)), integ_limit_(std::abs(integ_limit)) {}

void PidController::SetGains(double kp, double ki, double kd) { kp_ = kp; ki_ = ki; kd_ = kd; }

void PidController::SetLimits(double out_limit, double integ_limit) {
    out_limit_ = std::abs(out_limit);
    integ_limit_ = std::abs(integ_limit);
}

void PidController::Reset() { prev_error_ = 0.0; integral_ = 0.0; initialized_ = false; }

double PidController::Update(double error, double dt_s) {
    if (dt_s <= 0.0) dt_s = 1.0 / 60.0; // fallback
    if (!initialized_) { prev_error_ = error; initialized_ = true; }

    // 误差导数
    double derivative = (error - prev_error_) / dt_s;

    // 积分项
    integral_ += error * dt_s;
    // 积分限幅
    if (integ_limit_ > 0.0) {
        if (integral_ > integ_limit_) integral_ = integ_limit_;
        else if (integral_ < -integ_limit_) integral_ = -integ_limit_;
    }

    double output = kp_ * error + ki_ * integral_ + kd_ * derivative;

    // 输出限幅 + 抗饱和：若饱和则在下次积分前自然恢复（简单策略）
    if (out_limit_ > 0.0) {
        if (output > out_limit_) output = out_limit_;
        else if (output < -out_limit_) output = -out_limit_;
    }

    prev_error_ = error;
    return output;
}
