#pragma once

#include <cmath>

class PidController {
public:
    PidController() = default;
    PidController(double kp, double ki, double kd, double out_limit, double integ_limit);

    void SetGains(double kp, double ki, double kd);
    void SetLimits(double out_limit, double integ_limit);

    void Reset();

    double Update(double error, double dt_s);

private:
    double kp_ = 0.4;
    double ki_ = 0.0;
    double kd_ = 0.08;
    double out_limit_ = 50.0;    // 每帧输出像素限幅
    double integ_limit_ = 200.0; // 积分项限幅

    double prev_error_ = 0.0;
    double integral_ = 0.0;
    bool initialized_ = false;
};
