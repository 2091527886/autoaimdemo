#pragma once

#include <cmath>

class PidController {
public:
    PidController() = default;
    PidController(double kp, double ki, double kd, double out_limit, double integ_limit);

    void setK(double kp, double ki, double kd);
    void setLim(double out_limit, double integ_limit);

    void reset();

    double step(double err, double dt);

private:
    double kp = 0.4;
    double ki = 0.0;
    double kd = 0.08;
    double out_lim = 50.0;    // 每帧输出像素限幅
    double i_lim = 200.0; // 积分项限幅

    double prev_err = 0.0;
    double i_sum = 0.0;
    bool init = false;
};
