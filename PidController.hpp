#pragma once

class PidController {
public:
    PidController() = default;
    PidController(double kp, double ki, double kd, double out_limit, double integ_limit);

    void setK(double kp, double ki, double kd);
    void setLim(double out_limit, double integ_limit);

    // dt clamp config
    void setDtClamp(bool enable, double min_ms, double max_ms);
    // anti-windup config
    void setAntiWindupFreeze(bool enable);

    void reset();

    double step(double err, double dt);

private:
    double kp = 0.4;
    double ki = 0.0;
    double kd = 0.08;
    double out_lim = 50.0;    // 每帧输出像素限幅
    double i_lim = 200.0; // 积分项限幅

    bool clamp_dt = true;
    double dt_min = 0.001; // 1ms
    double dt_max = 0.010; // 10ms
    bool aw_freeze = true; // 饱和冻结积分

    double prev_err = 0.0;
    double i_sum = 0.0;
    bool init = false;
};
