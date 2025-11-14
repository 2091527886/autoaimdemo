#pragma once

#include "PropAlgo.hpp"
#include "Config.hpp"
#include <utility>

// 鼠标移动算法封装（双轴 PropAlgo）
class AimAlgo {
public:
    AimAlgo() = default;

    explicit AimAlgo(const Config& c) { setConfig(c); }

    void setConfig(const Config& c) {
                x_.setK(c.prop_k_x);
        x_.setLim(c.prop_out_limit);
        x_.setTc(c.prop_tc_ms);
        x_.setDz(c.prop_dead_px);
        y_.setK(c.prop_k_y);
        y_.setLim(c.prop_out_limit);
        y_.setTc(c.prop_tc_ms);
        y_.setDz(c.prop_dead_px);
        use_dt_ = c.prop_use_dt;
    }

    void reset() { x_.reset(); y_.reset(); }

    // 输入像素误差，返回像素级校正量（再乘以 mfac 映射为鼠标）
    std::pair<double,double> step(double ex, double ey, double dt) {
        double d = use_dt_ ? dt : (16.0/1000.0);
        return { x_.step(ex, d), y_.step(ey, d) };
    }

public:
    // 动态设置参数（分段用）
    void setParams(double kx, double ky, double tc_ms, double out_lim_px) {
        x_.setK(kx); y_.setK(ky);
        x_.setTc(tc_ms); y_.setTc(tc_ms);
        x_.setLim(out_lim_px); y_.setLim(out_lim_px);
    }
private:
    PropAlgo x_;
    PropAlgo y_;
    bool use_dt_ = true;
};
