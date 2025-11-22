#pragma once
#include <utility>
#include <memory>
#include "ctrl_adrc.hpp"
#include "Config.hpp"

class AimADRC {
public:
    AimADRC() = default;
    // 说明：
    // - use_dt_：为 true 时使用调用方传入的“实测 dt”；为 false 时在内部将 dt 视为 16ms（锁60节拍兼容）。
    // - ctrl_low_：控制律降阶/简化级别（0=完整，1/2=简化），便于快速稳定性切换。
    void setConfig(const Config& c){
        inv_sign_ = c.adrc_inv_sign;
        use_dt_ = c.adrc_use_dt;
        out_lim_ = (c.adrc_out_limit>0? c.adrc_out_limit : 360.0);
        ctrl_low_ = c.adrc_ctrl_low;
        // 记住参数用于 reset 重建
        prm_r_ = c.adrc_r; prm_b0_ = c.adrc_b0; prm_delta_ = c.adrc_delta;
        prm_b01_ = c.adrc_beta01; prm_b02_ = c.adrc_beta02; prm_b03_ = c.adrc_beta03;
        prm_b1_ = c.adrc_beta1; prm_b2_ = c.adrc_beta2; prm_a1_ = c.adrc_alpha1; prm_a2_ = c.adrc_alpha2;
        build();
    }
    void reset(){ build(); }
    std::pair<double,double> step(double ex, double ey, double dt){
        if (!ax_ || !ay_) return {0.0, 0.0};
        double d = use_dt_ ? dt : (16.0/1000.0);
        double v = 0.0; // 跟踪目标为0误差
        double fx = inv_sign_ ? -ex : ex;
        double fy = inv_sign_ ? -ey : ey;
        double ux = ax_->call(v, fx, d, ctrl_low_);
        double uy = ay_->call(v, fy, d, ctrl_low_);
        // already limited in controller; ensure again
        if (ux >  out_lim_) ux =  out_lim_; if (ux < -out_lim_) ux = -out_lim_;
        if (uy >  out_lim_) uy =  out_lim_; if (uy < -out_lim_) uy = -out_lim_;
        return { ux, uy };
    }
private:
    void build(){
        auto lim = std::pair<double,double>{-out_lim_, out_lim_};
        ax_ = std::make_unique<ADRCController>(prm_r_, prm_b0_, prm_delta_, prm_b01_, prm_b02_, prm_b03_, prm_b1_, prm_b2_, prm_a1_, prm_a2_, lim);
        ay_ = std::make_unique<ADRCController>(prm_r_, prm_b0_, prm_delta_, prm_b01_, prm_b02_, prm_b03_, prm_b1_, prm_b2_, prm_a1_, prm_a2_, lim);
    }
    std::unique_ptr<ADRCController> ax_;
    std::unique_ptr<ADRCController> ay_;
    int ctrl_low_ = 0;
    bool use_dt_ = true;
    bool inv_sign_ = false;
    double out_lim_ = 360.0;
    // 保存参数
    double prm_r_ = 80.0, prm_b0_ = 1.0, prm_delta_ = 0.05;
    double prm_b01_ = 100.0, prm_b02_ = 300.0, prm_b03_ = 100.0;
    double prm_b1_ = 0.6, prm_b2_ = 0.1, prm_a1_ = 0.9, prm_a2_ = 1.5;
};
