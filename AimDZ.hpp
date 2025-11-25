#pragma once
#include <utility>

// 简洁误差死区：半径 r（px），滞回 h（px）。
// 设计目标：最小侵入、无状态（先提供无状态版：区间内=0，区间外=原值；滞回先保留接口）。
class AimDZ {
public:
    void cfg(double r, double h){ r_ = (r>=0? r:0); h_ = (h>=0? h:0); if (h_ > r_) h_ = r_; }
    std::pair<double,double> apply(double ex, double ey, bool en) const {
        if (!en) return {ex, ey};
        double r = r_;
        if (r <= 0.0) return {ex, ey};
        double l2 = ex*ex + ey*ey;
        double r0 = (r_>h_? (r_-h_) : 0.0);
        double r1 = r_ + h_;
        double r0_2 = r0*r0;
        double r1_2 = r1*r1;
        if (l2 <= r0_2) return {0.0, 0.0};
        if (l2 >= r1_2) return {ex, ey};
        // 过渡带内：最小实现直接置零，保持简洁
        return {0.0, 0.0};
    }
private:
    double r_ = 0.0; // 半径 px
    double h_ = 0.0; // 滞回 px
};
