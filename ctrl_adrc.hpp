#pragma once
#include <vector>
#include <utility>
#include <stdexcept>
#include <cmath>
#include "ctrl_utils.hpp"

// 轻度改写版的 ADRCController (来源: Control-Algorithm/ctrl_cpp/adrc.*)
class ADRCController {
private:
    // 参数
    std::vector<double> __r;
    std::vector<double> __b0;
    std::vector<double> __delta;
    std::vector<double> __beta01;
    std::vector<double> __beta02;
    std::vector<double> __beta03;
    std::vector<double> __beta1;
    std::vector<double> __beta2;
    std::vector<double> __alpha1;
    std::vector<double> __alpha2;
    std::vector<std::pair<double,double>> __control_limits;
    // fhan 简化档位的耦合系数 c（每轴可独立，默认 1.5）
    std::vector<double> __fhan_c;
    // 状态
    std::vector<double> __v1, __v2;
    std::vector<double> __z1, __z2, __z3;
    std::vector<double> __u;

private:
    static void _td(double v, double& v1, double& v2, double r, double h);
    static void _eso(double y, double& z1, double& z2, double& z3, double b0, double u, double beta01, double beta02, double beta03, double delta, double h);
    static double _nlsef(double e1, double e2, double alpha1, double alpha2, double beta1, double beta2, double delta);
    static double _nlsef(double e1, double e2, double r, double h, double c);
    static double _nlsef(double e1, double e2, double beta1, double beta2);
    static double _fhan(double x1, double x2, double r, double h);
    static double _fal(double e, double alpha, double delta);
    static double _fsg(double x, double d);

public:
    ADRCController(double r, double b0, double delta,
                   double eso_beta01, double eso_beta02, double eso_beta03,
                   double nlsef_beta1, double nlsef_beta2,
                   double nlsef_alpha1 = 0.9, double nlsef_alpha2 = 1.5,
                   const std::pair<double,double>& control_limits = { -ctrlutil::Infinity, ctrlutil::Infinity });

    ADRCController(const std::vector<double>& r, const std::vector<double>& b0, const std::vector<double>& delta,
                   const std::vector<double>& eso_beta01, const std::vector<double>& eso_beta02, const std::vector<double>& eso_beta03,
                   const std::vector<double>& nlsef_beta1, const std::vector<double>& nlsef_beta2,
                   const std::vector<double>& nlsef_alpha1 = {}, const std::vector<double>& nlsef_alpha2 = {},
                   const std::vector<std::pair<double,double>>& control_limits = {});

    double call(double setpoint, double feedback, double dt, int ctrl_low = 0);
    double operator()(double setpoint, double feedback, double dt, int ctrl_low = 0);
    std::vector<double> call(const std::vector<double>& setpoint, const std::vector<double>& feedback, double dt, int ctrl_low = 0);
    std::vector<double> operator()(const std::vector<double>& setpoint, const std::vector<double>& feedback, double dt, int ctrl_low = 0);

    // 配置 fhan 简化档位的 c 系数（统一设置所有轴）
    void set_fhan_c(double c);
};

