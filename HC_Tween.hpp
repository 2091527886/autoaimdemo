#pragma once
#include <cmath>
#include <vector>

namespace hcconst { constexpr double PI = 3.14159265358979323846; }

enum class TweenType {
    Linear,
    EaseOutExpo,
    EaseInOutQuint,
    EaseInOutSine,
    EaseInOutQuart,
    EaseInOutExpo,
    EaseInOutCubic,
    EaseInOutCirc,
    EaseOutSine,
    EaseOutQuart,
    EaseOutQuint,
    EaseOutCubic,
    EaseOutCirc
};

inline double tween_eval(TweenType t, double x){
    x = (x<0)?0:((x>1)?1:x);
    switch(t){
        case TweenType::Linear: return x;
        case TweenType::EaseOutExpo: return (x==1.0)?1.0:1 - std::pow(2.0, -10.0 * x);
        case TweenType::EaseInOutQuint:{ if (x<0.5){ return 16*x*x*x*x*x; } double u = (2*x-2); return 0.5 * u*u*u*u*u + 1; }
        case TweenType::EaseInOutSine: return -(std::cos(hcconst::PI*x) - 1.0)/2.0;
        case TweenType::EaseInOutQuart:{ if (x<0.5) return 8*x*x*x*x; double u=(x-1); return 1-8*u*u*u*u; }
        case TweenType::EaseInOutExpo:{ if (x==0.0) return 0.0; if (x==1.0) return 1.0; if (x<0.5) return std::pow(2.0, 20.0*x-10.0)/2.0; return (2.0 - std::pow(2.0, -20.0*x+10.0))/2.0; }
        case TweenType::EaseInOutCubic:{ if (x<0.5) return 4*x*x*x; double u=(2*x-2); return 0.5*u*u*u + 1; }
        case TweenType::EaseInOutCirc:{ if (x<0.5) return (1-std::sqrt(1-4*x*x))/2.0; double u=(2*x-1); return (std::sqrt(1-u*u)+1)/2.0; }
        case TweenType::EaseOutSine: return std::sin((x*hcconst::PI)/2.0);
        case TweenType::EaseOutQuart: { double u=1-x; return 1 - u*u*u*u; }
        case TweenType::EaseOutQuint: { double u=1-x; return 1 - u*u*u*u*u; }
        case TweenType::EaseOutCubic: { double u=1-x; return 1 - u*u*u; }
        case TweenType::EaseOutCirc: return std::sqrt(1 - (x-1)*(x-1));
    }
    return x;
}

inline const std::vector<TweenType>& default_tween_pool(){
    static std::vector<TweenType> pool = {
        TweenType::EaseOutExpo,
        TweenType::EaseInOutQuint,
        TweenType::EaseInOutSine,
        TweenType::EaseInOutQuart,
        TweenType::EaseInOutExpo,
        TweenType::EaseInOutCubic,
        TweenType::EaseInOutCirc,
        TweenType::Linear,
        TweenType::EaseOutSine,
        TweenType::EaseOutQuart,
        TweenType::EaseOutQuint,
        TweenType::EaseOutCubic,
        TweenType::EaseOutCirc
    };
    return pool;
}
