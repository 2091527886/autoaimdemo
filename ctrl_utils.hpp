#pragma once
#include <algorithm>
#include <vector>
#include <utility>
#include <cmath>

namespace ctrlutil {

constexpr double Infinity = 100000000.0;

inline double clip(double x, double x_min, double x_max){
    if (x < x_min) return x_min; if (x > x_max) return x_max; return x;
}

inline std::vector<double> clip(const std::vector<double>& x, double x_min, double x_max){
    std::vector<double> r; r.reserve(x.size());
    for (auto v : x) r.push_back(clip(v, x_min, x_max));
    return r;
}

inline double sign(double x){ return (x>0) - (x<0); }
inline int sign(int x){ return (x>0) - (x<0); }

} // namespace ctrlutil
