#pragma once
#include <vector>
#include <cmath>
#include <utility>

namespace hc {

struct Point { double x; double y; };

struct BezierCalculator {
    static inline double binomial(int n, int k){
        if (k<0 || k>n) return 0.0; if (k==0 || k==n) return 1.0;
        double r = 1.0; for (int i=1; i<=k; ++i){ r *= (n - (k - i)); r /= i; } return r;
    }
    static inline double bernstein(double t, int i, int n){
        return binomial(n, i) * std::pow(t, i) * std::pow(1.0 - t, n - i);
    }
    static std::pair<double,double> eval_curve(double t, const std::vector<Point>& pts){
        int n = (int)pts.size()-1; double x=0,y=0; for(int i=0;i<=n;++i){ double b=bernstein(t,i,n); x+=pts[i].x*b; y+=pts[i].y*b; } return {x,y};
    }
    static std::vector<Point> calculate_points_in_curve(int n, const std::vector<Point>& control){
        if (n<2) n=2; std::vector<Point> out; out.reserve(n);
        for(int i=0;i<n;++i){ double t = (n==1)?0.0: (double)i / (double)(n-1); auto p = eval_curve(t, control); out.push_back({p.first, p.second}); }
        return out;
    }
};

}
