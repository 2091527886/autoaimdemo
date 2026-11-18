#pragma once
#include <vector>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <optional>
#include "HC_Bezier.hpp"
#include "HC_Tween.hpp"

namespace hc {

struct CurveParams {
    int offset_boundary_x = 80;
    int offset_boundary_y = 80;
    int knots_count = 2;
    double distortion_mean = 1.0;
    double distortion_std = 1.0;
    double distortion_freq = 0.5;
    TweenType tween = TweenType::EaseOutExpo;
    int target_points = 100;
    int left_boundary = 0, right_boundary = 0, down_boundary = 0, up_boundary = 0;
};

class HumanizeMouseTrajectory {
public:
    HumanizeMouseTrajectory(const hc::Point& from_point, const hc::Point& to_point, const CurveParams& p)
        : from_(from_point), to_(to_point), params_(p){ generate_curve(); }

    const std::vector<hc::Point>& points() const { return points_; }

private:
    hc::Point from_, to_;
    CurveParams params_;
    std::vector<hc::Point> points_;

    static bool is_numeric(double v){ (void)v; return true; }

    void generate_curve(){
        // boundaries
        int lx = std::min((int)from_.x, (int)to_.x) - params_.offset_boundary_x;
        int rx = std::max((int)from_.x, (int)to_.x) + params_.offset_boundary_x;
        int dy = std::min((int)from_.y, (int)to_.y) - params_.offset_boundary_y;
        int uy = std::max((int)from_.y, (int)to_.y) + params_.offset_boundary_y;
        params_.left_boundary = lx; params_.right_boundary = rx; params_.down_boundary = dy; params_.up_boundary = uy;

        auto knots = generate_internal_knots(lx, rx, dy, uy, params_.knots_count);
        auto pts = generate_points(knots);
        pts = distort_points(pts, params_.distortion_mean, params_.distortion_std, params_.distortion_freq);
        points_ = tween_points(pts, params_.tween, params_.target_points);
    }

    std::vector<hc::Point> generate_internal_knots(int l, int r, int d, int u, int knots_count){
        if (knots_count < 0) knots_count = 0;
        if (l>r || d>u) throw std::invalid_argument("boundary error");
        std::vector<hc::Point> ks; ks.reserve(knots_count);
        std::random_device rd; std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dx(l, r), dy(d, u);
        for (int i=0;i<knots_count;++i){ ks.push_back(hc::Point{ (double)dx(gen), (double)dy(gen)}); }
        return ks;
    }

    std::vector<hc::Point> generate_points(const std::vector<hc::Point>& knots){
        std::vector<hc::Point> cps; cps.reserve(knots.size()+2); cps.push_back(from_); cps.insert(cps.end(), knots.begin(), knots.end()); cps.push_back(to_);
        int midPtsCnt = std::max({ (int)std::abs((int)from_.x - (int)to_.x), (int)std::abs((int)from_.y - (int)to_.y), 2});
        return BezierCalculator::calculate_points_in_curve(midPtsCnt, cps);
    }

    std::vector<hc::Point> distort_points(const std::vector<hc::Point>& pts, double mean, double stdv, double freq){
        if (!(freq>=0.0 && freq<=1.0)) throw std::invalid_argument("freq range");
        if (!std::isfinite(mean) || !std::isfinite(stdv) || stdv < 0.0) throw std::invalid_argument("invalid distortion parameters");
        if (freq == 0.0) return pts;
        std::random_device rd; std::mt19937 gen(rd());
        std::optional<std::normal_distribution<double>> norm;
        if (stdv > 0.0) norm.emplace(mean, stdv);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<hc::Point> out; out.reserve(pts.size());
        for (size_t i=0;i<pts.size(); ++i){
            if (i==0 || i+1==pts.size()){ out.push_back(pts[i]); continue; }
            double delta = (uni(gen) < freq) ? (norm ? (*norm)(gen) : mean) : 0.0;
            out.push_back(hc::Point{ pts[i].x, pts[i].y + delta });
        }
        if (!pts.empty()){
            out.front() = pts.front(); out.back() = pts.back();
        }
        return out;
    }

    std::vector<hc::Point> tween_points(const std::vector<hc::Point>& pts, TweenType tw, int target_points){
        if (target_points < 2) target_points = 2;
        std::vector<hc::Point> res; res.reserve(target_points);
        for (int i=0;i<target_points; ++i){
            double ti = (double)i / (double)(target_points - 1);
            int idx = (int)(tween_eval(tw, ti) * (double)(pts.size()-1));
            if (idx < 0) idx = 0; if (idx >= (int)pts.size()) idx = (int)pts.size()-1;
            res.push_back(pts[idx]);
        }
        return res;
    }
};

}
