#pragma once
#include <random>
#include <vector>
#include <utility>
#include <windows.h>
#include "HC_Tween.hpp"
#include "HC_Curve.hpp"


namespace hc {

inline std::pair<int,int> get_screen_size(){
    int w = GetSystemMetrics(SM_CXSCREEN);
    int h = GetSystemMetrics(SM_CYSCREEN);
    return {w,h};
}

struct CurveRandOut {
    // 结果
    int offset_boundary_x;
    int offset_boundary_y;
    int knots_count;
    double distortion_mean;
    double distortion_std;
    double distortion_freq;
    TweenType tween;
    int target_points;
};

// 可配置的随机参数控制器（把 Python 的权重/范围暴露出来）
struct CurveRandConfig {
    // offset 三段分布区间与权重
    int off_r0_min=20, off_r0_max=45, off_r1_min=45, off_r1_max=75, off_r2_min=75, off_r2_max=100;
    int off_w0=20, off_w1=65, off_w2=15; // 权重
    // knots 权重，索引 0..9 对应 1..10
    int knots_w[10] = {15,36,17,12,8,4,3,2,15,5};
    // 扭曲参数范围（百分比→小数）
    int mean_min=80, mean_max=110, std_min=85, std_max=110, freq_min=25, freq_max=70;
    // tween 池（默认 default_tween_pool）与是否启用
    bool use_default_tween_pool = true;
    std::vector<TweenType> tween_pool;
    // 视口范围（屏幕内 15%-85%）
    double viewport_min_ratio = 0.15;
    double viewport_max_ratio = 0.85;
    // target_points 上限（防止 ADRC 小步产生过多点）
    int target_points_cap = 512;
};

inline CurveRandOut generate_random_curve_parameters(const hc::Point& pre_origin, const hc::Point& post_dest, const CurveRandConfig* cfg=nullptr){
    auto [vw, vh] = get_screen_size();
    double min_w = vw * (cfg?cfg->viewport_min_ratio:0.15), max_w = vw * (cfg?cfg->viewport_max_ratio:0.85);
    double min_h = vh * (cfg?cfg->viewport_min_ratio:0.15), max_h = vh * (cfg?cfg->viewport_max_ratio:0.85);

    static thread_local std::mt19937 g;
    static thread_local bool g_init = false;
    if (!g_init){
        std::random_device rd;
        std::seed_seq seed{ rd(), rd(), rd(), rd(), rd(), (unsigned)time(nullptr) };
        g.seed(seed);
        g_init = true;
    }

    // tween pool
    // tween pool
    std::vector<TweenType> tp = (cfg && !cfg->use_default_tween_pool)? cfg->tween_pool : default_tween_pool();
    if (tp.empty()) tp = default_tween_pool();
    std::uniform_int_distribution<int> twd(0, (int)tp.size()-1);

    auto pick_offset = [&](std::mt19937& g0){
        // 三段分布，由配置控制
        int w0 = cfg? cfg->off_w0 : 20; int w1 = cfg? cfg->off_w1 : 65; int w2 = cfg? cfg->off_w2 : 15;
        std::vector<double> _segw{ (double)w0, (double)w1, (double)w2 };
        std::discrete_distribution<int> seg(_segw.begin(), _segw.end());
        int s = seg(g0);
        std::uniform_int_distribution<int> r0(cfg?cfg->off_r0_min:20, cfg?cfg->off_r0_max:45);
        std::uniform_int_distribution<int> r1(cfg?cfg->off_r1_min:45, cfg?cfg->off_r1_max:75);
        std::uniform_int_distribution<int> r2(cfg?cfg->off_r2_min:75, cfg?cfg->off_r2_max:100);
        if (s==0) return r0(g); if (s==1) return r1(g); return r2(g);
    };

    auto offset_x = pick_offset(g);
    auto offset_y = pick_offset(g);

    // knots 分布近似
    std::vector<double> _kweights{
        (double)(cfg?cfg->knots_w[0]:15),
        (double)(cfg?cfg->knots_w[1]:36),
        (double)(cfg?cfg->knots_w[2]:17),
        (double)(cfg?cfg->knots_w[3]:12),
        (double)(cfg?cfg->knots_w[4]:8),
        (double)(cfg?cfg->knots_w[5]:4),
        (double)(cfg?cfg->knots_w[6]:3),
        (double)(cfg?cfg->knots_w[7]:2),
        (double)(cfg?cfg->knots_w[8]:15),
        (double)(cfg?cfg->knots_w[9]:5)
    };
    std::discrete_distribution<int> knots_d(_kweights.begin(), _kweights.end());
    int knots = knots_d(g); // 返回索引 0..9，对应 1..10
    knots = knots + 1;

    std::uniform_int_distribution<int> mean_d(cfg?cfg->mean_min:80, cfg?cfg->mean_max:110);
    std::uniform_int_distribution<int> std_d(cfg?cfg->std_min:85, cfg?cfg->std_max:110);
    std::uniform_int_distribution<int> freq_d(cfg?cfg->freq_min:25, cfg?cfg->freq_max:70);

    double distortion_mean = mean_d(g) / 100.0;
    double distortion_std  = std_d(g)  / 100.0;
    double distortion_freq = freq_d(g) / 100.0;

    // target_points = distance or >=2
    double dx = post_dest.x - pre_origin.x; double dy = post_dest.y - pre_origin.y;
    int target_points = std::max( (int)std::sqrt(dx*dx + dy*dy), 2);
    if (cfg && cfg->target_points_cap>0) target_points = std::min(target_points, cfg->target_points_cap);

    // 如果起点或终点不在 viewport 15%-85% 范围，强制最小扰动
    auto in_view = [&](const hc::Point& p){ return (p.x>=min_w && p.x<=max_w && p.y>=min_h && p.y<=max_h); };
    if (!in_view(pre_origin) || !in_view(post_dest)){
        offset_x = 1; offset_y = 1; knots = 1;
    }

    CurveRandOut out{};
    out.offset_boundary_x = offset_x;
    out.offset_boundary_y = offset_y;
    out.knots_count = knots;
    out.distortion_mean = distortion_mean;
    out.distortion_std = distortion_std;
    out.distortion_freq = distortion_freq;
    out.tween = tp[twd(g)];
    out.target_points = target_points;
    return out;
}

inline CurveParams to_params(const CurveRandOut& o){
    CurveParams p{};
    p.offset_boundary_x = o.offset_boundary_x;
    p.offset_boundary_y = o.offset_boundary_y;
    p.knots_count = o.knots_count;
    p.distortion_mean = o.distortion_mean;
    p.distortion_std = o.distortion_std;
    p.distortion_freq = o.distortion_freq;
    p.tween = o.tween;
    p.target_points = o.target_points;
    return p;
}

}
