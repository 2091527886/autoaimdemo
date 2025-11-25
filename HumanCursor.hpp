#pragma once
#include <vector>
#include <random>
#include <optional>
#include <algorithm>
#include "HC_Curve.hpp"
#include "HC_Random.hpp"
#include "HC_Buffer.hpp"
#include "HC_Duration.hpp"
#include "Clicker.hpp"
#include <windows.h>
#include <utility>

struct HumanCursorConfig {
    bool enable = true;
    bool steady = false;
    size_t queue_cap = 2048;
    int fps_for_duration = 60; // 来自 Config.fps（游戏锁60），用于 duration→points 映射
    hc::CurveRandConfig rand; // 可配置的人类曲线随机参数
};

class HumanCursor {
public:
    HumanCursor() = default;
    void init(const HumanCursorConfig& cfg){ cfg_ = cfg; steps_.set_cap(cfg.queue_cap); } // fps_from Config 已在 cfg_ 中
    void reset(){ steps_.clear(); origin_ = {0.0,0.0}; has_origin_=false; pending_clicks_=0; drag_active_=false; }
    void clear_steps(){ steps_.clear(); }

    // 绝对移动（屏幕坐标） - 生成完整人类曲线，入队逐帧消费
    void move_to(double x, double y, std::optional<double> duration = std::nullopt){
        move_to_impl(x, y, duration, std::nullopt);
    }
    // 可选 steady 覆盖版本（不改变全局 cfg_.steady）
    void move_to(double x, double y, std::optional<double> duration, std::optional<bool> steady_override){
        move_to_impl(x, y, duration, steady_override);
    }

    // 相对移动（用于 ADRC 每帧输出） - 打包为短子曲线
    void move_by_offset(double dx, double dy){
        ensure_origin();
        hc::Point from{origin_.first, origin_.second};
        hc::Point to{origin_.first + dx, origin_.second + dy};
        auto ro = hc::generate_random_curve_parameters(from, to, &cfg_.rand);
        if (cfg_.steady){ ro.offset_boundary_x = 10; ro.offset_boundary_y = 10; ro.distortion_mean = 1.2; ro.distortion_std = 1.2; ro.distortion_freq = 1.0; }
        // target_points 至少 2，至多 32（短步不产生巨量点）
        ro.target_points = std::max(2, std::min(ro.target_points, 32));
        hc::CurveParams p = hc::to_params(ro);
        hc::HumanizeMouseTrajectory curve(from, to, p);
        enqueue_curve(curve.points());
        origin_.first += dx; origin_.second += dy; has_origin_=true;
    }

    // 每帧弹出一个 double 步长（交给 mv_precise）。
    // 当队列为空时，会处理挂起的点击/拖拽动作，保持无阻塞、无 sleep。
    bool pop_step(double& dx, double& dy){
        if (steps_.pop_step(dx,dy)) return true;
        // 队列空：处理挂起动作
        if (pending_clicks_ > 0){
            Clicker::MouseDown();
            // 无 sleep，click_duration_ 被忽略（遵循“不主动 sleep，不改计时”）
            Clicker::MouseUp();
            --pending_clicks_;
            return false;
        }
        if (drag_active_){
            if (drag_phase_ == 0){
                // 到达 from_point 后按下
                Clicker::MouseDown();
                // 进入拖动阶段：生成到目标的曲线
                move_to(drag_to_.first, drag_to_.second, std::nullopt, drag_steady_);
                drag_phase_ = 1;
                return false;
            } else if (drag_phase_ == 1){
                // 已经到达 to_point（队列耗尽），释放
                Clicker::MouseUp();
                drag_active_ = false;
                drag_phase_ = 0;
                return false;
            }
        }
        return false;
    }

    // 点击/拖拽（完整 HumanCursor 功能保留，但不主动 sleep；动作按队列推进）
    void click_on(){ Clicker::MouseDown(); Clicker::MouseUp(); }
    void drag_begin(){ Clicker::MouseDown(); }
    void drag_end(){ Clicker::MouseUp(); }

    // 高层动作：移动到点后点击（不阻塞）。
    void click_on(double x, double y, int clicks = 1, double click_duration = 0.0, bool steady = false){
        // 先生成到目标点的曲线，等队列耗尽时逐个触发点击。
        move_to(x, y, std::nullopt, steady);
        pending_clicks_ = std::max(0, clicks);
        // 明确忽略 click_duration（不 sleep、不改计时），仅为接口兼容而保留参数
        (void)static_cast<void>(click_duration);
        click_duration_ = 0.0;
    }

    // 高层动作：拖拽 from → to（不阻塞）。
    void drag_and_drop(double from_x, double from_y, double to_x, double to_y, std::optional<double> duration = std::nullopt, bool steady=false){
        // 先移动到 from 点；到达后按下；再移动到 to 点；最后释放。
        move_to(from_x, from_y, duration, steady);
        drag_active_ = true; drag_phase_ = 0; drag_to_ = {to_x, to_y}; drag_steady_ = steady;
    }

private:
    void ensure_origin(){ if (!has_origin_) { POINT p; GetCursorPos(&p); origin_={ (double)p.x, (double)p.y }; has_origin_=true; } }
    void enqueue_curve(const std::vector<hc::Point>& pts){
        if (pts.size()<2) return;
        std::vector<std::pair<double,double>> steps; steps.reserve(pts.size()-1);
        for (size_t i=1;i<pts.size(); ++i){ steps.emplace_back(pts[i].x - pts[i-1].x, pts[i].y - pts[i-1].y); }
        steps_.push_curve(steps);
    }
    void move_to_impl(double x, double y, std::optional<double> duration, std::optional<bool> steady_override){
        ensure_origin();
        hc::Point from{origin_.first, origin_.second};
        hc::Point to{x,y};
        auto ro = hc::generate_random_curve_parameters(from, to, &cfg_.rand);
        bool steady_now = steady_override.has_value() ? *steady_override : cfg_.steady;
        if (steady_now){ ro.offset_boundary_x = 10; ro.offset_boundary_y = 10; ro.distortion_mean = 1.2; ro.distortion_std = 1.2; ro.distortion_freq = 1.0; }
        // 若指定了 duration，则将其映射为 target_points≈duration*fps（不 sleep，不改计时）
        int tp_from_duration = hc::map_duration_to_points(duration, cfg_.fps_for_duration, 2, cfg_.rand.target_points_cap>0?cfg_.rand.target_points_cap:4096);
        if (tp_from_duration > 0) ro.target_points = tp_from_duration;
        hc::CurveParams p = hc::to_params(ro);
        hc::HumanizeMouseTrajectory curve(from, to, p);
        enqueue_curve(curve.points());
        origin_ = {x,y}; has_origin_=true;
    }

    HumanCursorConfig cfg_{};
    hc::StepBuffer steps_{};
    std::pair<double,double> origin_{0.0,0.0};
    bool has_origin_ = false;

    // 高层动作状态
    int pending_clicks_ = 0;
    double click_duration_ = 0.0; // 不使用（不 sleep），仅保留接口兼容
    bool drag_active_ = false;
    int  drag_phase_ = 0; // 0: 待按下并生成 to 曲线, 1: 待释放
    std::pair<double,double> drag_to_{0.0,0.0};
    std::optional<bool> drag_steady_{};
};
