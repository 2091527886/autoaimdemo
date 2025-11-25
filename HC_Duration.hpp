#pragma once
#include <optional>
#include <cmath>

namespace hc {

// 将 duration(秒) 转换为目标点数；不改 FPS/计时，纯映射，不 sleep。
// fps 默认为 60（你的游戏已锁 60）。
inline int map_duration_to_points(std::optional<double> duration_sec, int fps = 60, int min_points = 2, int max_points = 4096){
    if (!duration_sec.has_value()) return -1; // -1 表示保持默认策略（用距离估计/随机）
    double d = *duration_sec; if (d <= 0.0) return -1;
    double points = d * static_cast<double>(fps);
    int tp = static_cast<int>(std::round(points));
    if (tp < min_points) tp = min_points; if (tp > max_points) tp = max_points;
    return tp;
}

}
