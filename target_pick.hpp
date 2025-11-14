#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "Detection.hpp"
#include "Config.hpp"

// helpers
// 从检测集合中选取距离中心点最近的目标（且类别符合 is_target）
const Detection* pick_target(const std::vector<Detection>& ds, const Config& c, const cv::Point& ctr);
