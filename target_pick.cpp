#include "target_pick.hpp"
#include "pch.hpp"

const Detection* pick_target(const std::vector<Detection>& ds, const Config& c, const cv::Point& ctr) {
    const Detection* target = nullptr;
    double min_dist_to_center = std::numeric_limits<double>::infinity();
    for (const auto& det : ds) {
        if (!c.is_target(det.class_id)) continue;
        cv::Point box_center(det.box.x + det.box.width / 2, det.box.y + det.box.height / 2);
        double dist = std::hypot(box_center.x - ctr.x, box_center.y - ctr.y);
        if (dist < min_dist_to_center) { min_dist_to_center = dist; target = &det; }
    }
    return target;
}
