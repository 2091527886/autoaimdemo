#pragma once
#include <vector>
#include <optional>
#include <string>
#include <cmath>
#include <opencv2/core.hpp>
#include "Detection.hpp"
#include "Game_overlay.h"
#include "OCSort.hpp"
#include "Config.hpp"

struct SelectedTarget {
    int track_id = -1;
    int cls = -1;
    // original observation box (for aim/mouse)
    cv::Rect obs;
    // track box (for overlay)
    cv::Rect trk;
    float score = 0.f;
    bool valid() const { return track_id >= 0; }
};

inline double center_score(const cv::Rect& r, const cv::Point& ctr, int win_w, int win_h){
    cv::Rect win(ctr.x - win_w/2, ctr.y - win_h/2, win_w, win_h);
    if ((r & win).area() > 0) return 1.0; // inside window
    // outside: inverse distance score
    double cx = r.x + r.width*0.5; double cy = r.y + r.height*0.5;
    double dx = cx - ctr.x, dy = cy - ctr.y;
    double d  = std::sqrt(dx*dx + dy*dy) + 1e-3;
    return 1.0 / (1.0 + d * 0.01);
}

inline SelectedTarget select_target(const Config& c, const std::vector<OCSort::Trk>& trks,
                                    const cv::Point& ctr){
    SelectedTarget best;
    for (const auto& t : trks){
        if (!t.valid) continue;
        if (!c.is_target(t.cls)) continue; // class filter
        cv::Rect obs((int)t.ox1,(int)t.oy1,(int)(t.ox2-t.ox1),(int)(t.oy2-t.oy1));
        cv::Rect trk((int)t.x1,(int)t.y1,(int)(t.x2-t.x1),(int)(t.y2-t.y1));
        double sc = 0.0;
        if (c.sel_mode == "center_win"){
            sc = center_score(obs, ctr, c.sel_center_win_w, c.sel_center_win_h) + t.score * c.sel_conf_w;
        } else if (c.sel_mode == "area_first"){
            double area = (double)obs.area();
            sc = c.sel_area_w * area + c.sel_conf_w * t.score + center_score(obs, ctr, c.sel_center_win_w, c.sel_center_win_h);
        } else /* conf_first */{
            sc = c.sel_conf_w * t.score + center_score(obs, ctr, c.sel_center_win_w, c.sel_center_win_h);
        }
        if (!best.valid() || sc > best.score){
            best.score = (float)sc;
            best.track_id = t.id; best.cls = t.cls; best.obs = obs; best.trk = trk;
        }
    }
    return best;
}
