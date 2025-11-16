#pragma once
#include "pch.hpp"

inline float box_iou(const cv::Rect& a, const cv::Rect& b){
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width,  b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    int w = std::max(0, x2 - x1);
    int h = std::max(0, y2 - y1);
    int inter = w * h;
    int ua = a.width * a.height;
    int ub = b.width * b.height;
    int uni = ua + ub - inter;
    if (uni <= 0) return 0.0f;
    return (float)inter / (float)uni;
}

#include <string>
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include "sh.hpp"
#include "Detection.hpp"

inline void trim_str(std::string& s){
  size_t a = s.find_first_not_of(" \t\r\n");
  size_t b = s.find_last_not_of(" \t\r\n");
  if (a == std::string::npos){ s.clear(); return; }
  s = s.substr(a, b - a + 1);
}

// From TRT DetectRes box {left,top,right,bottom} produce clamped cv::Rect within width/height
template<typename BoxT>
inline cv::Rect rect_from_box_clamped(const BoxT& b, int width, int height){
  int x = MX(0, (int)std::floor(b.left));
  int y = MX(0, (int)std::floor(b.top));
  int w = MX(0, (int)std::round(b.right - b.left));
  int h = MX(0, (int)std::round(b.bottom - b.top));
  w = MN(w, width - x);
  h = MN(h, height - y);
  return (w>0 && h>0) ? cv::Rect(x,y,w,h) : cv::Rect();
}

inline bool accept_det(float score, float conf_thresh){ return score >= conf_thresh; }

template<typename BoxT>
inline Detection make_det(const BoxT& b, int w, int h, float score, int cls){ return Detection{rect_from_box_clamped(b,w,h), score, cls}; }
