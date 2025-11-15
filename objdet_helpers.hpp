#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>

inline void trim_str(std::string& s){
  size_t a = s.find_first_not_of(" \t\r\n");
  size_t b = s.find_last_not_of(" \t\r\n");
  if (a == std::string::npos){ s.clear(); return; }
  s = s.substr(a, b - a + 1);
}

// From TRT DetectRes box {left,top,right,bottom} produce clamped cv::Rect within width/height
template<typename BoxT>
inline cv::Rect rect_from_box_clamped(const BoxT& b, int width, int height){
  int x = std::max(0, (int)std::floor(b.left));
  int y = std::max(0, (int)std::floor(b.top));
  int w = std::max(0, (int)std::round(b.right - b.left));
  int h = std::max(0, (int)std::round(b.bottom - b.top));
  w = std::min(w, width - x);
  h = std::min(h, height - y);
  return (w>0 && h>0) ? cv::Rect(x,y,w,h) : cv::Rect();
}
