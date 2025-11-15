#pragma once
#include "Game_overlay.h"
#include "pch.hpp"
#include <string>
void GetVirtualScreen(int& x, int& y, int& w, int& h);
void SetPerMonitorV2DpiAwareness();
D2D1_COLOR_F ToD2DColor(OverlayColor argb);
// Short rect/color helpers
inline D2D1_RECT_F RF(float x,float y,float w,float h){ return D2D1::RectF(x,y,x+w,y+h); }
inline D2D1_COLOR_F C(OverlayColor a){ return ToD2DColor(a); }
// Wide to UTF-8
inline std::string w2u(const std::wstring& ws){ int len = WideCharToMultiByte(CP_UTF8,0,ws.c_str(),(int)ws.size(),nullptr,0,nullptr,nullptr); std::string out; out.resize(std::max(0,len)); if(len>0) WideCharToMultiByte(CP_UTF8,0,ws.c_str(),(int)ws.size(),out.data(),len,nullptr,nullptr); return out; }
