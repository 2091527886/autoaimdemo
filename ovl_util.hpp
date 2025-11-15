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
inline const wchar_t* PickFont(const std::wstring& name){ return name.empty()? L"Segoe UI" : name.c_str(); }
// Wide to UTF-8
inline std::string w2u(const std::wstring& ws){ int len = WideCharToMultiByte(CP_UTF8,0,ws.c_str(),(int)ws.size(),nullptr,0,nullptr,nullptr); std::string out; out.resize(std::max(0,len)); if(len>0) WideCharToMultiByte(CP_UTF8,0,ws.c_str(),(int)ws.size(),out.data(),len,nullptr,nullptr); return out; }

// text draw small helper
#define OVL_DRAW_TEXT(_ctx,_txt,_x,_y,_size,_col,_fontName) do{ \
  ComPtr<IDWriteTextFormat> _fmt; \
  if(SUCCEEDED(CreateTextFormat(PickFont(_fontName), (_size), &_fmt))){ \
    D2D1_RECT_F _layout = RF((_x), (_y), 4000.f, 1200.f); \
    (_ctx)->DrawTextW((_txt).c_str(), (UINT32)(_txt).size(), _fmt.Get(), _layout, brush.Get()); \
  } \
}while(0)
