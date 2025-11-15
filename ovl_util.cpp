#include "pch.hpp"
#include "ovl_impl.hpp"

void GetVirtualScreen(int& x, int& y, int& w, int& h){ x=GetSystemMetrics(SM_XVIRTUALSCREEN); y=GetSystemMetrics(SM_YVIRTUALSCREEN); w=GetSystemMetrics(SM_CXVIRTUALSCREEN); h=GetSystemMetrics(SM_CYVIRTUALSCREEN); }

void SetPerMonitorV2DpiAwareness(){ HMODULE user32=GetModuleHandleW(L"user32.dll"); if(!user32) return; using Fn=BOOL(WINAPI*)(HANDLE); if(auto p=reinterpret_cast<Fn>(GetProcAddress(user32, "SetProcessDpiAwarenessContext"))) p(reinterpret_cast<HANDLE>(-4)); }

D2D1_COLOR_F ToD2DColor(OverlayColor argb){ float a=float((argb>>24)&0xFF)/255.f, r=float((argb>>16)&0xFF)/255.f, g=float((argb>>8)&0xFF)/255.f, b=float((argb>>0)&0xFF)/255.f; return D2D1::ColorF(r,g,b,a); }
