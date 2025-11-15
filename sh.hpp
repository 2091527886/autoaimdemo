#pragma once
// Tiny shared helpers for timing aliases与小宏，仅为减少源码体积，不改逻辑
#include <chrono>
#include <algorithm>
#include <cmath>

using clk = std::chrono::high_resolution_clock;
using tp  = clk::time_point;
using msd = std::chrono::duration<double, std::milli>;
inline tp tn(){ return clk::now(); }

// timing helpers
#define TSTART(v,cond) auto v = (cond) ? tn() : tp{}
#define TELAPSE(dst,cond,st) do{ if(cond){ (dst)=msd(tn()-(st)).count(); } }while(0)

// math short aliases (no logic change)
#define CLP(x,a,b) (std::clamp((x),(a),(b)))
#define MN(a,b) (std::min((a),(b)))
#define MX(a,b) (std::max((a),(b)))
#define HYP(x,y) (std::hypot((x),(y)))
#define LR(x) (std::lround((x)))

inline void unit(double x, double y, double& ux, double& uy){ double d=HYP(x,y); if(d>1e-6){ ux=x/d; uy=y/d; } else { ux=0.0; uy=0.0; } }
inline double bell_curve(double t, double curve){ t=CLP(t,0.0,1.0); curve=CLP(curve,0.0,1.0); double b=(1.0-curve)*t + curve*(1.0-std::fabs(2.0*t-1.0)); return CLP(b,0.1,1.0); }
inline double step_len_px(double dist,double bell,double mn,double mx){ return CLP(dist*0.25*bell,mn,mx); }
inline void backtrack(double &sx,double &sy,double wantx,double wanty,double k){ sx -= k * (wantx - sx); sy -= k * (wanty - sy); }
inline void minstep(int &dx,int &dy,double wantx,double wanty){ if(dx==0 && dy==0){ dx=(wantx>0)?1:((wantx<0)?-1:0); dy=(wanty>0)?1:((wanty<0)?-1:0);} }

// simple retry helper: run body up to n times, return true on first success
template<class F>
inline bool try_n(int n, F&& body){ for(int i=0;i<n;++i){ if(body()) return true; } return false; }

// common tiny helpers to reduce code duplication (no logic changes)
inline int upd_seg(int seg, double h, double near_h, double far_h, double hys){
  if (seg == 2) { if (h < near_h - hys) seg = 1; }
  else if (seg == 0) { if (h > far_h + hys) seg = 1; }
  else { if (h >= near_h) seg = 2; else if (h <= far_h) seg = 0; }
  return seg;
}
inline double lerp(double a, double b, double t){ return a + (b - a) * t; }
