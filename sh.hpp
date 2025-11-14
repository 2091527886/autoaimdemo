#pragma once
// Tiny shared helpers for timing aliases与小宏，仅为减少源码体积，不改逻辑
#include <chrono>

using clk = std::chrono::high_resolution_clock;
using tp  = clk::time_point;
using msd = std::chrono::duration<double, std::milli>;
inline tp tn(){ return clk::now(); }

// timing helpers
#define TSTART(v,cond) auto v = (cond) ? tn() : tp{}
#define TELAPSE(dst,cond,st) do{ if(cond){ (dst)=msd(tn()-(st)).count(); } }while(0)

// simple retry helper: run body up to n times, return true on first success
template<class F>
inline bool try_n(int n, F&& body){ for(int i=0;i<n;++i){ if(body()) return true; } return false; }
