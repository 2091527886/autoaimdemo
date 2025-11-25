#include "Aim.hpp"
#include "pch.hpp"
#include "sh.hpp"

double Aim::get_dt() {
  // 方案A：统一 dt 来源：始终使用“实测帧间隔 dt”。
  // 说明：
  // - 本函数不受 fps_lock 影响；始终返回实测 dt。
  // - ADRC 内部可由配置 adrc_use_measured_dt 决定是否使用实测 dt（false 时内部按 16ms）。
  auto now = tn();
  double d = std::chrono::duration<double>(now - last_dt_ts).count();
  last_dt_ts = now;
  return d;
}
