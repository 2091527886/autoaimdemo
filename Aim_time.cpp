#include "Aim.hpp"
#include "pch.hpp"
#include "sh.hpp"

double Aim::get_dt() {
  // 方案A：统一 dt 来源：始终使用“实测帧间隔 dt”。
  // 说明：
  // - 本函数不再受 prop_use_dt/adrc_use_dt 影响，也不受 fps_lock 影响；始终返回实测 dt。
  // - 各控制器内部（P/ADRC）可各自决定是否“使用 dt”（例如 ADRC 的 adrc_use_dt=false 时强制使用 16ms）。
  auto now = tn();
  double d = std::chrono::duration<double>(now - last_dt_ts).count();
  last_dt_ts = now;
  return d;
}
