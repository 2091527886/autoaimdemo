#include "Aim.hpp"
#include "pch.hpp"
#include "sh.hpp"

double Aim::get_dt() {
  if (c.prop_use_dt){ auto now = tn(); double d = std::chrono::duration<double>(now - last_dt_ts).count(); last_dt_ts = now; return d; }
  else { return std::max(0.001, 16.0 / 1000.0); }
}
