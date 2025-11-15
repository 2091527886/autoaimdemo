#include "FpsCtl.hpp"
#include "pch.hpp"

using clk = std::chrono::steady_clock;
using msd = std::chrono::duration<double, std::milli>;

FpsCtl::FpsCtl(const Config &cfg) {
  tgt_ms = 1000.0 / std::max(1, cfg.fps);
  per = std::chrono::duration_cast<clk::duration>(msd(tgt_ms));
  next_tp = clk::now() + per;
}

double FpsCtl::end() {
  auto now = clk::now();
  if (now < next_tp){ std::this_thread::sleep_until(next_tp); ovr=0.0; next_tp+=per; return tgt_ms; }
  ovr = msd(now - next_tp).count();
  while (next_tp <= now){ next_tp += per; ++drop; }
  return tgt_ms;
}
