#include "Aim.hpp"
#include "Log.hpp"
#include "pch.hpp"
Aim::Aim()
    : c(), cap(c, c.crop_size_px, c.crop_size_px), od(c), hk(c),
      crop((cap.w() - c.crop_size_px) / 2, (cap.h() - c.crop_size_px) / 2, c.crop_size_px, c.crop_size_px),
      ctr(c.crop_size_px / 2, c.crop_size_px / 2),
      lat(static_cast<size_t>(std::max(1, c.log_latency_every_n))), fc(c) {
  if (cap.w() < c.crop_size_px || cap.h() < c.crop_size_px) {
    throw std::runtime_error("Screen resolution is smaller than configured crop_size_px.");
  }
  mfac = (c.hfov_deg / static_cast<double>(cap.w())) * (c.px_per_360 / 360.0);
  lg::o() << "Detect center " << c.crop_size_px << "x" << c.crop_size_px << lg::endl;
  algo.setConfig(c);
  if (c.wind_rng_seed > 0) {
    wind_.rng.seed(static_cast<uint32_t>(c.wind_rng_seed));
  }
  if (c.ovl) {
    gover.UseVirtualScreen();
    gover.SetConfig(c);
    bool ok = gover.Start();
    lg::ok_or_werr(P_B, "Overlay.Start", ok);
  }
}

Aim::~Aim() {
  if (c.ovl)
    gover.Stop();
}
