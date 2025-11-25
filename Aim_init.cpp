#include "Aim.hpp"
#include "AdrcConfIO.hpp"
Aim::Aim()
    : c(), cap(c, c.crop_size_px, c.crop_size_px), pose(c), hk(c),
      crop((cap.w() - c.crop_size_px) / 2, (cap.h() - c.crop_size_px) / 2, c.crop_size_px, c.crop_size_px),
      ctr(c.crop_size_px / 2, c.crop_size_px / 2),
      lat(static_cast<size_t>(std::max(1, c.log_latency_every_n))), fc(c) {
  if (cap.w() < c.crop_size_px || cap.h() < c.crop_size_px) {
    throw std::runtime_error("Screen resolution is smaller than configured crop_size_px.");
  }
  mfac = (c.hfov_deg / (double)cap.w()) * (c.px_per_360 / 360.0);
  LO << "Detect center " << c.crop_size_px << "x" << c.crop_size_px << END;
  // 在控制器初始化前尝试加载外部 ADRC 配置
  (void)LoadAdrcConfIfPresent("adrc.conf", c);
  c.validate();
  adrc.setConfig(c);
  dz.cfg(c.dz_px, c.dz_hys);
  // 初始化 HumanCursor（默认开启，读取配置）
  {
    HumanCursorConfig hcfg; hcfg.enable = c.human_enable; hcfg.steady = c.human_steady; hcfg.queue_cap = (size_t)c.human_queue_cap; hcfg.fps_for_duration = c.fps;
    // 映射 Config → HumanCursor 随机参数
    hcfg.rand = c.to_hc_curve_rand_config();
    human.init(hcfg);
  }
  if (c.ovl){ gover.UseVirtualScreen(); gover.SetConfig(c); bool ok = gover.Start(); lg::ok_or_werr(P_B, "Overlay.Start", ok); }
  ocs.configure(c);
}

Aim::~Aim() {
  if (c.ovl)
    gover.Stop();
}
