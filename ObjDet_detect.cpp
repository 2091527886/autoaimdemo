#include "ObjDet.hpp"
#include "pch.hpp"

#include "Config.hpp"
#include "Timing.hpp"
#include "sh.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251)
#endif
#include <trtyolo.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// perf helper
void objdet_perf(trtyolo::DetectModel* model,
                 metrics::TrtPerfReporting& perf,
                 bool timing,
                 bool perf_report,
                 Timing& tm);

using trtyolo::DetectRes;
using trtyolo::Image;

void ObjDet::detect_view(unsigned char *data,
                         int width,
                         int height,
                         size_t pitch,
                         std::vector<Detection> &out,
                         const Config &cfg,
                         Timing &tm) {
  out.clear();

  auto fill = [&](const trtyolo::DetectRes &res) {
    if (res.num > 0 && res.boxes.size() == static_cast<size_t>(res.num)) {
      for (int i = 0; i < res.num; ++i) {
        float score = (i < (int)res.scores.size() ? res.scores[i] : 0.0f);
        if (score < cfg.conf_thresh)
          continue;
        int cls = (i < (int)res.classes.size() ? res.classes[i] : -1);
        const auto &b = res.boxes[i];
        int x = std::max(0, static_cast<int>(std::floor(b.left)));
        int y = std::max(0, static_cast<int>(std::floor(b.top)));
        int w = std::max(0, static_cast<int>(std::round(b.right - b.left)));
        int h = std::max(0, static_cast<int>(std::round(b.bottom - b.top)));
        w = std::min(w, width - x);
        h = std::min(h, height - y);
        if (w > 0 && h > 0)
          out.emplace_back(Detection{cv::Rect(x, y, w, h), score, cls});
      }
    }
  };

  if (cfg.timing) {
    auto t0 = tn();
    Image input(data, width, height, pitch);
    DetectRes res = model_->predict(input);
    auto t1 = tn();
    tm.infer_ms = msd(t1 - t0).count();
    auto t2 = tn();
    fill(res);
    auto t3 = tn();
    tm.post_ms = msd(t3 - t2).count();
  } else {
    Image input(data, width, height, pitch);
    DetectRes res = model_->predict(input);
    fill(res);
    tm.infer_ms = 0.0;
    tm.post_ms = 0.0;
  }

  objdet_perf(model_.get(), perf_, cfg.timing, cfg.perf_report, tm);
}
