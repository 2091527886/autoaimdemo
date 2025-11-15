#include "pch.hpp"
#include "ObjDet.hpp"
#include "Timing.hpp"

#include "trtyolo_inc.hpp"

using trtyolo::DetectModel;

void objdet_perf(DetectModel* model,
                 metrics::TrtPerfReporting& perf,
                 bool timing,
                 bool perf_report,
                 Timing& tm) {
  if (timing && perf_report && perf.ShouldReport()) {
    auto report = model->performanceReport();
    const std::string &tp = std::get<0>(report);
    if (!tp.empty()) {
      tm.trt_throughput = tp;
      tm.trt_cpu_lat = std::get<1>(report);
      tm.trt_gpu_lat = std::get<2>(report);
    }
  }
}
