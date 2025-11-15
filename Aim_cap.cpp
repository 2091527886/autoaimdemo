#include "Aim.hpp"
#include "sh.hpp"

bool Aim::capture(FrameLat &fl, ScreenCapturer::CapturedView &view) {
  double cap_ms = 0.0; bool got = false;
  if (c.timing) {
    auto t0 = tn();
    got = cap.capture_view(view, crop);
    auto t1 = tn();
    cap_ms = msd(t1 - t0).count();
  } else {
    got = cap.capture_view(view, crop);
  }
  if (!got) {
    fl.cap_ms = cap_ms;
    if (c.timing && c.log_latency && !c.latency_on_capture_only) {
      lat.push(fl);
    }
    return false;
  }
  c.frame_idx++;
  fl.idx = c.frame_idx;
  fl.cap_ms = cap_ms;
  return true;
}

bool Aim::infer_view(ScreenCapturer::CapturedView &view) {
  if (view.mapped && view.data) {
    od.detect_view(view.data, view.width, view.height, view.pitch, ds, c, tm);
    cap.release_view(view);
    return true;
  }
  cap.release_view(view);
  return false;
}
