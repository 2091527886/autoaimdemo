#include "Aim.hpp"
#include "sh.hpp"
#include "ovl_util.hpp"
#include "aim_ovl.hpp"

void Aim::draw_ovl(double &ms) {
  if (!c.ovl) { ms = 0.0; return; }
  auto t0 = (c.timing ? tn() : tp{});
  gover.BeginFrame();
  // 绘制原始检测框（obs）
  if (c.ovl_show_obs){
    for (size_t i=0;i<ds.size();++i) { const auto &d = ds[i];
    if (c.ovl_only >= 0 && d.class_id != c.ovl_only)
      continue;
    float x = (float)(crop.x + d.box.x);
    float y = (float)(crop.y + d.box.y);
    float w = (float)d.box.width;
    float h = (float)d.box.height;
    int cid = d.class_id;
    int rr = c.ovl_r, gg = c.ovl_g, bb = c.ovl_b;
    if (c.ovl_cls && cid >= 0) {
      if (cid < (int)c.ovl_cr.size()) rr = c.ovl_cr[cid];
      if (cid < (int)c.ovl_cg.size()) gg = c.ovl_cg[cid];
      if (cid < (int)c.ovl_cb.size()) bb = c.ovl_cb[cid];
    }
    auto col = ARGB((uint8_t)c.ovl_a, (uint8_t)rr, (uint8_t)gg, (uint8_t)bb);
    gover.AddRectL({x, y, w, h}, col, (float)c.ovl_thick, (uint16_t)c.ovl_layer_obs);
    float fsz = (float)std::max(1, c.ovl_font_px);
    float ty = MX(0.0f, y - fsz - 2.0f);
    std::string lbl = c.name_or_id(cid);
    std::wstring wlbl(lbl.begin(), lbl.end());
    if (!wlbl.empty()) {
      OverlayColor tcol = ARGB(255, (uint8_t)rr, (uint8_t)gg, (uint8_t)bb);
      if (!c.ovl_font_name.empty()) gover.AddTextL(x, ty, wlbl, fsz, tcol, c.ovl_font_name, (uint16_t)c.ovl_layer_text);
      else gover.AddTextL(x, ty, wlbl, fsz, tcol, (uint16_t)c.ovl_layer_text);
    }
    }
  }

  // 独立绘制骨架（不受 ovl_show_obs 影响）
  if (c.ovl_show_pose){
    size_t n = std::min(pose_ok.size(), pose_kpts.size());
    for (size_t i=0; i<n; ++i){ if (!pose_ok[i]) continue; const auto& k = pose_kpts[i];
      OverlayColor pcol = ARGB(220, 0, 255, 255);
      auto vis = [&](int id){ return id<(int)k.size() && (!k[id].conf || *k[id].conf >= c.pose_kpt_conf_thresh); };
      auto kp  = [&](int id){ return std::pair<float,float>((float)(crop.x + k[id].x), (float)(crop.y + k[id].y)); };
      auto line = [&](int a,int b){ if (vis(a)&&vis(b)){ auto pa=kp(a), pb=kp(b); gover.AddLineL(pa.first, pa.second, pb.first, pb.second, (float)c.ovl_thick, pcol, (uint16_t)c.ovl_layer_pose); }};
      for (size_t ki=0; ki<k.size(); ++ki){ if (k[ki].conf && *k[ki].conf < c.pose_kpt_conf_thresh) continue; float px = (float)(crop.x + k[ki].x) - 1.5f; float py = (float)(crop.y + k[ki].y) - 1.5f; gover.FillRectL({px, py, 3.0f, 3.0f}, pcol, (uint16_t)c.ovl_layer_pose); }
      line(5,6); line(5,11); line(6,12); line(11,12);
      line(5,7); line(7,9); line(6,8); line(8,10);
      line(11,13); line(13,15); line(12,14); line(14,16);
      line(0,5); line(0,6);
    }
  }

  // 绘制跟踪框
  if (c.ovl_show_trk) {

    if (c.ovl_show_trk_all) {
      // 绘制全部 track 框与 id
      for (const auto& t : last_trks_) {
        int rr = c.ovl_r, gg = c.ovl_g, bb = c.ovl_b;
        if (c.ovl_cls && t.cls >= 0) {
          if (t.cls < (int)c.ovl_cr.size()) rr = c.ovl_cr[t.cls];
          if (t.cls < (int)c.ovl_cg.size()) gg = c.ovl_cg[t.cls];
          if (t.cls < (int)c.ovl_cb.size()) bb = c.ovl_cb[t.cls];
        }
        auto col = ARGB((uint8_t)c.ovl_a, (uint8_t)rr, (uint8_t)gg, (uint8_t)bb);
        // add crop offset to track box to align with obs drawing
        float x = (float)(crop.x + t.trk.x);
        float y = (float)(crop.y + t.trk.y);
        float w = (float)t.trk.width;
        float h = (float)t.trk.height;
        gover.AddRectL({x,y,w,h}, col, (float)c.ovl_thick, (uint16_t)c.ovl_layer_trk_all);
        std::string text = std::string("id=") + std::to_string(t.id) + " " + c.name_or_id(t.cls);
        std::wstring wtxt(text.begin(), text.end());
        float fsz = (float)std::max(1, c.ovl_font_px);
        float ty = MX(0.0f, y - fsz - 2.0f);
        OverlayColor tcol = ARGB(255, (uint8_t)rr, (uint8_t)gg, (uint8_t)bb);
        if (!c.ovl_font_name.empty()) gover.AddTextL(x, ty, wtxt, fsz, tcol, c.ovl_font_name, (uint16_t)c.ovl_layer_text);
        else gover.AddTextL(x, ty, wtxt, fsz, tcol, (uint16_t)c.ovl_layer_text);
      }
    }
    // 绘制当前选择的 track（锁定目标用黄色+加粗）
    if (cur_sel.valid()){
      int rr = c.ovl_r, gg = c.ovl_g, bb = c.ovl_b;
      if (c.ovl_cls && cur_sel.cls >= 0) {
        if (cur_sel.cls < (int)c.ovl_cr.size()) rr = c.ovl_cr[cur_sel.cls];
        if (cur_sel.cls < (int)c.ovl_cg.size()) gg = c.ovl_cg[cur_sel.cls];
        if (cur_sel.cls < (int)c.ovl_cb.size()) bb = c.ovl_cb[cur_sel.cls];
      }
      auto col = ARGB((uint8_t)c.ovl_a, (uint8_t)rr, (uint8_t)gg, (uint8_t)bb);
      float x = (float)(crop.x + cur_sel.trk.x), y = (float)(crop.y + cur_sel.trk.y), w = (float)cur_sel.trk.width, h = (float)cur_sel.trk.height;
      // 如果正在锁定，则用黄色加粗绘制 track 框
      if (lock_ && cur_sel.track_id == locked_track_id) {
        OverlayColor ycol = ARGB((uint8_t)c.lock_a, (uint8_t)c.lock_r, (uint8_t)c.lock_g, (uint8_t)c.lock_b);
        gover.AddRectL({x,y,w,h}, ycol, (float)c.lock_th, (uint16_t)c.ovl_layer_lock);
      } else {
        gover.AddRectL({x,y,w,h}, col, (float)c.ovl_thick, (uint16_t)c.ovl_layer_trk_sel);
      }
      // 显示 id 与 cls
      std::string text = std::string("id=") + std::to_string(cur_sel.track_id) + " " + c.name_or_id(cur_sel.cls);
      std::wstring wtxt(text.begin(), text.end());
      float fsz = (float)std::max(1, c.ovl_font_px);
      float ty = MX(0.0f, y - fsz - 2.0f);
      OverlayColor tcol = (lock_ && cur_sel.track_id == locked_track_id)
        ? ARGB(255, (uint8_t)c.lock_r, (uint8_t)c.lock_g, (uint8_t)c.lock_b)
        : ARGB(255, (uint8_t)rr, (uint8_t)gg, (uint8_t)bb);
      if (!c.ovl_font_name.empty()) gover.AddTextL(x, ty, wtxt, fsz, tcol, c.ovl_font_name, (uint16_t)(lock_ && cur_sel.track_id == locked_track_id ? c.ovl_layer_lock : c.ovl_layer_text));
      else gover.AddTextL(x, ty, wtxt, fsz, tcol, (uint16_t)(lock_ && cur_sel.track_id == locked_track_id ? c.ovl_layer_lock : c.ovl_layer_text));
    }
 }

  aim_draw_cross(this);
  aim_draw_lock(this);
  gover.EndFrame();
  gover.Tick();
  auto t1 = (c.timing ? tn() : tp{});
  ms = c.timing ? msd(t1 - t0).count() : 0.0;
}
