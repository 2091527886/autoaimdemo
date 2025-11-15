#pragma once
#include <d3d11.h>
#include <opencv2/core.hpp>
#include "sh.hpp"
#include "ScreenCapturer.hpp"

inline D3D11_BOX box_from_rect(const cv::Rect& r){ D3D11_BOX b{}; b.left=r.x; b.top=r.y; b.right=r.x+r.width; b.bottom=r.y+r.height; b.front=0; b.back=1; return b; }

inline bool map_stage(ID3D11DeviceContext* ctx, ID3D11Texture2D* stage0, ScreenCapturer::CapturedView& view, bool do_tim, double& mp){
  D3D11_MAPPED_SUBRESOURCE mr; TSTART(t_map_s, do_tim);
  HRESULT hrm = ctx->Map(stage0, 0, D3D11_MAP_READ, 0, &mr);
  TELAPSE(mp, do_tim, t_map_s);
  if(FAILED(hrm)) return false;
  view.data = static_cast<unsigned char*>(mr.pData);
  view.pitch = mr.RowPitch; view.mapped = true; return true;
}

// Copy src->dst (with optional region and flush), then map dst into view; accumulate cp/mp
inline bool cpmp(ID3D11DeviceContext* ctx,
                 ID3D11Texture2D* dst,
                 ID3D11Texture2D* src,
                 const D3D11_BOX* region,
                 ScreenCapturer::CapturedView& view,
                 bool do_tim,
                 double& cp,
                 double& mp,
                 bool flush_after_copy){
  TSTART(t_copy_s, do_tim);
  ctx->CopySubresourceRegion(dst, 0, 0, 0, 0, src, 0, region);
  TELAPSE(cp, do_tim, t_copy_s);
  if(flush_after_copy) ctx->Flush();
  return map_stage(ctx, dst, view, do_tim, mp);
}
