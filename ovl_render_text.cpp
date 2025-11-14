#include "ovl_impl.hpp"
#include "pch.hpp"

HRESULT Game_overlay::Impl::CreateTextFormat(const std::wstring& font,float size,IDWriteTextFormat** out){
  uint64_t key = (std::hash<std::wstring>{}(font) ^ (uint64_t(std::lround(size*100))<<1));
  auto it=textFormatCache.find(key); if(it!=textFormatCache.end()){ *out=it->second.Get(); (*out)->AddRef(); return S_OK; }
  ComPtr<IDWriteTextFormat> fmt;
  if(FAILED(dwFactory->CreateTextFormat(font.c_str(), nullptr, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, size, L"en-US", &fmt))) return E_FAIL;
  fmt->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP);
  textFormatCache.emplace(key, fmt);
  *out = fmt.Detach();
  return S_OK;
}
