#include "ovl_gpu.hpp"
#include "ovl_impl.hpp"
using Microsoft::WRL::ComPtr;
using std::vector;
using std::wstring;

static std::string ws_to_utf8(const std::wstring& ws){ if(ws.empty()) return {}; int len=WideCharToMultiByte(CP_UTF8,0,ws.c_str(),(int)ws.size(),nullptr,0,nullptr,nullptr); std::string out; if(len>0){ out.resize(len); WideCharToMultiByte(CP_UTF8,0,ws.c_str(),(int)ws.size(),out.data(),len,nullptr,nullptr);} return out; }

struct GpuInfo {
  ComPtr<IDXGIAdapter1> adp;
  DXGI_ADAPTER_DESC1 desc{};
  int idx = -1;
};

static wstring wlower(wstring s){ std::transform(s.begin(), s.end(), s.begin(), [](wchar_t c){ return std::towlower(c); }); return s; }

static bool parse_luid(const wstring &s, LUID &out){ size_t p=s.find(L':'); if (p==wstring::npos) return false; auto sh=s.substr(0,p), sl=s.substr(p+1); auto to_u32=[](const wstring &t, DWORD &v)->bool{ try{ size_t idx=0; int base=10; if(t.size()>2 && (t.rfind(L"0x",0)==0 || t.rfind(L"0X",0)==0)) base=16; v=(DWORD)std::stoul(t,&idx,base); return idx==t.size(); }catch(...){ return false; } }; DWORD hi=0, lo=0; if(!to_u32(sh,hi) || !to_u32(sl,lo)) return false; out.HighPart=(LONG)hi; out.LowPart=(DWORD)lo; return true; }

static vector<GpuInfo> list_adapters() {
  std::vector<GpuInfo> v;
  ComPtr<IDXGIFactory1> fac;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&fac))))
    return v;
  for (UINT i=0;; ++i){ ComPtr<IDXGIAdapter1> a; DXGI_ADAPTER_DESC1 d{}; if (fac->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break; if (!a) break; if (FAILED(a->GetDesc1(&d))) continue; if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue; v.push_back({a,d,(int)i}); }
  return v;
}

static void log_adapters(const vector<GpuInfo> &v) {
  for (auto &g : v) {
    wstring name(g.desc.Description);
    double mb = (double)g.desc.DedicatedVideoMemory / (1024.0 * 1024.0);
    lg::o() << P_I "OVL GPU[" << g.idx << "] "
            << ws_to_utf8(name)
            << " | ven=0x" << std::hex << std::uppercase << g.desc.VendorId
            << " dev=0x" << g.desc.DeviceId << std::dec << std::nouppercase
            << " | memMB=" << (int)mb
            << " | luid=" << g.desc.AdapterLuid.HighPart << ":" << g.desc.AdapterLuid.LowPart
            << lg::endl;
  }
}

ComPtr<IDXGIAdapter1> pick_adapter_cfg(const wstring &name,
                                       const wstring &luid,
                                       int idx,
                                       bool do_enum) {
  auto v = list_adapters();
  if (do_enum) log_adapters(v);
  if (v.empty()) return {};
  if (!name.empty()){ auto kn=wlower(name); for (auto &g: v){ wstring dn(g.desc.Description); if (wlower(dn).find(kn) != wstring::npos) return g.adp; } }
  if (!luid.empty()) {
    LUID t{};
    if (parse_luid(luid, t)) {
      for (auto &g : v) {
        if (g.desc.AdapterLuid.HighPart == t.HighPart && g.desc.AdapterLuid.LowPart == t.LowPart)
          return g.adp;
      }
    }
  }
  if (idx >= 0){ for (auto &g: v) if (g.idx == idx) return g.adp; }
  return {};
}
