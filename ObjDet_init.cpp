#include "ObjDet.hpp"
#include "trtyolo_inc.hpp"
#include "Config.hpp"
#include "objdet_helpers.hpp"
using trtyolo::DetectModel;
using trtyolo::InferOption;

ObjDet::ObjDet(const Config &cfg)
    : perf_(cfg) {
  const std::string engine = cfg.resolve_engine_path();
  LO << P_I "Load TRT: " << engine << END;

  InferOption opt;
  if (cfg.trt_swap_rb)
    opt.set_rb();
  opt.setBorderValue(cfg.trt_border_value);
  if (cfg.trt_use_normalize && cfg.trt_mean.size() == 3 && cfg.trt_std.size() == 3) {
    opt.setNormalizeParams(cfg.trt_mean, cfg.trt_std);
  }
  if (cfg.trt_cuda_mem)
    opt.enableCudaMem();
  if (cfg.trt_managed_mem)
    opt.enableManagedMemory();

  if (cfg.trt_input_w > 0 && cfg.trt_input_h > 0) {
    opt.setInputDimensions(cfg.trt_input_w, cfg.trt_input_h);
  } else {
    opt.setInputDimensions(cfg.crop_size_px, cfg.crop_size_px);
  }
  if (cfg.perf_report) {
    opt.enablePerformanceReport();
  }

  {
    char cwd_buffer[MAX_PATH];
    if (GetCurrentDirectoryA(MAX_PATH, cwd_buffer)) {
      LO << P_D "cwd: " << cwd_buffer << END;
    }
    DWORD attrs = GetFileAttributesA(engine.c_str());
    if (attrs == INVALID_FILE_ATTRIBUTES) {
      LE << P_E "Engine not found: '" << engine << "'" << END;
    } else if (attrs & FILE_ATTRIBUTE_DIRECTORY) {
      LE << P_E "Engine is dir: '" << engine << "'" << END;
    } else {
      LO << P_D "Engine found: '" << engine << "'" << END;
    }
  }

  int final_bx = cfg.lb_block_x;
  int final_by = cfg.lb_block_y;
  bool final_use_lb = cfg.lb_use_launch_bounds;
  int final_minb = cfg.lb_min_blocks_per_sm;
  bool final_use_occ = cfg.lb_use_occupancy;
  if (!cfg.lb_block_cfg_path.empty()){
    std::ifstream fin(cfg.lb_block_cfg_path);
    if (fin){
      std::string line;
      while (std::getline(fin, line)){
        trim_str(line);
        if (line.empty() || line[0]=='#' || line[0]==';') continue;
        auto pos = line.find('='); if (pos==std::string::npos) continue;
        std::string k = line.substr(0,pos), v = line.substr(pos+1);
        trim_str(k); trim_str(v);
        if (k=="block_x") final_bx = std::stoi(v);
        else if (k=="block_y") final_by = std::stoi(v);
        else if (k=="use_launch_bounds") final_use_lb = (std::stoi(v)!=0);
        else if (k=="min_blocks_per_sm") final_minb = std::stoi(v);
        else if (k=="use_occupancy") final_use_occ = (std::stoi(v)!=0);
      }
    }
  }
  InferOption &opt_ref = opt;
  opt_ref.setLetterboxBlockXY(final_bx, final_by);
  opt_ref.setLetterboxLaunchBounds(final_use_lb, final_minb);
  opt_ref.enableLetterboxOccupancy(final_use_occ);
  LO << P_I "Letterbox kernel config: block_x=" << final_bx
          << ", block_y=" << final_by
          << ", min_blocks_per_sm=" << final_minb
          << ", use_launch_bounds=" << (final_use_lb ? "1" : "0")
          << ", use_occupancy=" << (final_use_occ ? "1" : "0")
          << END;

  try {
    model_ = std::make_unique<DetectModel>(engine.c_str(), opt);
    LO << P_I "Engine OK" << END;
  } catch (const std::exception &e) {
    std::ostringstream oss;
    oss << P_F "DetectModel failed: '" << engine << "'\nerr: " << e.what();
    throw std::runtime_error(oss.str());
  }
}

ObjDet::~ObjDet() = default;
