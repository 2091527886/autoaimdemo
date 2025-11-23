#include "AdrcConfIO.hpp"
#include "Config.hpp"
#include "Log.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace {
    static inline std::string trim(const std::string& s){
        size_t b = 0, e = s.size();
        while (b < e && (unsigned char)s[b] <= ' ') ++b;
        while (e > b && (unsigned char)s[e-1] <= ' ') --e;
        return s.substr(b, e-b);
    }
    static inline bool ieq(const std::string& a, const std::string& b){
        if (a.size()!=b.size()) return false;
        for(size_t i=0;i<a.size();++i){
            unsigned char ua = static_cast<unsigned char>(a[i]);
            unsigned char ub = static_cast<unsigned char>(b[i]);
            char ca = static_cast<char>(std::tolower(ua));
            char cb = static_cast<char>(std::tolower(ub));
            if(ca!=cb) return false;
        }
        return true;
    }
    static inline bool parse_bool(const std::string& v, bool& out){
        if (ieq(v, "true") || v=="1") { out = true;  return true; }
        if (ieq(v, "false")|| v=="0") { out = false; return true; }
        return false;
    }
    template<typename T>
    static inline bool parse_num(const std::string& v, T& out){
        std::istringstream is(v); is >> out; return is && is.eof();
    }
}

bool LoadAdrcConfIfPresent(const std::string& path, Config& c){
    std::ifstream fs(path); if(!fs.is_open()){ LO << "adrc.conf not found, using built-in defaults" << END; return false; }
    int updated = 0; std::string line; int ln = 0;
    while(std::getline(fs, line)){
        ++ln; std::string t = trim(line);
        if (t.empty()) continue;
        if (t[0]=='#' || t[0]==';') continue;
        auto pos = t.find('='); if (pos==std::string::npos){ LO << "adrc.conf:" << ln << ": missing '='" << END; continue; }
        std::string key = trim(t.substr(0,pos));
        std::string val = trim(t.substr(pos+1));
        if (key.empty()) continue;
        bool bok=false; int iok=0; double dok=0.0;
        auto upd_bool=[&](bool& ref){ if (parse_bool(val, bok)){ ref=bok; ++updated; } else { LO<<"adrc.conf:"<<ln<<": invalid bool for '"<<key<<"'"<<END; } };
        auto upd_int=[&](int& ref){ if (parse_num<int>(val, iok)){ ref=iok; ++updated; } else { LO<<"adrc.conf:"<<ln<<": invalid int for '"<<key<<"'"<<END; } };
        auto upd_double=[&](double& ref){ if (parse_num<double>(val, dok)){ ref=dok; ++updated; } else { LO<<"adrc.conf:"<<ln<<": invalid double for '"<<key<<"'"<<END; } };

        if (key=="adrc_enable") upd_bool(c.adrc_enable);
        else if (key=="adrc_use_measured_dt") upd_bool(c.adrc_use_measured_dt);
        else if (key=="adrc_ctrl_low_level") upd_int(c.adrc_ctrl_low_level);
        else if (key=="adrc_invert_output_sign") upd_bool(c.adrc_invert_output_sign);
        else if (key=="adrc_output_limit_px") upd_double(c.adrc_output_limit_px);
        else if (key=="adrc_td_bandwidth_r") upd_double(c.adrc_td_bandwidth_r);
        else if (key=="adrc_model_gain_b0") upd_double(c.adrc_model_gain_b0);
        else if (key=="adrc_fal_delta") upd_double(c.adrc_fal_delta);
        else if (key=="adrc_eso_beta01") upd_double(c.adrc_eso_beta01);
        else if (key=="adrc_eso_beta02") upd_double(c.adrc_eso_beta02);
        else if (key=="adrc_eso_beta03") upd_double(c.adrc_eso_beta03);
        else if (key=="adrc_nlsef_beta1") upd_double(c.adrc_nlsef_beta1);
        else if (key=="adrc_nlsef_beta2") upd_double(c.adrc_nlsef_beta2);
        else if (key=="adrc_fal_alpha1") upd_double(c.adrc_fal_alpha1);
        else if (key=="adrc_fal_alpha2") upd_double(c.adrc_fal_alpha2);
        else {
            // ignore unknown
        }
    }
    if (updated>0){ c.validate(); LO << "adrc.conf loaded, updated=" << updated << END; return true; }
    LO << "adrc.conf parsed, no recognized keys; keep defaults" << END; return false;
}
