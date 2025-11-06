#pragma once

// 简洁比例控制 + 一阶低通 + 限幅 + 死区
class PropAlgo {
public:
    PropAlgo() = default;
    PropAlgo(double k, double out_lim_px, double tc_ms, double dead_px)
        : k_(k), out_lim_(out_lim_px), tc_(ms2s(tc_ms)), dz_(dead_px) {}

    void setK(double k) { k_ = k; }
    void setLim(double out_lim_px) { out_lim_ = (out_lim_px >= 0.0 ? out_lim_px : -out_lim_px); }
    void setTc(double ms) { tc_ = ms2s(ms); }
    void setDz(double px) { dz_ = (px >= 0.0 ? px : -px); }

    void reset() { y_ = 0.0; init_ = false; }

    // err: 像素误差，dt: 秒（<=0 时回退为典型帧间隔约 1/60）
    double step(double err, double dt) {
        if (!(dt > 0.0)) dt = 1.0 / 60.0;
        // 死区
        double e = err;
        if (dz_ > 0.0) {
            if (e > -dz_ && e < dz_) e = 0.0;
        }
        // 比例 + 低通（目标 raw -> 输出 y）
        const double raw = k_ * e;
        if (!init_) { y_ = raw; init_ = true; }
        const double a = (tc_ > 0.0) ? (dt / (tc_ + dt)) : 1.0; // tc<=0 则无平滑
        y_ = y_ + a * (raw - y_);
        // 限幅（对称）
        if (out_lim_ > 0.0) {
            if (y_ >  out_lim_) y_ =  out_lim_;
            if (y_ < -out_lim_) y_ = -out_lim_;
        }
        return y_;
    }

private:
    static double ms2s(double ms) { return (ms > 0.0 ? ms / 1000.0 : 0.0); }

    double k_ = 0.16;       // 比例
    double out_lim_ = 400.0; // 每帧像素限幅（与 PID 对齐默认）
    double tc_ = 0.006;     // 平滑时间常数（秒）
    double dz_ = 0.5;       // 死区（像素）

    double y_ = 0.0;
    bool init_ = false;
};
