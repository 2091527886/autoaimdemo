#include "ByteTrack.hpp"
#include <algorithm>
#include <limits>
#include <cmath>

namespace bt {

KF8::KF8() {
    // dt = 1 (frame) model
    F = cv::Matx<double,8,8>::eye();
    for (int i=0;i<4;++i) F(i, i+4) = 1.0; // position += velocity
    H = cv::Matx<double,4,8>::zeros();
    for (int i=0;i<4;++i) H(i,i) = 1.0; // observe position components
    Q = cv::Matx<double,8,8>::eye() * 1e-3; // smaller process noise to reduce jitter
    R = cv::Matx<double,4,4>::eye() * 2e-1; // larger obs noise balances detector noise
    P = cv::Matx<double,8,8>::eye();
}

void KF8::init(const cv::Rect& box) {
    double x0 = box.x + box.width / 2.0;
    double y0 = box.y + box.height / 2.0;
    double a0 = (double)box.width / std::max(1.0, (double)box.height); // aspect
    double h0 = (double)box.height;
    x = { x0,y0,a0,h0, 0,0,0,0 };
    inited = true;
}

void KF8::predict() {
    if (!inited) return;
    x = F * x;
    P = F * P * F.t() + Q;
}

cv::Matx<double,4,1> KF8::toZ(const cv::Rect& box) const {
    cv::Matx<double,4,1> z;
    z(0) = box.x + box.width / 2.0;
    z(1) = box.y + box.height / 2.0;
    z(2) = (double)box.width / std::max(1.0, (double)box.height);
    z(3) = (double)box.height;
    return z;
}

void KF8::update(const cv::Rect& box) {
    if (!inited) { init(box); return; }
    auto z = toZ(box);
    auto y = z - H * x;
    auto S = H * P * H.t() + R;
    auto K = P * H.t() * S.inv(cv::DECOMP_SVD);
    x = x + K * y;
    P = (cv::Matx<double,8,8>::eye() - K * H) * P;
}

double KF8::maha(const cv::Rect& box) const {
    auto z = toZ(box);
    auto y = z - H * x; // innovation
    auto S = H * P * H.t() + R;
    cv::Matx<double,1,1> m = y.t() * S.inv(cv::DECOMP_SVD) * y;
    return m(0);
}

cv::Rect KF8::toRect() const {
    double cx = x(0), cy = x(1), a = x(2), h = x(3);
    double w = std::max(1.0, a * h);
    double x0 = cx - w/2.0;
    double y0 = cy - h/2.0;
    return cv::Rect((int)std::round(x0),(int)std::round(y0),(int)std::round(w),(int)std::round(h));
}

static void split_by_score(const std::vector<Detection>& dets, float high, float low, std::vector<int>& hi, std::vector<int>& lo) {
    for (int i=0;i<(int)dets.size();++i) {
        if (dets[i].confidence >= high) hi.push_back(i);
        else if (dets[i].confidence >= low) lo.push_back(i);
    }
}

double ByteTracker::IoU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width,  b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    int w = std::max(0, x2 - x1);
    int h = std::max(0, y2 - y1);
    int inter = w * h;
    int uni = a.area() + b.area() - inter;
    if (uni <= 0) return 0.0;
    return (double)inter / (double)uni;
}

// Simple Hungarian (O(n^3)), minimal external deps; cost is 1-IoU so lower is better
std::vector<int> ByteTracker::Hungarian(const std::vector<std::vector<double>>& cost) {
    int n = (int)cost.size();
    if (n==0) return {};
    int m = (int)cost[0].size();
    int dim = std::max(n, m);
    std::vector<std::vector<double>> a(dim, std::vector<double>(dim, 1.0));
    for (int i=0;i<n;++i) for (int j=0;j<m;++j) a[i][j] = cost[i][j];
    // potentials and matching
    std::vector<double> u(dim+1), v(dim+1);
    std::vector<int> p(dim+1), way(dim+1);
    for (int i=1;i<=dim;i++) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(dim+1, std::numeric_limits<double>::infinity());
        std::vector<char> used(dim+1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = std::numeric_limits<double>::infinity();
            for (int j=1;j<=dim;j++) if (!used[j]) {
                double cur = a[i0-1][j-1] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j=0;j<=dim;j++) if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else { minv[j] -= delta; }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    std::vector<int> ans(n, -1);
    for (int j=1;j<=dim;j++) if (p[j] && p[j] <= n && j <= m) ans[p[j]-1] = j-1;
    return ans;
}

bt::ByteTracker::ByteTracker(const Config& c) {
    p.maha_th = c.trk_maha;
    p.out_nms = c.trk_out_nms;
    p.out_nms_iou = c.trk_out_nms_iou;
    p.high_th = c.trk_th;
    p.low_th  = c.trk_th_lo;
    p.iou_th  = c.trk_iou;
    p.max_age = std::max(1, c.trk_max_age);
    p.min_hits = std::max(0, c.trk_min_hits);
    use_bt = c.trk_bt;
    use_kf = c.trk_kf;
}

void bt::ByteTracker::update(const std::vector<Detection>& dets, std::vector<Detection>& out) {
    out.clear();
    if (!use_bt) { out = dets; return; }
    // 1) Predict existing tracks
    for (auto& t : tracks) { if (use_kf) t.kf.predict(); t.age++; t.time_since_update++; }

    // 2) Split by score
    std::vector<int> hi, lo; hi.reserve(dets.size()); lo.reserve(dets.size());
    split_by_score(dets, p.high_th, p.low_th, hi, lo);

    auto associate = [&](const std::vector<int>& det_idx_vec){
        if (tracks.empty() || det_idx_vec.empty()) return std::vector<std::pair<int,int>>{};
        std::vector<std::vector<double>> cost(tracks.size(), std::vector<double>(det_idx_vec.size(), 1.0));
        for (size_t i=0;i<tracks.size();++i) {
            for (size_t j=0;j<det_idx_vec.size();++j) {
                const auto& d = dets[det_idx_vec[j]];
                if (p.cls_match && d.class_id>=0 && tracks[i].class_id>=0 && d.class_id!=tracks[i].class_id) {
                    cost[i][j] = 1.0; // disallow by high cost
                } else {
                    // IoU (+ optional Mahalanobis gating)
                    double iou = IoU(use_kf ? tracks[i].kf.toRect() : tracks[i].last_box, d.box);
                    if (use_kf) {
                        double md = tracks[i].kf.maha(d.box);
                        if (md > p.maha_th) { cost[i][j] = 1.0; }
                        else { cost[i][j] = 1.0 - iou; }
                    } else {
                        cost[i][j] = 1.0 - iou;
                    }
                }
            }
        }
        auto match = Hungarian(cost);
        std::vector<std::pair<int,int>> pairs;
        for (size_t i=0;i<tracks.size();++i) {
            int mj = (i<(size_t)match.size()? match[i] : -1);
            if (mj>=0) {
                double iou = 1.0 - cost[i][mj];
                if (iou >= p.iou_th) pairs.emplace_back((int)i, (int)mj);
            }
        }
        return pairs;
    };

    // 3) Associate high score first (use stricter IoU)
    std::vector<std::pair<int,int>> pairs_hi = associate(hi);
    std::vector<char> det_matched(dets.size(), false);
    std::vector<char> trk_matched(tracks.size(), false);
    for (auto [ti, mj] : pairs_hi) {
        int di = hi[mj];
        auto& t = tracks[ti];
        if (use_kf) t.kf.update(dets[di].box);
        t.last_box = dets[di].box;
        t.class_id = dets[di].class_id;
        t.last_conf = dets[di].confidence;
        t.hits++; t.time_since_update = 0;
        trk_matched[ti] = true; det_matched[di] = true;
    }

    // 4) Associate low score with unmatched tracks
    // Build temp tracker list of unmatched
    std::vector<int> trk_unmatched;
    for (int i=0;i<(int)tracks.size();++i) if (!trk_matched[i]) trk_unmatched.push_back(i);
    std::vector<int> lo_unmatched;
    for (int j : lo) if (!det_matched[j]) lo_unmatched.push_back(j);
    if (!trk_unmatched.empty() && !lo_unmatched.empty()) {
        std::vector<std::vector<double>> cost(trk_unmatched.size(), std::vector<double>(lo_unmatched.size(), 1.0));
        for (size_t ii=0; ii<trk_unmatched.size(); ++ii) {
            int i = trk_unmatched[ii];
            for (size_t jj=0; jj<lo_unmatched.size(); ++jj) {
                int j = lo_unmatched[jj];
                const auto& d = dets[j];
                if (p.cls_match && d.class_id>=0 && tracks[i].class_id>=0 && d.class_id!=tracks[i].class_id) {
                    cost[ii][jj] = 1.0;
                } else {
                    double iou = IoU(use_kf ? tracks[i].kf.toRect() : tracks[i].last_box, d.box);
                    if (use_kf) {
                        double md = tracks[i].kf.maha(d.box);
                        if (md > p.maha_th) { cost[ii][jj] = 1.0; }
                        else { cost[ii][jj] = 1.0 - iou; }
                    } else {
                        cost[ii][jj] = 1.0 - iou;
                    }
                }
            }
        }
        auto match = Hungarian(cost);
        for (size_t ii=0; ii<trk_unmatched.size(); ++ii) {
            int mj = (ii<(size_t)match.size()? match[ii] : -1);
            if (mj>=0) {
                int ti = trk_unmatched[ii];
                int di = lo_unmatched[mj];
                double iou = 1.0 - cost[ii][mj];
                if (iou >= p.iou_th) {
                    auto& t = tracks[ti];
                    if (use_kf) t.kf.update(dets[di].box);
                    t.last_box = dets[di].box;
                    t.class_id = dets[di].class_id;
                    t.last_conf = dets[di].confidence;
                    t.hits++; t.time_since_update = 0;
                    trk_matched[ti] = true; det_matched[di] = true;
                }
            }
        }
    }

    // 5) Create new tracks for remaining high-score detections
    for (int idx : hi) if (!det_matched[idx]) {
        TrkNode t; t.id = next_id++; t.class_id = dets[idx].class_id; t.last_conf = dets[idx].confidence; if (use_kf) t.kf.init(dets[idx].box); t.last_box = dets[idx].box; t.hits=1; t.age=1; t.time_since_update=0; tracks.push_back(std::move(t));
        det_matched[idx] = true;
    }

    // 6) For unmatched tracks, keep predicted boxes and age them; remove dead
    std::vector<TrkNode> alive;
    alive.reserve(tracks.size());
    for (auto& t : tracks) {
        bool matched = &t - &tracks[0] < (ptrdiff_t)trk_matched.size() ? trk_matched[&t - &tracks[0]] : false;
        if (!matched) {
            // keep prediction; decay conf slightly
            t.last_conf = std::max(0.0f, t.last_conf * 0.99f);
        }
        if (t.time_since_update <= p.max_age) alive.push_back(t);
    }
    tracks.swap(alive);

    // 7) Emit outputs (confirmed tracks only, no pure predictions)
    out.reserve(tracks.size());
    std::vector<Detection> tmp;
    for (auto& t : tracks) {
        if (t.hits >= p.min_hits && t.time_since_update==0) {
            Detection d; d.box = (use_kf ? t.kf.toRect() : t.last_box); d.class_id = t.class_id; d.confidence = t.last_conf; d.track_id = t.id; d.is_pred = false;
            tmp.push_back(d);
        }
    }
    // track-level NMS to avoid multiple boxes for same person
    if (p.out_nms) {
        const double nms_iou = std::clamp<double>(p.out_nms_iou, 0.0, 1.0);
        std::vector<char> keep(tmp.size(), 1);
        for (size_t i=0;i<tmp.size();++i) if (keep[i]) {
            for (size_t j=i+1;j<tmp.size();++j) if (keep[j]) {
                if (IoU(tmp[i].box, tmp[j].box) > nms_iou) {
                    if (tmp[i].confidence >= tmp[j].confidence) keep[j]=0; else keep[i]=0;
                }
            }
        }
        for (size_t i=0;i<tmp.size();++i) if (keep[i]) out.push_back(tmp[i]);
    } else {
        out = std::move(tmp);
    }
}

} // namespace bt
