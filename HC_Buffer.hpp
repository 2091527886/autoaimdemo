#pragma once
#include <deque>
#include <utility>
#include <algorithm>
#include <vector>

namespace hc {

class StepBuffer {
public:
    explicit StepBuffer(size_t cap=2048): cap_(cap){}
    void set_cap(size_t c){ cap_=c; trim(); }
    void clear(){ dq_.clear(); }
    size_t size() const { return dq_.size(); }
    void push_step(double dx, double dy){ if (cap_ && dq_.size()>=cap_) dq_.pop_front(); dq_.emplace_back(dx,dy); }
    void push_curve(const std::vector<std::pair<double,double>>& steps){ for (auto& s: steps){ push_step(s.first, s.second); } }
    bool pop_step(double& dx, double& dy){ if (dq_.empty()) return false; auto s=dq_.front(); dq_.pop_front(); dx=s.first; dy=s.second; return true; }
private:
    void trim(){ while(cap_ && dq_.size()>cap_) dq_.pop_front(); }
    std::deque<std::pair<double,double>> dq_;
    size_t cap_;
};

}
