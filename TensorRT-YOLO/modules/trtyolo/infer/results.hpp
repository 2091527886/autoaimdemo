#pragma once

#include "api.hpp"
#include "types.hpp"

namespace trtyolo {

// tiny helpers to print vectors and nested data without changing output format
static inline void pv(std::ostream& os, const std::vector<int>& v){ for(const auto& x:v) os<<x<<", "; }
static inline void pvf(std::ostream& os, const std::vector<float>& v){ for(const auto& x:v) os<<x<<", "; }
static inline void pb(std::ostream& os, const std::vector<Box>& v){ for(const auto& x:v) os<<"        "<<x<<",\n"; }
static inline void prb(std::ostream& os, const std::vector<RotatedBox>& v){ for(const auto& x:v) os<<"        "<<x<<",\n"; }
static inline void pm(std::ostream& os, const std::vector<Mask>& v){ for(const auto& x:v) os<<"        "<<x<<"\n"; }
static inline void pk(std::ostream& os, const std::vector<std::vector<KeyPoint>>& v){ for(const auto& l:v){ os<<"        [ "; for(const auto& k:l) os<<"            "<<k<<", "; os<<"        ],\n"; } }


struct TRTYOLOAPI BaseRes {
    int                num = 0;
    std::vector<int>   classes;
    std::vector<float> scores;

    BaseRes() = default;
    BaseRes(int num, const std::vector<int>& classes, const std::vector<float>& scores)
        : num(num), classes(classes), scores(scores) {}
};

struct TRTYOLOAPI ClassifyRes : public BaseRes {
    friend std::ostream& operator<<(std::ostream& os, const ClassifyRes& r){ os<<"ClassifyRes(\n    num="<<r.num<<",\n    classes=["; pv(os,r.classes); os<<"],\n    scores=["; pvf(os,r.scores); os<<"]\n)"; return os; }
};

struct TRTYOLOAPI DetectRes : public BaseRes {
    std::vector<Box> boxes;

    DetectRes() = default;
    DetectRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes)
        : BaseRes(num, classes, scores), boxes(boxes) {}

    friend std::ostream& operator<<(std::ostream& os, const DetectRes& r){ os<<"DetectRes(\n    num="<<r.num<<",\n    classes=["; pv(os,r.classes); os<<"],\n    scores=["; pvf(os,r.scores); os<<"],\n    boxes=[\n"; pb(os,r.boxes); os<<"    ]\n)"; return os; }
};

struct TRTYOLOAPI OBBRes : public BaseRes {
    std::vector<RotatedBox> boxes;

    OBBRes() = default;
    OBBRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<RotatedBox>& boxes)
        : BaseRes(num, classes, scores), boxes(boxes) {}

    friend std::ostream& operator<<(std::ostream& os, const OBBRes& r){ os<<"OBBRes(\n    num="<<r.num<<",\n    classes=["; pv(os,r.classes); os<<"],\n    scores=["; pvf(os,r.scores); os<<"],\n    boxes=[\n"; prb(os,r.boxes); os<<"    ]\n)"; return os; }
};

struct TRTYOLOAPI SegmentRes : public BaseRes {
    std::vector<Box>  boxes;
    std::vector<Mask> masks;

    SegmentRes() = default;
    SegmentRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes, const std::vector<Mask>& masks)
        : BaseRes(num, classes, scores), boxes(boxes), masks(masks) {}

    friend std::ostream& operator<<(std::ostream& os, const SegmentRes& r){ os<<"SegmentRes(\n    num="<<r.num<<",\n    classes=["; pv(os,r.classes); os<<"],\n    scores=["; pvf(os,r.scores); os<<"],\n    boxes: [\n"; pb(os,r.boxes); os<<"],\n    masks: [\n"; pm(os,r.masks); os<<"    ]\n)"; return os; }
};

struct TRTYOLOAPI PoseRes : public BaseRes {
    std::vector<Box>                   boxes;
    std::vector<std::vector<KeyPoint>> kpts;

    PoseRes() = default;
    PoseRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes, const std::vector<std::vector<KeyPoint>>& kpts)
        : BaseRes(num, classes, scores), boxes(boxes), kpts(kpts) {}

    friend std::ostream& operator<<(std::ostream& os, const PoseRes& r){ os<<"PoseRes(\n    num="<<r.num<<",\n    classes=["; pv(os,r.classes); os<<"],\n    scores=["; pvf(os,r.scores); os<<"],\n    boxes=[\n"; pb(os,r.boxes); os<<"],\n    kpts=[\n"; pk(os,r.kpts); os<<"    ]\n)"; return os; }
};

} // namespace trtyolo
