#pragma once

#include "api.hpp"
#include "types.hpp"

namespace trtyolo {

struct TRTYOLOAPI BaseRes {
    int                num = 0;
    std::vector<int>   classes;
    std::vector<float> scores;

    BaseRes() = default;
    BaseRes(int num, const std::vector<int>& classes, const std::vector<float>& scores)
        : num(num), classes(classes), scores(scores) {}
};

struct TRTYOLOAPI ClassifyRes : public BaseRes {
    friend std::ostream& operator<<(std::ostream& os, const ClassifyRes& res) {
        os << "ClassifyRes(\n    num=" << res.num << ",\n    classes=[";
        for (const auto& c : res.classes) os << c << ", ";
        os << "],\n    scores=[";
        for (const auto& s : res.scores) os << s << ", ";
        os << "]\n)";
        return os;
    }
};

struct TRTYOLOAPI DetectRes : public BaseRes {
    std::vector<Box> boxes;

    DetectRes() = default;
    DetectRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes)
        : BaseRes(num, classes, scores), boxes(boxes) {}

    friend std::ostream& operator<<(std::ostream& os, const DetectRes& res) {
        os << "DetectRes(\n    num=" << res.num << ",\n    classes=[";
        for (const auto& c : res.classes) os << c << ", ";
        os << "],\n    scores=[";
        for (const auto& s : res.scores) os << s << ", ";
        os << "],\n    boxes=[\n";
        for (const auto& box : res.boxes) os << "        " << box << ",\n";
        os << "    ]\n)";
        return os;
    }
};

struct TRTYOLOAPI OBBRes : public BaseRes {
    std::vector<RotatedBox> boxes;

    OBBRes() = default;
    OBBRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<RotatedBox>& boxes)
        : BaseRes(num, classes, scores), boxes(boxes) {}

    friend std::ostream& operator<<(std::ostream& os, const OBBRes& res) {
        os << "OBBRes(\n    num=" << res.num << ",\n    classes=[";
        for (const auto& c : res.classes) os << c << ", ";
        os << "],\n    scores=[";
        for (const auto& s : res.scores) os << s << ", ";
        os << "],\n    boxes=[\n";
        for (const auto& box : res.boxes) os << "        " << box << ",\n";
        os << "    ]\n)";
        return os;
    }
};

struct TRTYOLOAPI SegmentRes : public BaseRes {
    std::vector<Box>  boxes;
    std::vector<Mask> masks;

    SegmentRes() = default;
    SegmentRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes, const std::vector<Mask>& masks)
        : BaseRes(num, classes, scores), boxes(boxes), masks(masks) {}

    friend std::ostream& operator<<(std::ostream& os, const SegmentRes& res) {
        os << "SegmentRes(\n    num=" << res.num << ",\n    classes=[";
        for (const auto& c : res.classes) os << c << ", ";
        os << "],\n    scores=[";
        for (const auto& s : res.scores) os << s << ", ";
        os << "],\n    boxes: [\n";
        for (const auto& box : res.boxes) os << "        " << box << ",\n";
        os << "],\n    masks: [\n";
        for (const auto& mask : res.masks) os << "        " << mask << "\n";
        os << "    ]\n)";
        return os;
    }
};

struct TRTYOLOAPI PoseRes : public BaseRes {
    std::vector<Box>                   boxes;
    std::vector<std::vector<KeyPoint>> kpts;

    PoseRes() = default;
    PoseRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes, const std::vector<std::vector<KeyPoint>>& kpts)
        : BaseRes(num, classes, scores), boxes(boxes), kpts(kpts) {}

    friend std::ostream& operator<<(std::ostream& os, const PoseRes& res) {
        os << "PoseRes(\n    num=" << res.num << ",\n    classes=[";
        for (const auto& c : res.classes) os << c << ", ";
        os << "],\n    scores=[";
        for (const auto& s : res.scores) os << s << ", ";
        os << "],\n    boxes=[\n";
        for (const auto& box : res.boxes) os << "        " << box << "\n";
        os << "],\n    kpts=[\n";
        for (const auto& kp_list : res.kpts) {
            os << "        [ ";
            for (const auto& kp : kp_list) os << "            " << kp << ", ";
            os << "        ],\n";
        }
        os << "    ]\n)";
        return os;
    }
};

} // namespace trtyolo
