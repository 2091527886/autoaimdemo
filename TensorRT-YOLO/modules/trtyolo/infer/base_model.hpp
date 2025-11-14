#pragma once

#include "api.hpp"
#include <tuple>

namespace trtyolo {

class InferOption; // fwd

class TRTYOLOAPI BaseModel {
public:
    BaseModel();
    ~BaseModel();

    explicit BaseModel(const std::string& trt_engine_file, const InferOption& infer_option);

    int batch() const;
    std::tuple<std::string, std::string, std::string> performanceReport();

public:
    class Impl; // keep public for friend/impl visibility across TU
protected:
    std::unique_ptr<Impl> impl_;
};

} // namespace trtyolo
