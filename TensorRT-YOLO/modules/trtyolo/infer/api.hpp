#pragma once

#ifdef _MSC_VER
#define TRTYOLOAPI __declspec(dllexport)
#else
#define TRTYOLOAPI __attribute__((visibility("default")))
#endif

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace trtyolo {} // namespace trtyolo
