#pragma once

// Tiny shared helpers for timing aliases to reduce duplication.
// No logic change; purely aliases used across modules.
#include <chrono>

using clk = std::chrono::high_resolution_clock;
using tp  = clk::time_point;
using msd = std::chrono::duration<double, std::milli>;
inline tp tn() { return clk::now(); }
