#pragma once

// Centralized heavy system/STL headers to shrink per-file includes
// Do not change logic; just include once here.

#ifdef _WIN32
#  include <windows.h>
#  include <d3d11.h>
#  include <dxgi.h>
#  include <dxgi1_2.h>
#  include <d2d1.h>
#  include <d2d1_1.h>
#  include <d2d1helper.h>
#  include <dwrite.h>
#  include <dcomp.h>
#  include <dwmapi.h>
#  include <avrt.h>
#  include <wrl/client.h>
#endif

#include <cwctype>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
