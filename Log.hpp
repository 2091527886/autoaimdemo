#pragma once
#include <iostream>

namespace lg {
    inline std::ostream& o() noexcept { return std::cout; }
    inline std::ostream& e() noexcept { return std::cerr; }
    inline std::wostream& wo() noexcept { return std::wcout; }
}
