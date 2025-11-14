#pragma once
#include <wrl/client.h>
#include "pch.hpp"
#include <string>

Microsoft::WRL::ComPtr<IDXGIAdapter1> pick_adapter_cfg(const std::wstring& name,
                                       const std::wstring& luid,
                                       int idx,
                                       bool do_enum);
