#pragma once
#include "Game_overlay.h"
#include "Log.hpp"
#include <windows.h>
#include <dwmapi.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <dxgi1_3.h>
#include <d2d1_1.h>
#include <dwrite.h>
#include <dcomp.h>
#include <wrl/client.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <chrono>

#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "dcomp.lib")

using Microsoft::WRL::ComPtr;

struct DrawCmd
{
    enum Type { Rect, RectFilled, Text } type;
    OverlayColor color = 0;
    float thickness = 1.0f;
    union
    {
        OverlayRect rect;
        struct { float x, y, size; } textPos;
    };
    std::wstring text;
    std::wstring fontName;
};

struct DrawList
{
    std::vector<DrawCmd> cmds;
};

class Config;

struct Game_overlay::Impl
{
    HINSTANCE hinst = nullptr;
    HWND hwnd = nullptr;
    int winX = 0, winY = 0, winW = 0, winH = 0;
    bool useVirtual = true;

    ComPtr<ID3D11Device>           d3d;
    ComPtr<ID3D11DeviceContext>    d3dCtx;
    ComPtr<IDXGISwapChain1>        swapChain;
    ComPtr<ID3D11RenderTargetView> rtv;

    ComPtr<ID2D1Factory1>          d2dFactory;
    ComPtr<ID2D1Device>            d2dDevice;
    ComPtr<ID2D1DeviceContext>     d2dCtx;
    ComPtr<ID2D1Bitmap1>           d2dTarget;
    ComPtr<ID2D1SolidColorBrush>   brush;

    ComPtr<IDWriteFactory>         dwFactory;
    std::unordered_map<uint64_t, ComPtr<IDWriteTextFormat>> textFormatCache;

    ComPtr<IDCompositionDevice>    dcompDevice;
    ComPtr<IDCompositionTarget>    dcompTarget;
    ComPtr<IDCompositionVisual>    dcompRoot;

    std::atomic<bool> running{ false };
    std::atomic<bool> visible{ true };
    std::mutex pendingMutex;
    DrawList pending;
    std::shared_ptr<DrawList> current;

    // config view (overlay-only fields used)
    std::wstring gpu_name;
    std::wstring gpu_luid;
    int gpu_idx = -1;
    bool gpu_enum = false;
    bool vsync = false; // 默认不使用 vsync
    int  max_latency = 1;

    bool skip_empty = true;

    // small window
    bool small_window = true;
    int  cfg_win_x = 0, cfg_win_y = 0, cfg_win_w = 640, cfg_win_h = 640;

    void UseVirtualScreen() { useVirtual = true; }
    void SetBounds(int x, int y, int w, int h) { useVirtual = false; winX = x; winY = y; winW = w; winH = h; }

    bool Start();
    void Stop();

    void BeginFrame();
    void EndFrame();
    void AddCmd(const DrawCmd&);

    static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
    bool    CreateWindowAndDevices();
    void    DestroyWindowAndDevices();
    void    RenderOne();

    HRESULT CreateTextFormat(const std::wstring& font, float size, IDWriteTextFormat** out);
    HRESULT CreateTargets();
    void    ReleaseTargets();
};
