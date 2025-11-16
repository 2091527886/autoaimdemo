#pragma once
#include "pch.hpp"
#include "Game_overlay.h"
#include "Log.hpp"
#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <chrono>

// Forward declarations for COM interfaces to keep header light
struct ID3D11Device; struct ID3D11DeviceContext; struct IDXGISwapChain1; struct ID3D11RenderTargetView;
struct ID2D1Factory1; struct ID2D1Device; struct ID2D1DeviceContext; struct ID2D1Bitmap1; struct ID2D1SolidColorBrush;
struct IDWriteFactory; struct IDWriteTextFormat; struct IDCompositionDevice; struct IDCompositionTarget; struct IDCompositionVisual;

#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

struct DrawCmd{ enum Type{Rect,RectFilled,Text,Line} type; OverlayColor color=0; float thickness=1.0f; union{ OverlayRect rect; struct{float x1,y1,x2,y2;} line; struct{float x,y,size;} textPos; }; std::wstring text; std::wstring fontName; };
struct DrawList{ std::vector<DrawCmd> cmds; };

class Config;

struct Game_overlay::Impl{
    HINSTANCE hinst=nullptr; HWND hwnd=nullptr; int winX=0,winY=0,winW=0,winH=0; bool useVirtual=true;
    ComPtr<ID3D11Device> d3d; ComPtr<ID3D11DeviceContext> d3dCtx; ComPtr<IDXGISwapChain1> swapChain; ComPtr<ID3D11RenderTargetView> rtv;
    ComPtr<ID2D1Factory1> d2dFactory; ComPtr<ID2D1Device> d2dDevice; ComPtr<ID2D1DeviceContext> d2dCtx; ComPtr<ID2D1Bitmap1> d2dTarget; ComPtr<ID2D1SolidColorBrush> brush;
    ComPtr<IDWriteFactory> dwFactory; std::unordered_map<uint64_t, ComPtr<IDWriteTextFormat>> textFormatCache;
    ComPtr<IDCompositionDevice> dcompDevice; ComPtr<IDCompositionTarget> dcompTarget; ComPtr<IDCompositionVisual> dcompRoot;
    std::atomic<bool> running{false}, visible{true}; std::mutex pendingMutex; DrawList pending; std::shared_ptr<DrawList> current;
    std::wstring gpu_name, gpu_luid; int gpu_idx=-1; bool gpu_enum=false; bool vsync=false; int max_latency=1;
    bool skip_empty=true; bool small_window=true; int cfg_win_x=0,cfg_win_y=0,cfg_win_w=640,cfg_win_h=640;
    void UseVirtualScreen(){ useVirtual=true; } void SetBounds(int x,int y,int w,int h){ useVirtual=false; winX=x; winY=y; winW=w; winH=h; }
    bool Start(); void Stop();
    void BeginFrame(); void EndFrame(); void AddCmd(const DrawCmd&);
    static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
    bool CreateWindowAndDevices(); void DestroyWindowAndDevices(); void RenderOne();
    HRESULT CreateTextFormat(const std::wstring& font,float size,IDWriteTextFormat** out);
    HRESULT CreateTargets(); void ReleaseTargets();
};
