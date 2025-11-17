#pragma once
#include <string>
#include <cstdint>
#include <memory>

struct OverlayRect { float x, y, w, h; };
using OverlayColor = uint32_t; // ARGB: 0xAARRGGBB
inline constexpr OverlayColor ARGB(uint8_t a, uint8_t r, uint8_t g, uint8_t b) {
    return (OverlayColor(a) << 24) | (OverlayColor(r) << 16) | (OverlayColor(g) << 8) | OverlayColor(b);
}

class Config; // fwd

class Game_overlay
{
public:
    bool Start();
    void Stop();
    bool IsRunning() const;

    void BeginFrame();
    void EndFrame();
    // Single-threaded render pump: process window messages and present one frame
    void Tick();

    void AddRect(const OverlayRect& rc, OverlayColor color, float thickness = 1.0f);
    void AddRectL(const OverlayRect& rc, OverlayColor color, float thickness, uint16_t layer);
    void FillRect(const OverlayRect& rc, OverlayColor color);
    void FillRectL(const OverlayRect& rc, OverlayColor color, uint16_t layer);
    void AddText(float x, float y, const std::wstring& text, float sizePx,
        OverlayColor color, const std::wstring& font = L"Segoe UI");
    void AddTextL(float x, float y, const std::wstring& text, float sizePx,
        OverlayColor color, const std::wstring& font, uint16_t layer);
    void AddTextL(float x, float y, const std::wstring& text, float sizePx,
        OverlayColor color, uint16_t layer);
    void AddLine(float x1, float y1, float x2, float y2, float thickness, OverlayColor color);
    void AddLineL(float x1, float y1, float x2, float y2, float thickness, OverlayColor color, uint16_t layer);

    void UseVirtualScreen();

    // Provide config for GPU/vsync/latency
    void SetConfig(const Config& cfg);

    Game_overlay();
    ~Game_overlay();

private:
    Game_overlay(const Game_overlay&) = delete;
    Game_overlay& operator=(const Game_overlay&) = delete;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};
