#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <algorithm>
#include "Game_overlay.h"
#include "ovl_util.hpp"

// 轻量文本批处理与去重/合并/防重叠布局
// 估算宽度: len * size * 0.6f, 行高: size * 1.2f
struct TxtLn {
    float x = 0.f;
    float y = 0.f;
    float size = 12.f;
    OverlayColor color = 0xFFFFFFFF;
    uint16_t layer = 0;
    int prio = 10; // 数值越小优先级越高
    std::wstring w;
    std::wstring font; // 空则用默认
};

struct TxtBlkKey {
    uint16_t layer;
    OverlayColor color;
    float size;
    std::wstring font;
    bool operator==(const TxtBlkKey& o) const{
        return layer==o.layer && color==o.color && size==o.size && font==o.font;
    }
};

struct TxtBlkKeyHash {
    size_t operator()(const TxtBlkKey& k) const noexcept{
        size_t h = 1469598103934665603ull;
        auto mix=[&](size_t v){ h^=v; h*=1099511628211ull; };
        mix(std::hash<uint16_t>{}(k.layer));
        mix(std::hash<uint32_t>{}((uint32_t)k.color));
        mix(std::hash<uint32_t>{}(*reinterpret_cast<const uint32_t*>(&k.size))); // 近似
        mix(std::hash<std::wstring>{}(k.font));
        return h;
    }
};

struct TxtBlk {
    float x=0.f, y=0.f, w=0.f, h=0.f; // 位置和整体大小
    float size=12.f;
    OverlayColor color=0xFFFFFFFF;
    uint16_t layer=0;
    std::wstring font; // 空则默认
    std::vector<std::wstring> lines;
};

class TextBatch {
public:
    void add(const TxtLn& ln){ items_.push_back(ln); }

    void flush(Game_overlay& gover){
        if (items_.empty()) return;
        // 分组
        std::unordered_map<TxtBlkKey, std::vector<TxtLn>, TxtBlkKeyHash> groups;
        groups.reserve(items_.size());
        for (auto& ln : items_) {
            TxtBlkKey k{ln.layer, ln.color, ln.size, ln.font};
            groups[k].push_back(ln);
        }
        // 每组内部布局
        for (auto& kv : groups){
            const auto& key = kv.first;
            auto& vec = kv.second;
            std::sort(vec.begin(), vec.end(), [](const TxtLn& a, const TxtLn& b){
                if (a.prio != b.prio) return a.prio < b.prio; // 高优先级先排
                if (a.y==b.y) return a.x < b.x; return a.y < b.y;
            });
            std::vector<TxtBlk> blks;
            blks.reserve(vec.size());
            const float step = key.size * 1.2f;
            auto est_w = [](const std::wstring& ws, float size){ return (float)ws.size() * size * 0.6f; };
            auto overlap = [](float x1,float y1,float w1,float h1,float x2,float y2,float w2,float h2){
                return !(x1+w1 <= x2 || x2+w2 <= x1 || y1+h1 <= y2 || y2+h2 <= y1);
            };
            // 屏幕范围（尽量不出界）：这里取虚拟屏，若不可用则不夹紧
            int vx=0, vy=0, vw=0, vh=0; GetVirtualScreen(vx, vy, vw, vh);
            auto clampf=[&](float v, float lo, float hi){ return v<lo?lo:(v>hi?hi:v); };
            for (const auto& ln : vec){
                float lw = std::max(1.f, est_w(ln.w, ln.size));
                float lh = std::max(1.f, step);
                float px = ln.x, py = ln.y;
                // 先尝试合并到已有块（同组）
                int merge_idx = -1;
                for (int i=0;i<(int)blks.size();++i){
                    const auto& b = blks[i];
                    if (overlap(px, py, lw, lh, b.x, b.y, b.w, b.h)) { merge_idx = i; break; }
                }
                if (merge_idx >= 0){
                    auto& b = blks[merge_idx];
                    b.lines.push_back(ln.w);
                    b.w = std::max(b.w, lw);
                    b.h += lh;
                } else {
                    // 轻微位移避免重叠（优先向下，再向上，再向右）
                    int tries=0; bool placed=false; int phase=0;
                    float ox=px, oy=py;
                    while (tries < 6 && !placed){
                        bool hit=false;
                        for (const auto& b : blks){ if (overlap(px, py, lw, lh, b.x, b.y, b.w, b.h)){ hit=true; break; } }
                        if (!hit) { placed=true; break; }
                        if (phase==0){ py += step; }
                        else if (phase==1){ py = oy - step; phase=2; continue; }
                        else { px += step; }
                        ++tries; if (tries==2) phase=1;
                    }
                    // 边界夹紧
                    if (vw>0 && vh>0){
                        px = clampf(px, (float)vx, (float)(vx + vw - lw));
                        py = clampf(py, (float)vy, (float)(vy + vh - lh));
                    }
                    TxtBlk nb; nb.x=px; nb.y=py; nb.w=lw; nb.h=lh; nb.size=ln.size; nb.color=ln.color; nb.layer=ln.layer; nb.font=ln.font; nb.lines.push_back(ln.w);
                    blks.push_back(std::move(nb));
                }
            }
            // 绘制
            for (const auto& b : blks){
                for (size_t i=0;i<b.lines.size();++i){
                    float yy = b.y + (float)i * step;
                    if (!b.font.empty()) gover.AddTextL(b.x, yy, b.lines[i], b.size, b.color, b.font, b.layer);
                    else gover.AddTextL(b.x, yy, b.lines[i], b.size, b.color, b.layer);
                }
            }
        }
        items_.clear();
    }

private:
    std::vector<TxtLn> items_;
};
