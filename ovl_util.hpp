#pragma once
#include "Game_overlay.h"
#include <d2d1_1.h>

void GetVirtualScreen(int& x, int& y, int& w, int& h);
void SetPerMonitorV2DpiAwareness();
D2D1_COLOR_F ToD2DColor(OverlayColor argb);
