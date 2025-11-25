#pragma once
#include <windows.h>

class Clicker {
public:
    static void MouseDown(){ send(true); }
    static void MouseUp(){ send(false); }
private:
    static void send(bool down){
        INPUT in{}; in.type = INPUT_MOUSE; in.mi.dwFlags = down? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_LEFTUP; SendInput(1, &in, sizeof(INPUT));
    }
};
