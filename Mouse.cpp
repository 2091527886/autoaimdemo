#include "Mouse.hpp"
#include "Log.hpp"
#include "pch.hpp"

using E = Send::Error;
using T = Send::SendType;
using M = Send::MoveMode;

Mouse::Mouse() {
  LO << P_M "Init driver" << END;
  dll_ = LoadLibrary(L"IbInputSimulator.dll");
  if(!dll_) throw std::runtime_error(P_M "Load IbInputSimulator.dll failed");

  init_ = (SendInitFn)GetProcAddress(dll_, "IbSendInit");
  destroy_ = (SendDestroyFn)GetProcAddress(dll_, "IbSendDestroy");
  move_ = (SendMouseMoveFn)GetProcAddress(dll_, "IbSendMouseMove");

  if(!init_ || !destroy_ || !move_){ FreeLibrary(dll_); dll_=nullptr; throw std::runtime_error(P_M "Required functions not found in IbInputSimulator.dll"); }
  E error = init_(T::Logitech, static_cast<Send::InitFlags>(0), nullptr);
  if (error != E::Success){ FreeLibrary(dll_); dll_=nullptr; throw std::runtime_error(P_M "Init failed: " + std::to_string((int)error)); }
  LO << P_M "Driver OK" << END;
}

Mouse::~Mouse() {
  if (dll_){ if (destroy_) destroy_(); FreeLibrary(dll_); LO << P_M "Driver bye" << END; }
}

void Mouse::Move(int dx, int dy) {
  if (move_){ bool ok = move_(dx, dy, M::Relative); if (!ok) lg::ok_or_werr_msg(false, P_M, std::string("mv dx=")+std::to_string(dx)+" dy="+std::to_string(dy), ok); }
}
