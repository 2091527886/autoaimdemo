#include "SystemTuner.hpp"
#include "pch.hpp"

void SetProcPrioHigh() {
  BOOL ok = SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
  lg::ok_or_werr(P_B, "ProcPrio(HIGH)", ok);
}

void SetThreadPrioHighest() {
  BOOL ok = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST); lg::ok_or_werr(P_B, "ThreadPrio(HIGHEST)", ok);
}

void SetThreadMemPrioNormal(bool log) {
  MEMORY_PRIORITY_INFORMATION memPri{};
  memPri.MemoryPriority = MEMORY_PRIORITY_NORMAL;
  bool ok = SetThreadInformation(GetCurrentThread(), ThreadMemoryPriority, &memPri, sizeof(memPri)) ? true : false;
  lg::ok_or_werr_if(log, P_B, "ThreadMemPrio=NORMAL", ok);
}

void SetProcPrioBoost(bool enable, bool log) {
  BOOL ok = SetProcessPriorityBoost(GetCurrentProcess(), enable ? TRUE : FALSE);
  lg::ok_or_werr_if(log, P_B, enable ? "ProcPrioBoost(TRUE)" : "ProcPrioBoost(FALSE)", ok);
}

void SetProcPrioBoostDisabled(bool log) {
  SetProcPrioBoost(false, log);
}

void disablePowerThrottle(bool log) {
  PROCESS_POWER_THROTTLING_STATE pState = {};
  pState.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
  pState.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
  pState.StateMask = 0;
  {
    BOOL ok = SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pState, sizeof(pState)); lg::ok_or_werr_if(log, P_B, "ProcPowerThrottling=OFF", ok);
  }

  THREAD_POWER_THROTTLING_STATE tState = {};
  tState.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
  tState.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
  tState.StateMask = 0;
  {
    BOOL ok = SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &tState, sizeof(tState)); lg::ok_or_werr_if(log, P_B, "ThreadPowerThrottling=OFF", ok);
  }

  {
    BOOL ok = SetThreadPriorityBoost(GetCurrentThread(), FALSE); lg::ok_or_werr_if(log, P_B, "ThreadPriorityBoost(FALSE)", ok);
  }
}

void TuneDxgi(ID3D11Device *dev, int gpu_prio, int max_latency) {
  if (!dev)
    return;
  IDXGIDevice1 *dxgi1 = nullptr;
  HRESULT hr = dev->QueryInterface(__uuidof(IDXGIDevice1), (void **)&dxgi1);
  if (FAILED(hr) || !dxgi1) {
    lg::hrfail(P_E, "QueryInterface(IDXGIDevice1)", hr);
    return;
  }
  hr = dxgi1->SetGPUThreadPriority(gpu_prio);
  if (FAILED(hr)) {
    lg::hrfail(P_E, "SetGPUThreadPriority(7)", hr);
  }
  hr = dxgi1->SetMaximumFrameLatency(max_latency);
  if (FAILED(hr)) {
    lg::hrfail(P_E, "SetMaximumFrameLatency(1)", hr);
  }
  dxgi1->Release();
}
