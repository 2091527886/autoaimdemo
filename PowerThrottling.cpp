#include "PowerThrottling.hpp"
#include "Log.hpp"

void disablePowerThrottle(bool log) {
   PROCESS_POWER_THROTTLING_STATE pState = {};
   pState.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
   pState.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
   pState.StateMask = 0;
   if (!SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pState, sizeof(pState))) {
       if (log) lg::e() << "[BOOT] ProcPowerThrottling=OFF failed: " << GetLastError() << std::endl;
   } else if (log) {
       lg::o() << "[BOOT] ProcPowerThrottling=OFF OK" << std::endl;
   }

   THREAD_POWER_THROTTLING_STATE tState = {};
   tState.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
   tState.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
   tState.StateMask = 0;
   if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &tState, sizeof(tState))) {
       if (log) lg::e() << "[BOOT] ThreadPowerThrottling=OFF failed: " << GetLastError() << std::endl;
   } else if (log) {
       lg::o() << "[BOOT] ThreadPowerThrottling=OFF OK" << std::endl;
   }

   if (!SetThreadPriorityBoost(GetCurrentThread(), FALSE)) {
       if (log) lg::e() << "[BOOT] ThreadPriorityBoost(FALSE) failed: " << GetLastError() << std::endl;
   } else if (log) {
       lg::o() << "[BOOT] ThreadPriorityBoost(FALSE) OK" << std::endl;
   }
}
