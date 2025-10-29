#include "PowerThrottling.hpp"

void DisablePowerThrottlingForProcessAndThread(bool log) {
   PROCESS_POWER_THROTTLING_STATE pstate = {};
   pstate.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
   pstate.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
   pstate.StateMask = 0;
   if (!SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pstate, sizeof(pstate))) {
       if (log) std::cerr << "[BOOT] ProcPowerThrottling=OFF failed: " << GetLastError() << std::endl;
   } else if (log) {
       std::cout << "[BOOT] ProcPowerThrottling=OFF OK" << std::endl;
   }

   THREAD_POWER_THROTTLING_STATE tstate = {};
   tstate.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
   tstate.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
   tstate.StateMask = 0;
   if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &tstate, sizeof(tstate))) {
       if (log) std::cerr << "[BOOT] ThreadPowerThrottling=OFF failed: " << GetLastError() << std::endl;
   } else if (log) {
       std::cout << "[BOOT] ThreadPowerThrottling=OFF OK" << std::endl;
   }

   if (!SetThreadPriorityBoost(GetCurrentThread(), FALSE)) {
       if (log) std::cerr << "[BOOT] ThreadPriorityBoost(FALSE) failed: " << GetLastError() << std::endl;
   } else if (log) {
       std::cout << "[BOOT] ThreadPriorityBoost(FALSE) OK" << std::endl;
   }
}
