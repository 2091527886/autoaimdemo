#pragma once
#include <windows.h>
#include <iostream>

// Disable power throttling for process and current thread (migrated)
void DisablePowerThrottlingForProcessAndThread(bool log=true);
