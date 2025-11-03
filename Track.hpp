#pragma once
#include <opencv2/core.hpp>
#include <vector>

// legacy header from build system including Track; keep minimal to avoid conflicts
// Ensure we don't collide with bt::Track by using a different guard or removing usage.
#pragma once
// This header is kept for build compatibility but avoids name clash.
struct LegacyTrack { int dummy=0; };
// Backward compatibility alias for external code that still uses Track
using Track = LegacyTrack;

