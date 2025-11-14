#include "letterbox_kernels.cuh"
// This TU compiles the kernel definitions exactly once (generic and template definitions are present here via header).
// No explicit instantiations are emitted; template kernels are instantiated in the launcher TUs that include
// letterbox_kernels_templates.cuh.
