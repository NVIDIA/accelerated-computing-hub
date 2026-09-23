# Step 0 — import MuJoCo Warp and Warp.
#
# `mujoco_warp` is conventionally aliased to `mjw`. It does not replace the
# `mujoco` package: you still compile MJCF with the standard bindings and only
# hand the compiled model over to MJWarp.
#
# `warp` (aliased `wp`) is the kernel framework MJWarp is written in. You need
# it directly for device management (wp.init, wp.get_device), host<->device
# copies (wp.array, wp.copy) and CUDA graph capture.

import mujoco_warp as mjw
import warp as wp
