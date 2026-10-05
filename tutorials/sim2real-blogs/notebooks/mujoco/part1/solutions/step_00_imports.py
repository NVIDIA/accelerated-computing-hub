# Step 0 — import the MuJoCo Python bindings.
#
# `mujoco` is the engine (MjModel, MjData, mj_step, mj_forward, mj_jac...).
# `mujoco.viewer` is the interactive passive viewer and is a separate submodule,
# so importing `mujoco` alone is not enough.

import mujoco
import mujoco.viewer
