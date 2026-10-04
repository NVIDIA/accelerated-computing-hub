# Step 0 — import Newton.
#
# `newton` is the top-level package (ModelBuilder, Model, State, Control,
# Contacts, CollisionPipeline, eval_fk). `JointTargetMode` selects position
# and/or velocity servos; Control.joint_f supplies direct generalized forces.
# `SolverMuJoCo` drives native MuJoCo-C or MuJoCo Warp underneath.

import newton
from newton import JointTargetMode
from newton.solvers import SolverMuJoCo
