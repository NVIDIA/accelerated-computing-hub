# Step 7 — construct the solver.
#
# SolverMuJoCo is Newton's reduced-coordinate rigid-body solver, and it drives
# native MuJoCo-C or MuJoCo Warp underneath. This snippet goes at TODO Step 7;
# the surrounding scaffold selects the backend/contact path before this call.
#
#   use_mujoco_cpu   True runs native MuJoCo-C, which ignores Newton Contacts
#                    in 1.6.0. False runs MuJoCo Warp (CPU or CUDA).
# This tutorial selects native MuJoCo contacts on CPU. On CUDA, either robot
# may use Newton contacts (default) or opt in to --use-mujoco-contacts.
#   njmax / nconmax  constraint/contact memory budgets per world; Newton's
#                    contact manifolds need more constraint rows than Part 2
#   solver           "newton" (default, robust) or "cg"
#   integrator       "implicitfast" is the usual choice for contact-rich scenes
#   cone             "elliptic" is more accurate, "pyramidal" is cheaper
#   impratio         how much stiffer normal constraints are than friction;
#                    raise it when objects slip in the gripper
#   iterations /     solver and line-search effort per step; lowering them is a
#   ls_iterations    speed/accuracy trade, secondary to the memory budgets

self.solver = newton.solvers.SolverMuJoCo(
    self.model,
    use_mujoco_cpu=use_mujoco_cpu,
    solver="newton",
    integrator="implicitfast",
    njmax=max(self.spec.njmax, 1024),
    nconmax=self.spec.nconmax,
    cone="elliptic",
    impratio=100,
    iterations=100,
    ls_iterations=50,
    use_mujoco_contacts=use_mujoco_contacts,
)
