# Newton: From Rigid Robots to Coupled Simulation

First build and step a small Newton scene. Then migrate the MuJoCo two-cube receiving-box task through `SolverMuJoCo`. The advanced exercise uses the robot's **gripper** to pick up two cubes, a free cloth patch and a cable, carry each object to a box, and release it inside. MuJoCo owns the robot and cubes; VBD owns the cloth and cable.

The advanced coupled task picks and places all four objects with the gripper; the default migration and final-check notebooks use only the two rigid cubes. See [VERIFICATION.md](VERIFICATION.md) for the tested robot/device combinations and notebook results.

## Setup

Use **Python 3.12** in this tutorial’s own environment. The included MuJoCo baselines use this same pinned stack.

From `tutorials/newton`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r notebooks/newton/requirements.lock.txt
source .venv/bin/activate
PXR_WORK_THREAD_LIMIT=1 jupyter lab --notebook-dir notebooks
```

The optional hash lock fixes transitive packages as well as the physics stack. To resolve from the shorter direct requirements instead, replace the install command with `uv pip install --python .venv/bin/python -r notebooks/newton/requirements.txt`. The latter allows compatible notebook/tooling versions to vary. Select the Newton interpreter as the notebook kernel; the first cell prints it and its package versions.

The physics/import pins are **Newton 1.6.0, Warp 1.17.0, MuJoCo/MuJoCo Warp 3.12.0, newton-usd-schemas 0.5.0, usd-core 26.3 and NumPy 2.5.3**. First asset downloads and kernel compilation take longer than later runs.

For the supplied Python 3.12 binary stack on **Linux x86-64**, use **glibc 2.35 or later**: the Open3D 0.20.0 wheel included by `newton[examples]` sets that minimum. NumPy 2.5.3's Linux wheels are tagged glibc 2.27/2.28, so NumPy does not impose the higher limit. The lock also resolves on macOS ARM64, where physics runs on CPU; this is not a claim of support for every Linux architecture or libc. See the [Open3D](https://pypi.org/project/open3d/0.20.0/#files) and [NumPy](https://pypi.org/project/numpy/2.5.3/#files) wheel lists.

**OpenUSD startup:** with usd-core 26.3, launch Python/Jupyter with `PXR_WORK_THREAD_LIMIT=1` and restart any existing kernel. The helper checks the actual Work concurrency limit before importing a robot USD. This version-specific workaround follows native parser failures on macOS and Linux; setting it after `pxr` loads is too late. On PowerShell, set `$env:PXR_WORK_THREAD_LIMIT="1"` before launching, and use `.venv\Scripts\Activate.ps1` to activate. Windows has not been validated.

The lock is retained byte-for-byte from the validated companion. To refresh it deliberately, run `uv pip compile notebooks/newton/requirements.txt --python-version 3.12 --universal --generate-hashes --no-annotate -o notebooks/newton/requirements.lock.txt` from the tutorial directory and repeat the checks before adopting changed versions.

## Learning path

Open the notebooks under `notebooks/newton/part3/` in order:

| Notebook | What you build or verify |
|---|---|
| [01 — Newton](part3/01__newton_fundamentals.ipynb) | A small scene, state/control/contact ownership, solver choice and replicated worlds |
| [02 — MuJoCo to Newton](part3/02__mujoco_to_newton.ipynb) | Pick and place both cubes in the box, with mapped joint targets and Newton state |
| [03 — Clean the table](part3/03__clean_the_table.ipynb) | The bin, free cloth, rod, solver ownership and coupled loop for sequential gripper manipulation |
| [04 — Final check](part3/04__final_check.ipynb) | Both CPU box tasks by default; CUDA and coupled checks opt-in |
| [05 — Benchmark](part3/05__migration_benchmark.ipynb) | Preflight by default; opt into matched CPU/GPU box batches |

Explanations live in the notebooks; exercises live in external Python files. Complete the `TODO Step` markers in your editor and compare with `part3/solutions/`. **`REFERENCE = True`** selects reference scripts directly. Set it to `False` to check your completed exercises. No cell copies a solution over your work, and backup cells preserve existing backups.

An unfinished exercise must fail. Notebook 03 recognizes only the five documented `NotImplementedError` messages for Steps 0–4; missing dependencies, download problems and unrelated exceptions are not expected TODO outcomes. A reference run does not grade student files.

## The box comparison

See [BENCHMARK.md](BENCHMARK.md) for the full 40-second, 1–2048-world protocol, raw failures and matching CPU/GPU results. [Current notebook evidence](notebook-validation/2026-10-10/README.md) distinguishes fresh box/preflight executions from the retained October 4 sphere/coupling runs. [Viewer validation](VIEWER_VALIDATION.md) covers finite rigid-task viewers separately from measured throughput.

## The coupled gripper task

Solver coupling means advancing interacting parts with different solvers while exchanging their motion and contact reactions.

| Owner | Simulated content |
|---|---|
| `SolverMuJoCo` | Robot articulation, both gripper jaws and the two cubes |
| `SolverVBD` | Cable segment bodies/rod joints and free cloth particles/triangles |
| `SolverCoupledProxy` | Both jaw interfaces and relevant cube proxies, with force/torque feedback to MuJoCo |
| Shared static geometry | Table, ground and the receiving box's floor and four walls |

Each dynamic object has one owner. Source-owned robot/cube contacts are resolved in MuJoCo; the VBD view handles material interactions against the proxies. Do not integrate a cube independently in both entries. VBD must still integrate the cable bodies it owns, so `integrate_with_external_rigid_solver=True` is inappropriate for that entry.

The dynamic robot comes from the official structured USD. The separate Menagerie MJCF model supplies scratch inverse kinematics and reBot's robot-only gravity feedforward through bounded servo targets. Newton advances the actual scene. The gripper must acquire each payload through opposing contacts, lift the whole object, carry it, open, and move clear. No payload is welded to a jaw, repositioned by the controller, or carried by a hidden attachment.

The cloth is a small, single-layer shirt silhouette with an initial fold; it is not sewn clothing. The cable is a physical rod approximation. The setup and sequence are task-specific; VBD and coupling are experimental in Newton 1.6.

### Authoring essentials

- Use `builder.add_cloth_mesh` with positive particle masses and call `builder.color()` before finalization.
- Prepare `newton.Rod` and pass it to `builder.add_rod`, recording its body, joint and shape IDs for ownership and measurements. Rod stretch/bend coefficients and discretization affect the material response.
- Register MuJoCo custom attributes before the robot import. `newton_assets.py` supplies both `SchemaResolverMjc()` and `SchemaResolverNewton()` to retain passive damping, collision filters, actuators and gripper constraints.
- Write named robot controls through the coordinate-layout `joint_target_q` mapping. Preserve separation from the free cube joints.
- The VBD entry uses compliant ALM for its rigid contacts/joints; cloth contacts have their own contact path. Contact margins, friction and solver iterations require task validation.

The controller runs at 50 Hz with physics substeps, handling the red cube, blue cube, cable and shirt sequentially. SO-101's cloth release turns the opening wrist to let draped material fall free. Contact-capacity checks reject overflow before accepting a rollout. Host IK and diagnostics make this an instructional simulation, not a throughput benchmark.

## Run reference or student implementations

With the Newton environment active, change to `notebooks/newton/part3`:

```bash
# The migrated rigid stack.
python solutions/so101_newton_solution.py --robot so101 --device cpu --viewer null --num-frames 600 --test
python solutions/so101_newton_solution.py --robot rebot --device cpu --viewer null --num-frames 600 --test

# Request both CPU task checks and save their reports/recordings.
PXR_WORK_THREAD_LIMIT=1 python final_check.py --solutions --skip-gpu --robot both --include-clean-table

# Inspect one gripper rollout on a local desktop.
PXR_WORK_THREAD_LIMIT=1 python solutions/clean_the_table_solution.py --robot so101 --device cpu --viewer gl --test
```

Remove `solutions/` and the `_solution` suffix to run completed student scripts. Omit `--solutions` from the final checker to select those scripts. A checked run stops at measured completion or fails when its `--num-frames` budget expires. The rigid-stack and box viewers stop at their frame budget; the box path checks physical completion. `--test` enables the stack assertion. Pausing does not advance physics, and cancellation never proves completion. The coupled-material viewer retains its separate behavior.

Select CUDA explicitly when testing a GPU:

```bash
PXR_WORK_THREAD_LIMIT=1 WARP_CACHE_PATH="$PWD/.generated/warp-cuda0" \
  python final_check.py --solutions --robot both --include-clean-table \
  --include-newton-cuda --device cuda:0
```

Repeat with `cuda:1` and a separate `warp-cuda1` cache for a second visible device. Each process uses one GPU. A private cache also avoids reusing incomplete files left by an interrupted run; its first use compiles kernels. `--include-newton-cuda` adds the rigid-stack checks with Newton-generated and MuJoCo-generated contacts; the native CPU baseline remains. `--skip-gpu` skips CUDA checks and forces the coupled task onto CPU. A skipped check remains unverified. The CPU path uses native MuJoCo and Warp CPU VBD; CUDA selects MuJoCo Warp for the rigid solver.

## What counts as a completed pickup

The pure `validate_clean_table_report` gate requires `task="gripper_pick_place_into_bin"`, `schema_version=2`, and exactly `red_cube`, `blue_cube`, `shirt` and `cable`. For **each** object it checks:

1. At least 10 loaded bilateral-contact frames establish the grasp.
2. At least 25 consecutive lifted frames keep the whole object at least 0.01 m above the table.
3. At least 10 carry frames include 0.05 m of measured, uninterrupted airborne travel in XY. The whole object reaches above the box before opening.
4. An actual opening fraction of at least 0.8 precedes release. Losing the grasp before that command fails the task.
5. The detached object stays fully contained and below 0.04 m/s maximum point speed for 50 frames. Containment includes cube corners, cloth particle radii and cable capsule extents, with a 3 mm numerical tolerance.

Accepted per-object grasp/release cycles must be sequential. The final gripper must clear the higher of the table and bin rim by at least 0.02 m. Finite state, measured material contacts and coupling feedback are also required. A final object center in the box, a phase label, or a zero process exit code is insufficient.

## Record and inspect states

```bash
PXR_WORK_THREAD_LIMIT=1 python solutions/clean_the_table_solution.py --robot so101 --device cpu --viewer null --test \
  --report .generated/gripper-so101.json --record .generated/gripper-so101.npz
python ../tools/render_clean_table.py \
  --report .generated/gripper-so101.json --record .generated/gripper-so101.npz \
  --output-dir .generated/gripper-figures --gif
```

Use these export commands only with a passing current-task report. The NPZ records simulated states plus `phase` and `active_object`; `joint_drive_target` distinguishes sent servo targets from nominal IK commands in `gripper_command`. The JSON's `record_sha256` binds those trajectory bytes. The renderer uses the same task gate and refuses mismatched recordings. Its default `plot` renderer shows collision geometry. Add `--renderer gl` to replay the recorded poses with the matching official robot meshes; this also checks the Newton/assets versions and model topology. Neither renderer advances physics.

The final checker saves pairs in `.generated/final_check/{reference|student}_clean_{robot}.{json,npz}`. An explicitly selected CUDA device adds `_cuda0`, `_cuda1`, etc. to the stem. Keep reference/student and device results separate. Current task evidence belongs in [VERIFICATION.md](VERIFICATION.md); no older figures are presented here as gripper-task results.

## Checks

From `tutorials/newton`:

```bash
PXR_WORK_THREAD_LIMIT=1 .venv/bin/python -m unittest discover -s test -v
.venv/bin/python notebooks/newton/tools/build_clean_scaffolds.py --check

# Execute the teaching notebooks, including the coupled task.
PXR_WORK_THREAD_LIMIT=1 NEWTON_NOTEBOOKS_EXECUTE=1 NEWTON_NOTEBOOKS_INCLUDE_FINAL=1 \
  .venv/bin/python -m unittest discover -s test -p test_notebooks.py -v
```

Set `NEWTON_NOTEBOOKS_INCLUDE_FINAL=0` to execute notebooks 01/02/04 and the default preflight-only notebook 05, excluding the coupled notebook 03. Executed copies are saved under `part3/.generated/notebooks/`; source notebooks and student files are preserved. Without the execution flags, the notebook suite runs static contracts and reports explicit execution skips. Passing static contracts does not establish a completed physics task.

Notebook 04 runs both CPU box tasks sequentially by default. Its separate stack/coupled block is disabled unless `RUN_LEGACY_AND_COUPLED=True`; each optional CPU clean-task subprocess allows up to 30 minutes before timing out. On the validation workstation, a complete CPU pickup sequence took roughly 26–29 minutes per robot, and the four notebooks took about 83 minutes in total. Slower machines or first-time compilation may exceed the checker budget. The advanced cells capture subprocess output until completion, so a quiet cell can still be running; use the CUDA commands above to check the task on a supported GPU.

## Assets, caches and licenses

Clean-table dynamics use the official SO-101 and reBot structured USD folders at the revision pinned in `newton_assets.py`. Rigid migration and scratch IK use the separate Menagerie revisions in `robots.py`. Complete robot folders are downloaded with their referenced layers and licenses; no USD import failure silently selects MJCF instead.

Automatic caches must have the pinned Git HEAD and unchanged tracked robot files. An unverified cache is left untouched. Point `NEWTON_CACHE_PATH` at a fresh directory for official USDs, or `MUJOCO_MENAGERIE_CACHE` at a fresh parent directory for automatic MJCF downloads. Deliberate Menagerie models can still be selected with `--menagerie-path`, `MUJOCO_MENAGERIE_PATH` or `NEWTON_MENAGERIE_PATH`; these overrides are not assertions that the model matches the reference asset. They do not replace the dynamic robot's USD.

SO-101 assets retain Apache-2.0; reBot retains MIT © 2026 Seeed Studio. Live robot models download into a cache; retained evidence archives may include historical model/source assets with their original licenses. See [third-party notices](THIRD_PARTY_NOTICES.md), the retained license copies, [Newton's stable documentation](https://newton-physics.github.io/newton/stable/), and the [version-fixed source](https://github.com/newton-physics/newton/tree/v1.6.0).
