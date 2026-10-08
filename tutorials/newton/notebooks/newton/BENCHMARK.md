# When is migration worth it?

Measure the two-cube stacking task on your CPU and GPU with [the benchmark notebook](part3/05__migration_benchmark.ipynb) or `part3/migration_benchmark.py`. The output records the machine, packages, workload, task checks and repeated measurements together. These are community measurements of this example on the recorded hardware; they are not official product benchmarks.

The [October 8 workstation results](benchmark-results/2026-10-08-rtx-pro-6000/README.md) include recorded measurements, complete JSON reports and the hardware/load limitations for review.

Start with one world and 16 worlds. A larger GPU batch may improve aggregate throughput while a single interactive task still runs faster on CPU. The useful crossover depends on your CPU, GPU, batch size and whether the application needs a host controller or observations each frame. A result from an RTX PRO workstation does not establish that crossover on a laptop or cloud GPU.

## Choose the comparison

| Scope | What runs | What the result answers |
|---|---|---|
| `replay` (default) | Native MuJoCo and MuJoCo Warp receive the same MJCF model and recorded task commands. The CPU runs serially and with a worker pool; the GPU runs a batch. | How does simulation throughput change with world count for this rigid manipulation task? |
| `workflow` | Native MuJoCo and Newton's CPU/CUDA migration run the actual host controller for one world. | What does the complete scripted migration cost, including the controller and required state transfers? |
| `both` | Both experiments, reported separately. | How do the simulation and application measurements differ? |

The workflow experiment reconstructs the model through Newton and has its own contact and state interfaces. It is not an identical-model comparison. The replay experiment excludes controller computation: every backend consumes the same control tape. Neither experiment measures policy training, rendering, or the coupled cloth-and-cable exercise. Those workloads need their own measurements and correctness gates.

Replay compiles the common MJCF with **100 solver iterations, 50 line-search iterations and `impratio = 100` on both backends**. These settings override the source XML's values before CPU and GPU execution; the report retains the actual solver options and compiled-model hash. Compare results produced with the same configuration, rather than mixing a default XML run with the benchmark's settings.

Keep task duration fixed when changing world count. Every measured world must complete the stack with finite state and valid capacity checks; a fast failed task is not a successful benchmark. Warm-up and CUDA compilation are separated from repeated measurements, CUDA work is synchronized at timing boundaries, and the report identifies the timing scope. Compare CPU and GPU rows from the same experiment and configuration.

Replay includes the initial state reset and records the full physics state after every substep in each backend's simulation timer. GPU download and task validation have separate timings, so distinguish simulation throughput from the time until checked host results are available. Native MuJoCo uses double-precision state while the GPU replay uses single precision; identical inputs do not imply bitwise-identical trajectories. The task gates apply to both.

Workflow records both cubes' poses and velocities at each 50 Hz control frame inside the simulation timer, together with online IK, control updates and physics stepping. Native MuJoCo stores double-precision observations in RAM; Newton uses single-precision observations in RAM on CPU or VRAM on CUDA. The final host download and task validation are timed separately. Each warm-up and measured episode gets a fresh model, controller and solver outside the simulation timer; setup cost remains in the report.

The CPU replay uses MuJoCo's native [`rollout` API](https://mujoco.readthedocs.io/en/stable/python.html#rollout), including its worker pool. The GPU replay uses [MuJoCo Warp](https://mujoco.readthedocs.io/en/stable/mjwarp/) with a captured frame loop. These references explain the execution interfaces; the timing boundaries and observation policy above belong to this task-specific benchmark.

## What counts as a successful task

The shared trajectory gate checks every world at the 50 Hz control frames, with finite state and the full 12-second episode required. The red cube must clear the table by more than 1 cm for at least 10 consecutive lift/transport frames, then travel at least 5 cm within one uninterrupted airborne transport interval lasting at least 10 frames. Grounded intervals cannot be joined to create apparent airborne travel.

During the final second, the cubes must remain stacked within 15 mm in XY and 35–55 mm in height difference, with the blue cube supported on the tabletop within 5 mm. Both cubes must also satisfy all of these settling conditions:

- RMS of the point-speed bound `|v| + sqrt(3) * cube_half * |omega|` over the final second at most **0.05 m/s**, for each cube.
- Largest position difference between any two samples at most **5 mm**.
- Largest orientation difference between any two samples at most **3 degrees**.

RMS measures sustained motion; the position and orientation limits use unsmoothed maximum pairwise differences. Raw peak linear, angular and point-speed bounds remain in the results alongside RMS, so brief contact jitter stays visible. Capacity overflow and invalid time/state observations fail the task; solver iteration-limit diagnostics remain visible rather than being silently discarded. The same acceptance thresholds apply across CPU/GPU and replay/workflow measurements, while the model and recording differences remain explicit. Results from an earlier acceptance rule or timing boundary must be rerun before comparison.

## Run on your machine

Use the tutorial's [Python 3.12 setup](README.md#setup), with its unchanged hash lock. From `tutorials/newton/notebooks/newton/part3`:

```bash
python migration_benchmark.py --preflight
python migration_benchmark.py --robot so101 --worlds 1 16 --device cuda:0 \
  --repeats 3 --warmups 1 --cpu-threads 4 --output-dir .generated/benchmark-so101
```

The preflight inspects the environment without producing performance numbers. Select `--robot rebot` to repeat with the other robot. Set `--cpu-threads` to the CPU pool size you want to compare; this is part of the reported configuration, not permission to consume every core on a shared machine. `--cpu-only` runs the CPU measurements explicitly and leaves the CPU/GPU comparison unmeasured.

Use `--scope workflow` for the single-world host-controller comparison, or `--scope both` to collect both experiments. Select `--device cuda:1` to use a second visible device in a separate run. Each invocation uses one GPU; this is not a multi-GPU scaling benchmark.

After the small batch passes, a workstation sweep can use `--worlds 1 16 64 256`. Reduce world count if memory is limited. Do not shorten the task or disable validation to obtain an apparently faster result. Use at least five repetitions for a result intended for publication, and retain the warm-up, configuration and individual samples.

`--preset developer` selects the small batch; `--preset workstation` selects the larger sweep. Explicit `--worlds` takes precedence. `--max-trajectory-mib 1024` is the default per-row trajectory budget; an oversized row fails before allocation instead of shortening the episode. This budget covers the stored trajectory, not every solver allocation or the machine's total memory demand.

Each output directory contains:

- `results.json`: machine and software metadata, configuration, measurements and task outcomes.
- `summary.csv` and `summary.md`: tables for comparison and review.
- Raw worker reports, retaining the evidence behind the summary.

Use a fresh output directory for each run. Preserve the whole directory when sharing a result. Record the source commit and any local changes, rather than citing only a mutable branch name. Inspect all task outcomes and configuration fields before taking a number from the summary.

## Google Colab

Open [the benchmark notebook on Colab](https://colab.research.google.com/github/johnnynunez/accelerated-computing-hub/blob/feature/newton-task-benchmark/tutorials/newton/notebooks/newton/part3/05__migration_benchmark.ipynb), select a GPU runtime, then run the setup and preflight cells. The notebook's `REPO_URL` and `REF` select the source checkout. Before publishing a comparison, replace `REF` with the full commit shown by a successful setup run and repeat from that revision.

The setup installs [uv 0.12.5](https://pypi.org/project/uv/0.12.5/) into a separate tools environment, creates a Python 3.12 environment, and installs the same hash-locked dependencies as the local tutorial. Benchmark commands run in subprocesses using that interpreter; the notebook does not replace Colab's Python or physics packages. The hosted notebook only displays the reports. Set `RUN_BENCHMARK = True` after reviewing the printed configuration, then run the measurement and export cells.

Colab's [available GPU types and usage limits vary](https://research.google.com/colaboratory/faq.html). The actual assigned device and driver determine whether the pinned stack runs; notebook metadata requesting a T4 is not a compatibility check. Download the result archive before the runtime disconnects. A Colab runtime supplies its own CPU baseline, which is not your laptop CPU.

This launch path is provided for validation. No Colab GPU or lower-end device is claimed as tested merely because setup code is included.

## NVIDIA Brev

On an existing Brev GPU instance, use the same local setup and open the benchmark notebook in JupyterLab. Brev's [Launchables support a public Git repository with VM setup, or Docker Compose](https://docs.nvidia.com/brev/concepts/launchables). Select one available GPU, record the instance CPU and memory as well, and run the small batch first. T4 and L4 are possible validation targets when available; neither is a measured minimum requirement for this tutorial.

The tutorial's [Docker/Compose configuration](../../brev/docker-compose.yml) is another option. It retains its CUDA 13.1 base, whose host-driver requirement can be stricter than installing the pinned Python wheels directly in a VM. Warp 1.17's [PyPI wheels use CUDA 12.9](https://github.com/NVIDIA/warp/discussions/1886); do not infer wheel requirements from a different Warp release or silently change the physics versions to make a comparison run.

The benchmark needs no interactive viewer, hosted model service, or access token. It can run in a Jupyter terminal with the command above. Save the complete output directory before stopping the instance. A local Docker test does not establish that a hosted Brev deployment has been validated.

## Interpret and publish results

Read task success before throughput. Compare the native CPU serial baseline, the CPU worker-pool baseline and the GPU at equal world count and task duration. Report the CPU worker count beside a speedup; GPU versus one CPU thread and GPU versus several CPU workers answer different questions. Repeated timings describe variability on that machine, not a guarantee for another device.

Use the workflow results when deciding whether to migrate a one-world application with host IK and per-frame observations. Use replay results when considering batches with controls already available. Neither result alone predicts a device-resident learning workflow. Account for one-time setup and compilation separately if your application runs only a short episode.

Before adding numbers to the article, have the team review the raw reports and reproduce them on the selected developer GPU. Include the exact CPU, GPU, RAM, driver, package versions, source revision, world counts, worker count, repetitions, timing scope and task outcome. Describe other applications using the machine during the run; shared-machine timings are observations under that load. Keep untested hardware and cloud runtimes explicitly unvalidated, and do not label community observations as product guarantees.
