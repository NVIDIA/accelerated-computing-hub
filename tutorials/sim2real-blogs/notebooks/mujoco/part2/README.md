# Part 2 — Run MJWarp on GPU

[Article 2 start page](../README.md) · [CPU example](../part1/README.md) · [Benchmarking](../BENCHMARK.md)

Complete the [Article 2 setup](../README.md#set-up-once) first. This demo requires an NVIDIA GPU with CUDA. Run every terminal command below from the **`tutorials/sim2real-blogs/notebooks/mujoco` directory**, with the Article 2 environment active.

## Run the completed demo

```bash
python part2/solutions/so101_mjwarp_solution.py --task box --sim-substeps 20 --device cuda:0
```

This runs the same two-cube task as the CPU example, with physics on the GPU. The viewer closes after 40 simulated seconds. The first run compiles kernels and can take several minutes. Success ends with:

```text
Task complete: both cubes are in the receiving box (verified).
```

Without a display, run the complete task headlessly:

```bash
python part2/solutions/so101_mjwarp_solution.py --task box --sim-substeps 20 --device cuda:0 --headless-steps 2000
```

To use the reBot DevArm instead:

```bash
python part2/solutions/so101_mjwarp_solution.py --task box --sim-substeps 20 --device cuda:0 --robot rebot
```

Use `--device cuda:1` to select a second visible GPU. See [viewer controls](../README.md#viewer-controls-and-results) for pausing and stopping.

## Follow the lesson (optional)

After the two CPU notebooks, open [MJWarp](03__mujoco_warp.ipynb). Select **Sim2Real (MuJoCo 3.8)** and run cells from top to bottom, beginning with setup. See the [complete notebook order and kernel setup](../README.md#learn-with-notebooks-optional).

The notebook uses `REFERENCE = True` to run the completed solution headlessly for both robots. Use the terminal commands above to open a viewer. No benchmark is required to finish the main task demonstration.

## Complete the exercise (optional)

Edit the `TODO Step` markers in [`so101_mjwarp.py`](so101_mjwarp.py). This file is intentionally incomplete. The files under `solutions/step_NN_*.py` are answer snippets for individual steps; they are not programs to execute in sequence. The [completed solution](solutions/so101_mjwarp_solution.py) shows how the steps fit together.

After completing the exercise, either set `REFERENCE = False` in the notebook and rerun its selector and task cells, or run:

```bash
python part2/so101_mjwarp.py --task box --sim-substeps 20 --device cuda:0 --headless-steps 2000
```

The other Python modules are imported helpers or optional benchmark tools. You do not need to run them to use the demo.

## Measure performance (optional)

The demo checks the task in one world. For measurements, choose a command in the [benchmark guide](../BENCHMARK.md):

| Tool | What it measures |
| --- | --- |
| `migration_benchmark.py --task box` or [CPU/GPU benchmark notebook](04__cpu_gpu_benchmark.ipynb) | The complete box-task replay on CPU and GPU |
| `--benchmark` on the solution script, or `scaling_study.py` | Physics throughput with fixed controls in the earlier stacking scene |

The benchmark notebook is the optional fourth notebook. It starts with `RUN_BENCHMARK = False`, so Run All performs setup and preflight without collecting timings. Do not combine `--benchmark` with `--task box` on the demo script; use `migration_benchmark.py` for that measurement.

For a box-task JSON report, add `--report part2/.generated/box_report.json` to a completed demo run. The earlier stacking exercise remains available with `--task stack`; its full headless run takes 600 frames.
