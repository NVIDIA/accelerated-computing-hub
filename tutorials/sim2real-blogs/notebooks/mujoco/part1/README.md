# Part 1 — Run MuJoCo on CPU

[Article 2 start page](../README.md) · [Next: MJWarp on GPU](../part2/README.md)

Complete the [Article 2 setup](../README.md#set-up-once) first. Run every terminal command below from the **`tutorials/sim2real-blogs/notebooks/mujoco` directory**, with the Article 2 environment active.

## Run the completed demo

```bash
python part1/solutions/so101_pick_place_solution.py --task box --sim-substeps 20
```

The SO-101 arm places the red and blue cubes in a receiving box. A viewer opens and closes after 40 simulated seconds. Success ends with:

```text
Task complete: both cubes are in the receiving box (verified).
```

Without a display, run the complete task headlessly:

```bash
python part1/solutions/so101_pick_place_solution.py --task box --sim-substeps 20 --headless-steps 2000
```

To use the reBot DevArm instead:

```bash
python part1/solutions/so101_pick_place_solution.py --task box --sim-substeps 20 --robot rebot
```

Robot assets download automatically the first time. See [viewer controls](../README.md#viewer-controls-and-results) for pausing and stopping.

## Follow the lessons (optional)

Register the [Article 2 notebook kernel](../README.md#learn-with-notebooks-optional), then open:

1. [MuJoCo fundamentals](01__mujoco_fundamentals.ipynb).
2. [CPU pick and place](02__pick_and_place.ipynb).

Select **Sim2Real (MuJoCo 3.8)** and run cells in order, beginning with setup. The pick-and-place notebook uses `REFERENCE = True` to run the completed solution headlessly for both robots. Use the terminal commands above to open a viewer.

## Complete the exercise (optional)

Edit the `TODO Step` markers in [`so101_pick_place.py`](so101_pick_place.py). This file is intentionally incomplete. Consult [`solutions/step_00_imports.py`](solutions/step_00_imports.py) and the other `step_NN` snippets for individual answers, or compare with the [completed solution](solutions/so101_pick_place_solution.py). The snippets are not standalone commands to run in sequence.

After completing the exercise, either set `REFERENCE = False` in the pick-and-place notebook and rerun its selector and task cells, or run:

```bash
python part1/so101_pick_place.py --task box --sim-substeps 20 --headless-steps 2000
```

The remaining Python files provide the scene, controller, robot definitions and viewer. The demo imports them; no separate launch is needed.

## Optional task and report

The earlier red-on-blue stacking exercise remains available with `--task stack`; its full headless run takes 600 frames. For a box-task JSON report, add `--report part1/.generated/box_report.json` to a completed run. A frame budget alone does not establish success: check the final task result.

Continue with [Part 2: MJWarp on GPU](../part2/README.md), or use the [benchmark guide](../BENCHMARK.md) when you want performance measurements.
