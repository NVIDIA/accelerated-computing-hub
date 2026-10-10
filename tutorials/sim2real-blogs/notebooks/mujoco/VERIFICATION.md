# Article 2 verification — 4 October 2026

The optional receiving-box task was checked independently from Article 3's
Newton task. It uses two rigid cubes and a five-geom box, with the same controller
and measured acceptance checks on native MuJoCo and single-world MuJoCo Warp.

## Environment

The public `requirements.lock.txt` reproduces the tested Python 3.12 environment:
MuJoCo **3.8.0**, MuJoCo Warp **3.8.0.3**, Warp **1.15.0** and NumPy **2.5.3**.
Newton is not installed in this environment. The lock's requirements and hashes
match the validation lock; only generated comments were relocated. A dry-run
sync of the public lock required no changes to the 55-package macOS environment.
The Linux environment contains 54 packages because of platform-specific markers.

CPU checks ran on macOS ARM64. CUDA checks ran on Linux x86-64 with two NVIDIA
RTX PRO 6000 Blackwell Workstation Edition GPUs and driver **615.71.09**. Each
process selected one GPU; these checks do not measure multi-GPU scaling.

## Physical task checks

| Backend | Robot | Stack | Receiving box |
|---|---|---|---|
| Native MuJoCo CPU | SO-101 | Pass, 600 frames | Pass, 1,800 frames |
| Native MuJoCo CPU | reBot | Pass, 600 frames | Pass, 1,800 frames |
| MJWarp CUDA 0 | SO-101 | Pass, 600 frames | Pass, 2,000 frames |
| MJWarp CUDA 0 | reBot | Pass, 600 frames | Pass, 2,000 frames |
| MJWarp CUDA 1 | SO-101 | Pass, 600 frames | Pass, 2,000 frames |
| MJWarp CUDA 1 | reBot | Pass, 600 frames | Pass, 2,000 frames |

The controller runs at 50 Hz with ten physics substeps per control frame, giving
a 0.002 s physics timestep. The published box commands allow 2,000 frames, or
40 simulated seconds. Reaching that budget does not count as success.

Every final box rollout was independently audited from its 50 Hz measured samples.
Both cubes had loaded contacts on opposing jaws, sustained whole-object lift,
airborne transport, release inside the box, detached settling, and final jaw
withdrawal. The object order was checked. GPU contact forces were exported from
the GPU solve, without recomputing them with the CPU solver. Raw contact and
constraint capacities were checked after every substep; no final validation run
overflowed those capacities.

An initial 1,800-frame CUDA 0 reBot run completed both pickups and releases but
failed the settling check: a brief blue-cube speed pulse left only 35 of the
required 50 consecutive settled frames at the cutoff. The 40-second budget adds
settling margin. The physics, controller and acceptance thresholds were unchanged;
all four final CUDA box cases were checked with the longer budget.

The GPU checks also exposed a contact-export compatibility issue: MJWarp fills
`contact.geom`, while deprecated `geom1`/`geom2` fields can retain stale IDs.
The adapter now reads the modern pair, and a regression test protects that path.

## Regression scope

The 16 box tests include both real CPU reference runs, preservation of the
original stack scene, and rejection of incomplete grasps, supported dragging,
lost grips, short lifts, premature releases, corner escape, unsettled objects,
overlapping pickups and insufficient jaw withdrawal. Capacity tests cover raw
overshoots and newer sticky overflow flags. The three existing Article 2 scaffold
checks also passed.

```bash
Article_2/.venv/bin/python -m unittest discover -s tests -p test_box_task.py -v
Article_2/.venv/bin/python -m unittest discover -s tests -p test_tutorial_scaffolds.py -v
```

Student starter scripts intentionally retain their TODOs. Completed references
are executed directly; tests do not fill or overwrite the exercises. The optional
notebook cells call those references unless the reader selects a completed student
implementation. Box-task reports are written after the frame loop, so an earlier
simulation exception can stop the process before a JSON report is created.

A combined project regression run in the separate Article 3 environment also
passed **133 tests and 274 subtests**, with the four opt-in notebook-execution
tests explicitly skipped. This checked that the new tests coexist with the
Newton tests; it does not replace the independent 3.8 physics runs above.

The Accelerated Computing Hub contribution has separate MuJoCo 3.12 / Warp 1.17
requirements and its own validation record. Results here apply to the 3.8 line
above; they are not a substitute for checking that other environment. Interactive
viewers, Windows and Colab execution were not validated in this check.
