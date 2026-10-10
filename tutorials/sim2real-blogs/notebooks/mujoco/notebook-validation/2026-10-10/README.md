# Notebook validation

Four published notebook payloads passed actual Python kernels on the shared Linux workstation on 10 October 2026: MuJoCo Fundamentals, CPU pick and place, MJWarp pick and place, and the CPU/GPU benchmark preflight. Both pick-and-place lessons checked both robots with their default reference implementations. The benchmark notebook retained `RUN_BENCHMARK=False`; it collected no performance measurements.

Fundamentals produced finite 480×640 RGB pixels. The launch explicitly selected `MUJOCO_GL=egl`, with CUDA device remapping unset. The pixel evidence does not identify the GL vendor or establish visual QA.

The [evidence archive](notebook-evidence.tar.gz) retains executed notebooks, original inputs, logs, task/preflight reports, package/interpreter identities, and kernel/worker cleanup receipts. The [manifest](manifest.json) binds every member. Current notebook code matches the executed code; subsequent Markdown wording edits are recorded separately from full-file identity. Original published notebooks keep empty outputs.

These are notebook integration checks, separate from the measured benchmark and desktop viewer validation. Deliberately unfinished student exercises, optional legacy runs, fresh Colab/Brev installation, macOS graphics and lower-end GPUs are not covered.
