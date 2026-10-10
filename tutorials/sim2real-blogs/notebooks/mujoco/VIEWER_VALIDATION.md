# Viewer behavior and validation

The robot teaching viewer has a finite episode: 600 control frames for stacking, or 2,000 for placing both cubes in the box. Space pauses; Esc, Q, window close and Ctrl+C cancel without claiming task success. Camera orbit, pan and zoom remain available. Completed box runs execute their physical task checks; the legacy stack exercise prints diagnostic distances.

The renderer now uses public MuJoCo and GLFW APIs on the main thread, including deterministic context cleanup. On Linux, both robots, both tasks and native CPU/MuJoCo Warp completed six checks each: automatic completion, Escape, Q, pause/resume, window close and Ctrl+C (**48 cases**). These used an actual NVIDIA desktop context on an RTX PRO 6000 Blackwell, not software-renderer inference.

The executed Linux source was viewer candidate 3 (manifest SHA-256 `e89944197ace52ef737d576a5c8753eefe512dbdf2810e4accdcd2f36c8c02ea`). The published candidate 4 (manifest `57cebd3149ac6f463d9de3c37294287cc2e459c86ff2d0765aff3c31aa2e39da`) adds only the reviewed macOS launch correction: use ordinary environment `python` for this custom GLFW window, not `mjpython`. The Darwin guard checks the main OS thread before GLFW initialization. The Linux renderer and physics paths are unchanged by that correction; it has mocked regression coverage, but **macOS GUI execution has not been validated**. Separate sphere/passive-viewer instructions are unchanged.

The legacy SO-101 stack model also raises its line-search budget from 20 to 50, leaving its outer iteration budget unchanged. This change is restricted to that legacy scene; reBot and the receiving-box benchmark options are unchanged. It is a solver-setting fix in addition to the viewer work.

The CPU/GPU study in [the result guide](benchmark-results/2026-10-09-validated-box-gpu0/README.md) was measured before these viewer changes, at [commit `bab983ea`](https://github.com/johnnynunez/blogs/commit/bab983ea69422758f1df9bd5c672329bedeeca1a). Its raw results, failed cases and exact source archive remain unchanged. Viewer tests are functional checks, not throughput measurements. Reproduce the published timings from that commit or the archived sources; current source manifests also include the subsequent teaching changes.

The [raw viewer evidence](validation/viewer-lifecycle-2026-10-10/README.md) includes original reports, logs, source manifests and cleanup receipts.

The [legacy solver diagnostic](validation/legacy-linesearch-2026-10-10/README.md) retains the warning-producing baseline and corrected 3.12 run, including the indeterminate 3.8 outcomes.
