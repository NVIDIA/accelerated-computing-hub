# Viewer behavior and validation

The box viewer stops after its full 2,000-frame episode and runs the physical task checks. The rigid-stack viewer stops at its frame budget (600 by default); `--test` enables the physical stack assertion. Pausing renders without advancing physics. Cancellation exits without claiming completion. The coupled-material examples keep their existing behavior.

Viewer candidate 1 (manifest SHA-256 `822dc3408380b4979b3a9498331380ac8e8184a428aedbcbfed5d56f61e98fd4`) was checked on an actual NVIDIA desktop context on an RTX PRO 6000 Blackwell. Both robots on CPU and CUDA passed all five box lifecycle cases: automatic completion, Escape, pause/resume, window close and Ctrl+C (**20 cases**). Each of the four robot/backend combinations also passed an automatic 600-frame stack run with `--test`, natural exit and verified process cleanup. These are Linux results; macOS GUI execution has not been validated.

The CPU/GPU study in [the result guide](benchmark-results/2026-10-09-validated-box-gpu0/README.md) was measured before these viewer changes, at source commit `ee9119a0d45bf887168fa30ba73c174d3f5538c0` retained in that package. Its raw results, failed cases and exact source archive remain unchanged. Viewer tests are functional checks, not throughput measurements. Reproduce the published timings from that commit or the archived sources; current source manifests also include the subsequent teaching changes.

The [raw viewer evidence](validation/viewer-lifecycle-2026-10-10/README.md) includes original reports, logs, source manifests and cleanup receipts.
