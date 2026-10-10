# Viewer validation on the workstation

48 checks across both robots, CPU/MuJoCo Warp, stacking and box tasks, including automatic completion, Escape, Q, pause/resume, window close and Ctrl+C. Each accepted case has raw logs, events, source hashes and verified process cleanup from the NVIDIA desktop on the RTX PRO 6000 Blackwell. Cancellation checks are clean-exit tests, not successful manipulation episodes.

The Linux GUI matrix used candidate 3. Candidate 4 adds the reviewed Darwin main-thread launch correction; actual macOS GUI execution is untested. The original passive-viewer crash and the intentionally cancelled slow-render attempt remain in the archive.

These are functional checks, not performance measurements. The [viewer notes](../../VIEWER_VALIDATION.md) identify the separate measured-source commit. Original benchmark failures and timings remain unchanged.

[Raw evidence](raw-evidence.tar.gz) includes the complete method, original README, per-file manifest and retained failures. Archive SHA-256: `db893ee08312daad369a69512f611dffd6b93b45de38d73dc1a0d516c09d704f`. The [source manifest](source-manifest.json) is an unchanged copy of the manifest inside the archive; extract the archive before verifying its member paths. Compiler caches are inventoried but excluded. No native document snapshots, document images or authentication contents are included.
