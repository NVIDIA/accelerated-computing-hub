# Viewer validation on the workstation

20 box lifecycle checks across both robots and CPU/CUDA, plus four automatic rigid-stack runs with the original physical assertions. Each accepted case has raw logs, events, source hashes and verified process cleanup from the NVIDIA desktop on the RTX PRO 6000 Blackwell. Cancellation checks are clean-exit tests, not successful manipulation episodes.

The box control tests and stack completion tests retain their distinct scopes. Coupled-material controls are not covered by this viewer matrix.

These are functional checks, not performance measurements. The [viewer notes](../../VIEWER_VALIDATION.md) identify the separate measured-source commit. Original benchmark failures and timings remain unchanged.

[Raw evidence](raw-evidence.tar.gz) includes the complete method, original README, per-file manifest and retained failures. Archive SHA-256: `c486d3624bdc9ac413df42503ecb048e244d769cad35aec2a914ee5468ca6470`. The [source manifest](source-manifest.json) is an unchanged copy of the manifest inside the archive; extract the archive before verifying its member paths. Compiler caches are inventoried but excluded. No native document snapshots, document images or authentication contents are included.
