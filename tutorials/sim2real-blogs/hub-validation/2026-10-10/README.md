# Hub relocation checks — 10 October 2026

The corrected Docker image built from this tutorial passed **151 tests, with one explicit skip**, in 463.54 seconds on Linux / Python 3.12.15. Tests used one RTX PRO 6000 Blackwell device and the measured MuJoCo 3.8.0 / MuJoCo Warp 3.8.0.3 / Warp 1.15.0 lock. Both robot box tasks passed on CPU and GPU, and the benchmark notebook executed preflight in a real kernel without measurements.

The skipped test is the optional CPU viewer integration harness. Mocked viewer regressions passed; no new desktop/GL test is claimed here. The full performance study and prior actual GUI evidence retain their original provenance.

The first image also built, but test collection failed because an imported asset-cache test expected a helper that belongs to the Newton environment. The replacement tests target the actual MuJoCo helper. Only that test file changed among executable inputs between snapshots; physics, notebooks, dependencies and Docker recipe were unchanged.

[The receipt](receipt.json) binds the [raw logs, invocation records and source-file manifests](evidence.tar.gz), including the first failure. [Corrected run records](runs-v2.json) retain exact commands and exit status. The build used an isolated context with the tutorial, shared Brev scripts and `.dockerignore`; a live Compose/Brev deployment and Colab were not exercised. Notebook bootstrap SHA pins and links are subsequent publication-only changes, checked separately.
