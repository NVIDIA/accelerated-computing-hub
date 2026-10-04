# Verification scope

The full robot task was validated before this tutorial was relocated into the Hub. The relocation preserves the physics and task gate, and separately checks the new paths and notebook packaging. These are correctness checks, not throughput measurements.

## Original full validation — 4 October 2026

A fresh hash-locked Python 3.12.13 environment on Linux x86-64 ran all four notebooks: **32 code cells, no errors**, in about 83 minutes. The regression suite passed **117 tests**, with four notebook-execution tests explicitly skipped in that separate regression invocation. Both selected-device suites passed **12 checks** each; each suite includes native CPU baselines as well as CUDA checks.

The host had two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs and driver 615.71.09. All devices used the pinned Newton 1.6.0 / Warp 1.17.0 / MuJoCo and MuJoCo Warp 3.12.0 stack. The seven full gripper rollouts were independently checked against their recorded states:

| Execution | Robot | Frames | Simulated time | Result |
|---|---|---:|---:|---|
| Clean-table notebook, CPU | SO-101 | 3150 | 63.00 s | Pass |
| Final-check notebook, CPU | SO-101 | 3150 | 63.00 s | Pass |
| Final-check notebook, CPU | reBot | 3110 | 62.20 s | Pass |
| CUDA 0 | SO-101 | 3145 | 62.90 s | Pass |
| CUDA 0 | reBot | 3110 | 62.20 s | Pass |
| CUDA 1 | SO-101 | 3145 | 62.90 s | Pass |
| CUDA 1 | reBot | 3110 | 62.20 s | Pass |

Every rollout established opposing loaded jaw contacts, whole-object lift, continuous airborne carry, commanded opening, detached settling inside the box, and final jaw clearance for all four objects. Capacity overflow bits were zero. reBot CUDA reports retain flag 1024 (`LS_ITERATIONS`); it is a solver iteration-limit diagnostic, not a capacity-overflow bit, and is not erased from the records.

The canonical [SO-101](assets/clean-table-so101.json) and [reBot](assets/clean-table-rebot.json) reports and NPZ recordings are from CUDA 0. Their six-panel PNGs and GIFs replay those exact states without advancing physics. [The asset manifest](assets/clean-table-results.json) retains its original source-path names and byte hashes. [The validation summary](original-validation.json) records the seven rollout identities and original evidence hashes without local machine paths.

## Hub relocation

[relocation-manifest.json](relocation-manifest.json) maps original files and all 32 executable cell sources to their Hub paths. All robot dynamics, controllers, material authoring, acceptance gates and frozen MuJoCo/MJWarp Python baselines are unchanged. The only runner edits replace two `Article_2` baseline directory names with `mujoco` in `final_check.py`.

The four renamed notebook setup cells adapt directory/requirements path strings; the other 28 executable cells are byte-identical to the validated source. Instructional Markdown and standard Hub metadata were updated for the new location. The renderer already includes the reviewed portrait-layout amendment; it does not alter the simulation. The hash lock and canonical artifacts remain byte-identical to the original validated package. Only setup comments in the direct requirements file changed; all dependency constraints are preserved.

Relocation checks on macOS ARM64 with the pinned Python environment:

- Regression suite: **117 passed, 4 explicitly skipped**, with 269 subtests passing.
- Generated teaching scaffolds: **7 files verified**.
- First two relocated notebooks: **21 code cells executed, no errors**, in 23.36 seconds; source notebooks and student files preserved.
- Compose configuration and `brev/test.bash` shell syntax: passed static validation.
- Hub notebook format, local links and final byte mapping: passed.

The long coupled tasks were not rerun merely for relocation. The default CPU notebooks do not establish CUDA behavior on a reader's hardware; use explicit selected-device checks when verifying another environment.

## Container checks

The Docker recipe built successfully on the Linux workstation with the CUDA 13.1 / Ubuntu 22.04 base and Python 3.12.15. Installation used the unchanged transitive hash lock; `pip check` reported no broken requirements. The default `brev/test.bash` suite passed **119 tests**, with the two long notebook tests explicitly skipped, in **178.67 seconds**. The first two notebooks executed all 21 code cells inside the container.

The build used an isolated context containing only the common Brev scripts, this tutorial, the root README/license and `.dockerignore`; no host environments, model caches, credentials or audit directories were copied. Final documentation/comment-only edits were copied into the image afterward, with executable Python and notebook cell sources verified unchanged. A live Brev Compose deployment and Colab remain unverified.
