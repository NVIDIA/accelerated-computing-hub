# A reproducible CPU/GPU manipulation reference

This companion benchmark replays the upstream ALOHA pot-lifting controls through native MuJoCo on CPU and MuJoCo Warp on one GPU. It measures the same scene, initial state, control tape, integration count and solver settings at 1, 16, 32, 64, 128, 256, 512, 1024 and 2048 environments. It is a community benchmark you can reproduce on your machine. It is separate from the blogs' two-cube task and does not measure the Newton API.

Each batch performs one complete warmup and five measured episodes. An episode integrates 1001 steps of 2 ms. The CPU uses a persistent native thread pool; the GPU keeps the controls resident and replays captured groups of ten steps. No rendering, online inverse kinematics or policy training is included. Both backends record full state histories, using native float64 on CPU and float32 on GPU.

## Run one pinned stack

Use a Linux machine with an NVIDIA GPU and Python 3.12. Create separate environments for the two package sets. From this folder:

```bash
python3.12 -m venv .venv-blog2
.venv-blog2/bin/python -m pip install -r requirements-blog2.txt
.venv-blog2/bin/python fetch_reference_assets.py --output "$HOME/.cache/blogs-reference/aloha-pot"
.venv-blog2/bin/python run_reference.py --stack blog2 --assets "$HOME/.cache/blogs-reference/aloha-pot" --output results --device cuda:0
```

Repeat with `.venv-blog3`, `requirements-blog3.txt` and `--stack blog3` to evaluate the second version set. The Blog 3 environment includes Newton 1.6.0 to reproduce its package environment; this reference still calls MuJoCo and MuJoCo Warp directly. Both stack runs can write to the same `results` folder because their case names differ. Use a new output folder when repeating a stack so that earlier evidence remains available.

The default run is 18 serial cases per stack: both backends at every requested count. `resource-config.json` records the fixed protocol. The wrapper uses at most 32 available CPU cores by default, records the actual worker count, and accepts `--workers` and `--device` for a documented machine configuration. Use `--dry-run` to inspect all commands before execution. Use the same CPU worker count for both stack runs when collecting a combined summary.

To start with a bounded diagnostic, add `--counts 1,16` and choose a separate output folder. This still runs full episodes and five measurements; it simply reduces the number of batches. The full GPU state history at 2048 worlds is approximately 375 MiB, in addition to solver data, contacts and graph workspaces. No claim is made that every batch fits or succeeds on every GPU. Preserve allocation failures and other unsuccessful outcomes.

## Compare accepted results

Use the device and actual worker count recorded in `blog2-run-plan.json` and `blog3-run-plan.json`:

```bash
.venv-blog2/bin/python collect_reference_sweep.py --results-root results --device cuda:0 --workers 32 > summary.json
.venv-blog2/bin/python -m pip install -r requirements-figures.txt
.venv-blog2/bin/python make_figures.py --summary summary.json --stack blog2 --output figures-blog2
.venv-blog2/bin/python make_figures.py --summary summary.json --stack blog3 --output figures-blog3
```

Replace `32` with the actual requested worker count. The collector reports missing cases and failures explicitly. Figures require nine accepted CPU/GPU pairs for the selected stack; `--allow-partial` creates an explicitly labeled diagnostic chart with gaps. A speedup compares CPU and GPU at the same environment count and package stack. It is never derived by dividing a one-environment CPU result by a 128-environment GPU result.

The primary timer includes reset, control replay, integration, state-history recording and dispatch/completion. The checked timer also includes output collection and numerical/physical checks. Setup/JIT and artifact file writes are excluded. Reports retain every attempted episode. A failed check excludes that entire case from accepted medians and comparisons. Checks cover finite states, clocks, quaternion norms, coarse motion bounds, GPU capacities and source-derived final pot/lid heights. These checks do not prove successful contact-based grasp and hold. Full intermediate histories are checked in memory but only endpoints and health summaries are saved.

Publish the hardware inventory, run plans, raw JSON, endpoint NPZ files, logs and collector output with any figure. The wrapper records CPU, RAM, GPU identity and driver information. Review background utilization separately when interpreting measurements.

## Colab and Brev

In Colab, select a GPU runtime, copy this folder into the session, and use notebook shell cells to run one pinned environment's commands. Start with `--counts 1,16`; let the wrapper select the available CPU count, and save the result folder before ending the runtime. GPU types and limits vary according to [Colab's documentation](https://research.google.com/colaboratory/faq.html). This recipe has not yet been validated on Colab or a lower-end GPU.

For Brev, create and connect to a GPU instance using the [Brev quickstart](https://docs.nvidia.com/brev/getting-started/quickstart), then run the same Linux commands. Begin with the bounded counts and retain the generated hardware inventory. No Brev deployment or specific GPU configuration is claimed as validated here.

See `ASSET_ORIGINS.md` before redistributing assets. The fetcher pins upstream commits and file hashes; assets are downloaded into a cache instead of being bundled in this code folder.
