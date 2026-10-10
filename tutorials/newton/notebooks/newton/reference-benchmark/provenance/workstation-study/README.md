# Full ALOHA reference sweep — prepared, not launched

Root may launch this stage only after accepting the complete reference-stage1
CPU/GPU smoke reports. No smoke outcomes were assumed or used while preparing
this stage. No remote transfer, hardware job or dependency change was performed.

The self-contained plan has **36 sequential configurations**: both pinned blog
stacks, CPU and GPU, at 1, 16, 32, 64, 128, 256, 512, 1024 and 2048 worlds. Each
configuration executes the complete 1001-step recording, one full warmup and
five measured episodes. CPU uses up to 32 native rollout threads; GPU uses only
`cuda:1`. Every job has `prepare_output_dir: false`. Existing pinned interpreters,
Warp cache paths and the shared workstation lock match reference-stage1.

`sources/frozen-v3/` is the exact reviewed source/assets snapshot. Its 189 source
manifest entries were checked after copying. The supervisor is an unchanged
copy of the existing supervisor. There is no shortened replay, output
downsampling, alternative model, solver retuning or new memory cutoff.

## Memory

At 2048 worlds, full GPU history is exactly **393,609,216 bytes / 375.375 MiB**:
2048 × 1001 × 48 FULLPHYSICS fields × 4 bytes. This is below the existing box
study's 1024 MiB trajectory budget and about 0.384% of the known selected GPU's
97,887 MiB capacity. The existing budget is only a comparison; it is not a new
reference acceptance limit. Native float64 history occupies **750.75 MiB**;
native expanded controls and additional buffers are itemized in
`memory-estimate.json`.

Those values cover history, not total solver memory. Model, contacts, workspaces
and CUDA graphs add version-dependent allocations. Historical device capacity
does not establish current free memory. Root must preserve fresh hardware/load
and allocation receipts. An allocation failure remains a failed configuration;
do not silently reduce worlds or shorten the episode.

## Result interpretation

Require every accepted report to have status `complete`, one accepted warmup
and five accepted measured samples. The supervisor's zero exit code alone does
not establish physical acceptance. Failed configurations retain their diagnostics
and null accepted timings. The fixed source includes identical host physical
health checks, actual model-upload audits and capacity/clock checks.

Pair CPU/GPU reports only at matching stack, world count, model/source identity,
control tape, complete replay length and settings. Do not combine the two version
stacks or quote CPU1/GPU2048 as a matched-batch speedup. The measurement is a
native MuJoCo/MJWarp replay pipeline with float64/float32 respectively, including
full state recording. It is neither Newton API timing nor proof that the blogs'
two-cube task succeeds. Full interpretation and upstream provenance remain in
`sources/frozen-v3/README.md`.

`launch-source-manifest.json` fixes the prepared bytes. Any source or protocol
change requires a new stage, preserving this one and any resulting evidence.
