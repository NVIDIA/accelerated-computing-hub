# Notebook validation

Three published notebook payloads passed actual Python kernels on the shared Linux workstation on 10 October 2026: the benchmark preflight, robot migration, and final box check. The two box notebooks each validated SO-101 and reBot on CPU, including the original 40,000-step, physical-task, capacity and trajectory checks. GPU/legacy/coupling options stayed disabled; the benchmark retained `RUN_BENCHMARK=False` and collected no performance measurements.

The sphere and Solver Coupling notebooks reuse successful 4 October executions. Their executable cells still match exactly, as do the nine audited coupling runtime files. Those tasks were not rerun on 10 October. The archive retains both executed notebooks, the original execution/source receipts, and the gripper-based two-cube, cable and shirt reports and recordings. Historical summaries also describe older migration/final checks; current coverage for those comes from the new kernels.

The [evidence archive](notebook-evidence.tar.gz) preserves exact source/output bytes, logs, reports, trajectory files, package/interpreter identities and cleanup receipts. The [manifest](manifest.json) binds every member and distinguishes fresh execution from source-identity reuse. Published notebooks retain empty outputs.

These checks are separate from measured benchmarks and desktop viewer validation. Student TODOs, fresh hosted installation, macOS graphics and lower-end GPUs are not covered.
