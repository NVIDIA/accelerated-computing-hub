# Legacy SO-101 solver warning check

On MuJoCo/MuJoCo Warp 3.12, the legacy SO-101 stack scene at 10 outer iterations and 20 line-search iterations emitted 349 line-search warnings and one outer-iteration warning during 6,000 steps. Changing only the line-search limit to 50 produced no exhaustion warnings or flags in the complete check. reBot already used 100/50. This is a legacy-scene correctness diagnostic, not a receiving-box benchmark.

The four MuJoCo 3.8 checks remain **indeterminate** for exhaustion because that version lacks the required telemetry; silence is not counted as convergence. All eight outcomes remain in the [raw archive](raw-evidence.tar.gz), with original logs, endpoint arrays, source/assets, method and licenses. Compiled model binaries remain retained separately and their sizes and hashes are listed in the archive.

Archive SHA-256: `7f9d1e2832794362676ba78611be9a134030e1e8a08866a5e462ecb46bec6400`. The [source manifest](source-manifest.json) is an unchanged copy from inside the archive. The published solver-option change is limited to the legacy SO-101 scene; receiving-box settings are unchanged.
