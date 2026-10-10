#!/usr/bin/env python3
"""Plot accepted same-count pairs from collect_reference_sweep.py JSON."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COUNTS = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True, type=Path)
    p.add_argument("--stack", required=True, choices=["blog2", "blog3"])
    p.add_argument("--output", required=True, type=Path, help="Fresh output directory")
    p.add_argument("--allow-partial", action="store_true", help="Explicit diagnostic chart; missing pairs remain gaps")
    args = p.parse_args()
    summary = json.loads(args.summary.read_text())
    if summary.get("expected_reports") != 36 or summary.get("expected_pairs") != 18 or not summary.get("collector_sha256"):
        p.error("Use the strict collector's summary JSON")
    pairs = {x["worlds"]: x for x in summary["pairs"] if x["stack"] == args.stack and x["accepted"] is True}
    reports = {(x["backend"], x["worlds"]): x for x in summary["reports"] if x["stack"] == args.stack and x["accepted"] is True}
    if not pairs or (set(pairs) != set(COUNTS) and not args.allow_partial):
        p.error("Nine accepted same-count pairs are required; --allow-partial is for explicitly labeled diagnostics")
    if args.output.exists():
        p.error("Preserve previous figures and select a new output directory")
    for worlds in pairs:
        if any((backend, worlds) not in reports for backend in ("cpu", "gpu")):
            p.error("Accepted pair lacks accepted CPU/GPU evidence")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for backend, color, label in [("cpu", "#4B5563", "Native MuJoCo CPU (up to 32 threads)"), ("gpu", "#0072B2", "MuJoCo Warp (one RTX PRO 6000)")]:
        y = [reports[(backend, n)]["metrics"]["world_steps_per_second"] if n in pairs else float("nan") for n in COUNTS]
        axes[0].plot(COUNTS, y, marker=("s" if backend == "cpu" else "o"), color=color, label=label)
    for key, color, label in [("simulation_speedup_cpu_over_gpu", "#0072B2", "Simulation + state history"),
                              ("checked_speedup_cpu_over_gpu", "#D55E00", "Including collection + checks")]:
        axes[1].plot(COUNTS, [pairs[n][key] if n in pairs else float("nan") for n in COUNTS], marker=("o" if key.startswith("simulation") else "^"), color=color, label=label)
    axes[0].set_ylabel("World integration steps / second")
    axes[0].set_title("Replay-pipeline throughput")
    axes[1].set_ylabel("CPU elapsed time / GPU elapsed time")
    axes[1].set_title("GPU speedup at the same batch size")
    axes[1].axhline(1, color="#555555", linewidth=.8, linestyle="--")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(COUNTS, [str(n) for n in COUNTS], rotation=40, ha="right")
        ax.set_xlabel("Environments")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8, frameon=False)
    title = "ALOHA control replay · MuJoCo 3.8.0 / MuJoCo Warp 3.8.0.3" if args.stack == "blog2" else "ALOHA control replay · MuJoCo / MuJoCo Warp 3.12.0"
    if set(pairs) != set(COUNTS):
        title += " · PARTIAL DIAGNOSTIC"
    fig.suptitle(title, fontsize=12)
    args.output.mkdir(parents=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(args.output / f"{args.stack}-reference.{suffix}", dpi=300)
    plt.close(fig)
    caption = ("Median of five complete 1001-step control-replay episodes after one complete warmup. "
               "Identical batch size, source scene and control tape; native CPU uses float64 and GPU float32. "
               "Simulation timing includes reset, integration, full state-history recording and dispatch/completion. "
               "The second speedup series also includes output collection and physical/numerical checks. "
               "Setup/JIT and artifact file writes are excluded. Coarse health checks passed for every accepted episode. "
               "This is a community MuJoCo/MuJoCo Warp reference, not Newton API timing or two-cube task validation. "
               "Pair hardware/configuration inventory with this figure before publication.")
    (args.output / "caption.txt").write_text(caption + "\n" + "Shared AMD Threadripper PRO 9985WX workstation, up to 32 native C++ rollout threads, one RTX PRO 6000 Blackwell on cuda:1. No concurrent-utilization trace was recorded; these are community measurements.\n")
    (args.output / "figure-provenance.json").write_text(json.dumps({"input_summary": str(args.summary.resolve()),
        "collector_sha256": summary["collector_sha256"], "stack": args.stack, "accepted_counts": sorted(pairs),
        "expected_resources": summary.get("expected_resources"), "partial": set(pairs) != set(COUNTS)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
