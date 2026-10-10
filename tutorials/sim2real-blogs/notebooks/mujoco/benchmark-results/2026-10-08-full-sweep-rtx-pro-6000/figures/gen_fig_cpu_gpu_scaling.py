#!/usr/bin/env python3
"""Render accepted full-task scaling measurements directly from raw reports.

Run from any directory with an optional plotting environment containing matplotlib:
    python figures/gen_fig_cpu_gpu_scaling.py

Defaults to ../{so101,rebot}/results.json.gz relative to this script. Overrides:
    --package-dir PATH --output-dir PATH --article {2,3}

The physics environment and its dependency lock are never imported or modified.
Only matplotlib is a non-stdlib dependency (rendered with matplotlib 3.10.7).
PNG is 300 dpi; PDF/SVG are vector. The adjacent data JSON contains the exact
accepted samples, min/median/max, input hashes, exclusions and recording scope.
No bootstrap, confidence intervals, interpolation, smoothing or failed-run
timings are used. Lines break at every excluded requested environment count.

Article 2 measures native MuJoCo vs MuJoCo Warp and records FULLPHYSICS after
each physics step (CPU float64 / GPU float32). Article 3 measures Newton CPU
vs Newton CUDA and records float32 payload poses/velocities at 50 Hz. Each uses
its own pinned package versions. These are separate within-article comparisons,
not measurements of cross-article speedup. Both run one complete 12-second
pick-and-place task per environment, with 6,000 physics steps, one excluded
warm-up and five measured repetitions. simulation_seconds includes reset,
control replay, stepping and recording, with CPU dispatch/completion or CUDA
synchronization; setup, host output collection and validation are separate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


COUNTS = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
ROBOTS = {"so101": "SO-101", "rebot": "reBot"}
COLORS = {"cpu": "#0072B2", "gpu": "#D55E00"}  # Okabe–Ito
STEM = "fig_cpu_gpu_scaling"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def read_report(path):
    packed = path.read_bytes()
    payload = gzip.decompress(packed) if path.suffix == ".gz" else packed
    return json.loads(payload), {
        "file": f"{path.parent.name}/{path.name}",
        "sha256": sha256(packed),
        "uncompressed_sha256": sha256(payload),
    }


def passed_episode(sample, worlds):
    validation = sample.get("validation", {})
    world_checks = validation.get("worlds", [])
    seconds = sample.get("simulation_seconds")
    return (
        sample.get("passed") is True
        and validation.get("passed") is True
        and sample.get("task_success_count") == worlds
        and sample.get("task_total_count") == worlds
        and validation.get("task_success_count") == worlds
        and validation.get("task_total_count") == worlds
        and sample.get("diagnostics", {}).get("capacity_passed") is True
        and len(world_checks) == worlds
        and all(w.get("passed") is True and w.get("finite") is True
                and w.get("time_progression") is True for w in world_checks)
        and isinstance(seconds, (int, float))
        and math.isfinite(seconds) and seconds > 0
    )


def extract_case(case, robot, series, worlds):
    config = case["config"]
    workload = case.get("workload", {})
    require(config.get("repeats") == 5 and config.get("warmups") == 1,
            f"{robot}/{series}/{worlds}: expected five repetitions and one warm-up")
    if workload:
        require(workload.get("simulated_seconds") == 12.0
                and workload.get("physics_steps") == 6000,
                f"{robot}/{series}/{worlds}: expected the full 12-second task")
    if series == "cpu":
        workers = case.get("device_info", {}).get("cpu_workers",
                  case.get("device_info", {}).get("cpu_threads"))
        if workers is not None:
            require(workers == min(32, worlds),
                    f"{robot}/{worlds}: expected min(32, environments) CPU workers")
    else:
        workers = None

    measured = case.get("samples", [])
    warmups = case.get("warmup_samples", [])
    accepted = (case.get("status") == "passed"
                and case.get("summary", {}).get("eligible") is True
                and len(measured) == 5 and len(warmups) == 1
                and all(passed_episode(s, worlds) for s in warmups + measured))
    # A nominally passing report with inconsistent evidence is an error, not a
    # silently discarded point. Failed rows retain counts, never timing values.
    if case.get("status") == "passed" or case.get("summary", {}).get("eligible"):
        require(accepted, f"{robot}/{series}/{worlds}: passing status lacks complete evidence")

    point = {
        "robot": robot, "series": series, "backend": config["backend"],
        "worlds": worlds, "case_id": case.get("case_id", config.get("case_id")),
        "status": case.get("status"), "accepted": accepted,
        "cpu_workers": workers, "warmup_count": len(warmups),
        "measured_count": len(measured), "median_seconds": None,
        "min_seconds": None, "max_seconds": None, "samples_seconds": None,
        "comparison_signature": workload.get("comparison_signature"),
        "excluded_reason": None if accepted else case.get("error", case.get("diagnostics", {}).get("error", "Incomplete or failed case")),
        "failed_world_samples": sum(
            sum(w.get("passed") is not True for w in s.get("validation", {}).get("worlds", []))
            for s in warmups + measured
        ),
    }
    if accepted:
        times = [s["simulation_seconds"] for s in measured]
        point.update(samples_seconds=times, median_seconds=statistics.median(times),
                     min_seconds=min(times), max_seconds=max(times))
        for key in ("median_seconds", "min_seconds", "max_seconds"):
            require(math.isclose(point[key], case["summary"][key], rel_tol=1e-12),
                    f"{robot}/{series}/{worlds}: raw samples disagree with {key}")
    return point


def extract_data(package, article):
    backend_names = {2: {"cpu": "mujoco", "gpu": "mjwarp"},
                     3: {"cpu": "newton_cpu", "gpu": "newton_cuda"}}[article]
    result = {"schema_version": 1, "article": article, "environment_counts": COUNTS,
              "statistic": "median", "range": "minimum and maximum of five measured repetitions; not a confidence interval",
              "metric": "simulation_seconds", "warmups_excluded": 1,
              "simulated_seconds_per_environment": 12.0,
              "reports": {}, "points": []}
    source_hashes, packages = set(), []
    for robot in ROBOTS:
        path = package / robot / "results.json.gz"
        if not path.exists():
            path = package / robot / "results.json"
        report, provenance = read_report(path)
        config = report["configuration"]
        require(report.get("finished_utc"), f"{robot}: report is not finished")
        require(sorted(config["worlds"]) == COUNTS, f"{robot}: incomplete requested count coverage")
        require(config.get("repeats") == 5 and config.get("warmups") == 1,
                f"{robot}: wrong repetition protocol")
        require(config.get("cpu_threads") == 32, f"{robot}: wrong CPU worker cap")
        source_hashes.add(report["source"]["aggregate_sha256"])
        packages.append(report["hardware"]["packages"])
        provenance.update(started_utc=report["started_utc"], finished_utc=report["finished_utc"],
                          cpu_model=report["hardware"]["cpu_model"], gpu=report["cuda"],
                          packages=report["hardware"]["packages"],
                          source_aggregate_sha256=report["source"]["aggregate_sha256"],
                          measurement_label=report.get("measurement_label"))
        result["reports"][robot] = provenance
        for worlds in COUNTS:
            pair = []
            for series, backend in backend_names.items():
                cases = [c for c in report["cases"]
                         if c.get("config", {}).get("backend") == backend
                         and c["config"].get("worlds") == worlds]
                require(len(cases) == 1, f"{robot}/{backend}/{worlds}: expected exactly one case")
                pair.append(extract_case(cases[0], robot, series, worlds))
            signatures = [p["comparison_signature"] for p in pair]
            require(signatures[0] and signatures[0] == signatures[1],
                    f"{robot}/{worlds}: CPU/GPU comparison signatures differ")
            result["points"].extend(pair)
    require(len(source_hashes) == 1, "Robot reports have different source snapshots")
    require(packages[0] == packages[1], "Robot reports have different installed package versions")
    result["source_aggregate_sha256"] = source_hashes.pop()
    result["recording_scope"] = (
        "FULLPHYSICS recorded after each of 6,000 physics steps; CPU float64 in RAM, GPU float32 in device memory."
        if article == 2 else
        "Float32 payload poses and velocities recorded at 50 Hz; 6,000 Newton SolverMuJoCo.step calls per environment."
    )
    result["timer_scope"] = "State reset, control replay, physics and recording, including CPU dispatch/completion or CUDA synchronization. Setup, warm-up, host output collection and validation are separate."
    result["comparison_scope"] = "Within this article's pinned stack only; other articles use different stacks and observation policies. No cross-article speedup is measured."
    return result


def caption(data):
    excluded = [p for p in data["points"] if not p["accepted"]]
    exclusions = "; ".join(f"{ROBOTS[p['robot']]} {p['series'].upper()} at {p['worlds']} environments"
                           for p in excluded) or "none"
    return (
        f"Blog {data['article']}: CPU/GPU scaling for the complete 12-second red-cube pick-and-place task. "
        "Each point is the median batch wall time across five accepted measured repetitions after one warm-up. "
        "Whiskers show the observed minimum and maximum, not confidence intervals. Both axes use logarithmic scaling; "
        "x positions reflect the actual environment counts. CPU uses min(32, environments) workers; GPU uses one device. "
        f"{data['recording_scope']} {data['timer_scope']} "
        f"Excluded cases (no accepted time): {exclusions}. Lines stop at excluded counts; no partial-run times are plotted. "
        "Measured 8 October 2026 on a shared AMD Ryzen Threadripper PRO 9985WX / NVIDIA RTX PRO 6000 Blackwell workstation. "
        f"{data['comparison_scope']} These community task measurements are not official product benchmarks.\n"
    )


def render(data, out):
    # Keep optional plotting cache outside the source/physics environment.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blog-scaling-matplotlib"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    except ImportError as error:
        raise SystemExit("Install optional matplotlib in a separate plotting environment; do not change the physics lock.") from error

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#88919B", "axes.labelcolor": "#26313C",
        "text.color": "#182634", "xtick.color": "#45515C", "ytick.color": "#45515C",
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "johnnys-full-task-scaling-v1",
        "savefig.dpi": 300, "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), sharey=True)
    fig.subplots_adjust(left=.075, right=.975, bottom=.315, top=.755, wspace=.17)
    article = data["article"]
    stack_title = "Native MuJoCo and MuJoCo Warp" if article == 2 else "Newton CPU and Newton CUDA"
    fig.text(.075, .96, "Full-task CPU / GPU scaling", fontsize=19, weight="bold", va="top")
    fig.text(.075, .914, f"Blog {article}  ·  {stack_title}  ·  12 simulated seconds per environment", fontsize=11.5)
    fig.text(.075, .882, "8 October 2026  ·  Shared Threadripper PRO 9985WX + RTX PRO 6000 Blackwell workstation", fontsize=9.8, color="#596573")
    handles = [Line2D([], [], color=COLORS["cpu"], marker="o", linewidth=2, markersize=6,
                      label="CPU · up to 32 workers"),
               Line2D([], [], color=COLORS["gpu"], marker="s", linestyle="--", linewidth=2,
                      markersize=6, label="GPU · one device")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.067, .858), ncol=2,
               handlelength=2.5, columnspacing=2.4, fontsize=10.5)

    accepted = [p for p in data["points"] if p["accepted"]]
    require(accepted, "No accepted points to render")
    low = min(p["min_seconds"] for p in accepted) / 1.4
    high = max(p["max_seconds"] for p in accepted) * 1.4
    for ax, (robot, name) in zip(axes, ROBOTS.items()):
        ax.set_title(name, loc="left", pad=11)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(.75, 2750)
        ax.set_ylim(low, high)
        ax.xaxis.set_major_locator(FixedLocator(COUNTS))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
        ax.xaxis.set_minor_locator(NullLocator())
        y_ticks = [base * 10 ** power for power in range(-3, 6) for base in (1, 2, 5)
                   if low <= base * 10 ** power <= high]
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.grid(which="major", axis="y", color="#E1E5E9", linewidth=.8)
        ax.tick_params(axis="x", labelsize=9.5, rotation=40, pad=4)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlabel("Environments per batch (log scale)", labelpad=9)
        for series, marker, linestyle in (("cpu", "o", "-"), ("gpu", "s", "--")):
            points = [p for p in data["points"] if p["robot"] == robot and p["series"] == series]
            ys = [p["median_seconds"] if p["accepted"] else math.nan for p in points]
            ax.plot(COUNTS, ys, color=COLORS[series], marker=marker, linestyle=linestyle,
                    linewidth=2, markersize=5.5, markeredgecolor="white", markeredgewidth=.7, zorder=4)
            passed = [p for p in points if p["accepted"]]
            if passed:
                ax.errorbar([p["worlds"] for p in passed], [p["median_seconds"] for p in passed],
                            yerr=[[p["median_seconds"] - p["min_seconds"] for p in passed],
                                  [p["max_seconds"] - p["median_seconds"] for p in passed]],
                            fmt="none", ecolor=COLORS[series], elinewidth=1.2, capsize=3,
                            capthick=1.2, zorder=3)
        excluded = [p for p in data["points"] if p["robot"] == robot and not p["accepted"]]
        if excluded:
            text = "; ".join(f"{series.upper()}: " + ", ".join(str(p["worlds"]) for p in excluded if p["series"] == series)
                             for series in ("cpu", "gpu") if any(p["series"] == series for p in excluded))
            note = f"Excluded configurations — {text} environments"
        else:
            note = "All 18 configurations accepted"
        ax.text(0, -.285, note, transform=ax.transAxes, fontsize=9.5, color="#5C4A3D")
    axes[0].set_ylabel("Median batch wall time (seconds, log scale)", labelpad=10)
    fig.text(.075, .14, "Median of 5 accepted repetitions after 1 warm-up; whiskers show min–max (not confidence intervals).", fontsize=9.5)
    fig.text(.075, .105, "Timed: reset, commands, 6,000 physics steps and state recording; setup, download and validation are separate.", fontsize=9.5)
    fig.text(.075, .07, "Failed cases have no accepted time and appear as gaps. Community measurements on a shared workstation.", fontsize=9.5, color="#596573")

    description = caption(data)
    base_meta = {"Title": f"Blog {article} complete-task CPU/GPU scaling", "Creator": "gen_fig_cpu_gpu_scaling.py"}
    fig.savefig(out / f"{STEM}.png", dpi=300, metadata={"Title": base_meta["Title"], "Description": description})
    fig.savefig(out / f"{STEM}.pdf", metadata={**base_meta, "Subject": description, "CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{STEM}.svg", metadata={**base_meta, "Description": description, "Date": None})
    svg = out / f"{STEM}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    data["renderer"] = {"matplotlib": matplotlib.__version__, "png_dpi": 300,
                        "figure_inches": [12.8, 7.2], "x_scale": "log2", "y_scale": "log10",
                        "colors": COLORS, "script_sha256": sha256(Path(__file__).read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--article", type=int, choices=(2, 3))
    args = parser.parse_args()
    article = args.article
    if article is None:
        names = {p.name for p in args.package_dir.resolve().parents}
        article = 3 if "Article_3" in names else 2 if "Article_2" in names else None
    require(article in (2, 3), "Specify --article 2 or --article 3 for a package outside the article tree")
    data = extract_data(args.package_dir, article)
    out = args.output_dir or args.package_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    render(data, out)
    (out / f"{STEM}.data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (out / f"{STEM}.caption.txt").write_text(caption(data))
    print(json.dumps({"accepted_points": sum(p["accepted"] for p in data["points"]),
                      "excluded_points": sum(not p["accepted"] for p in data["points"]),
                      "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
