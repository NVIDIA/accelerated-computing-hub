# SPDX-License-Identifier: Apache-2.0
"""The figures and animations of notebooks 90 and 95."""

import numpy as np

from .config import DEMO_BOX_MPC
from .fields import bin_edges, quantise, cic_density_device, blur_device, standardise_device
from .zeldovich import zeldovich_positions
from .ecc import ecc_cells_cupy, ecc_from_hist, euler_2d, zero_crossings
from .bench import load_stage, bytes_model

_MARKERS = "osdv^Dpx+"
_KIND_COLOUR = {"kernel": "#27a", "memcpy": "#c33", "memset": "#e90"}
_HIGHLIGHT_COLOUR = "#80c"   # the kernels `plot_timelines` is asked to single out
MIN_BAR_MS = 0.01   # a shorter GPU operation is drawn this wide, so that a memset of a few microseconds stays visible


# --- The intro ------------------------------------------------------------------------

def animate_particles(frames, n, frame_ms=500):
    """Embedded animation of `[(D, (x, y))]` point clouds in a periodic box of side n, one frame per D."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    growth, (x, y) = frames[0]
    points = ax.scatter(x, y, s=0.4, c="#123", lw=0)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title(f"D = {growth:.2f}")

    def update(i):
        growth, (x, y) = frames[i]
        points.set_offsets(np.column_stack([x, y]))
        title.set_text(f"D = {growth:.2f}")
        return points, title

    anim = FuncAnimation(fig, update, frames=len(frames), interval=frame_ms, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def plot_points_to_field(n=128, growth=1.0, r_voxels=1.5):
    """Four panels on the middle plane at D = growth: particles near the plane, mass per cell, contrast blurred over
    r_voxels, ν."""
    import cupy as cp
    import matplotlib.pyplot as plt
    pos = zeldovich_positions(n, growth, DEMO_BOX_MPC)
    z = n // 2
    rho = cic_density_device(pos, n, box=n)
    delta_s, sigma0_sq, sigma1_sq = blur_device(rho - np.float32(1.0), r_voxels)
    nu, _ = standardise_device(delta_s.copy(), sigma0_sq, sigma1_sq)   # the copy keeps the contrast for panel 3

    def plane(a):
        return cp.asnumpy(a[:, :, z])
    near = np.abs(np.mod(pos[:, 2] - z + n / 2, n) - n / 2) < 0.5
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))
    axes[0].scatter(np.mod(pos[near, 0], n), np.mod(pos[near, 1], n), s=0.3, c="#123", lw=0)
    axes[0].set_xlim(0, n)
    axes[0].set_ylim(0, n)
    axes[0].set_aspect("equal")
    axes[0].set_title("1. particles within half a cell of the plane")
    panels = [(plane(rho), "2. mass per cell", 0, 4),
              (plane(delta_s), f"3. contrast, blurred over R = {r_voxels:g} cells", -1, 3),
              (plane(nu), "4. ν = contrast / standard deviation", -3, 3)]
    for ax, (plane, label, lo, hi) in zip(axes[1:], panels):
        ax.imshow(plane.T, origin="lower", cmap="magma", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(label)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_slices(nu, thresholds):
    """The middle slice of ν beside its superlevel masks at the given levels."""
    import matplotlib.pyplot as plt
    nu = np.asarray(nu)
    s = nu[:, :, nu.shape[2] // 2]
    fig, axes = plt.subplots(1, 1 + len(thresholds), figsize=(3.6 * (1 + len(thresholds)), 3.8))
    axes[0].imshow(s, cmap="magma", vmin=-3, vmax=3, interpolation="nearest")
    axes[0].set_title("ν on one slice")
    for ax, t in zip(axes[1:], thresholds):
        ax.imshow(s >= t, cmap="Greys_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"ν ≥ {t:g}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def animate_euler_buildup(mask, frame_ms=250):
    """Vertices, then edges, then faces are drawn on a pixel mask while V, E, F and V − E + F count up."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle
    from IPython.display import HTML
    m = np.asarray(mask, dtype=bool)
    rows, cols = m.shape
    verts = [(r, c) for r in range(rows) for c in range(cols) if m[r, c]]
    edges = ([((r, c), (r + 1, c)) for r in range(rows - 1) for c in range(cols) if m[r, c] and m[r + 1, c]]
             + [((r, c), (r, c + 1)) for r in range(rows) for c in range(cols - 1) if m[r, c] and m[r, c + 1]])
    faces = [(r, c) for r in range(rows - 1) for c in range(cols - 1)
             if m[r, c] and m[r + 1, c] and m[r, c + 1] and m[r + 1, c + 1]]
    steps = [("V", v) for v in verts] + [("E", e) for e in edges] + [("F", f) for f in faces]
    fig, ax = plt.subplots(figsize=(5, 5.4))
    ax.imshow(m, cmap="Greys", vmin=0, vmax=3, interpolation="nearest")
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("V = 0   E = 0   F = 0      χ = V − E + F = 0")
    # FuncAnimation draws frame 0 for the initial draw and again as the first frame, so every cell is counted once.
    drawn = set()
    counts = {"V": 0, "E": 0, "F": 0}

    def update(i):
        if i in drawn:
            return ()
        drawn.add(i)
        kind, item = steps[i]
        counts[kind] += 1
        if kind == "V":
            ax.plot(item[1], item[0], "o", color="#27a", ms=9, zorder=3)
        elif kind == "E":
            (r0, c0), (r1, c1) = item
            ax.plot([c0, c1], [r0, r1], "-", color="#27a", lw=3, zorder=2)
        else:
            r, c = item
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#5a8", alpha=0.6, zorder=1))
        v, e, f = counts["V"], counts["E"], counts["F"]
        title.set_text(f"V = {v}   E = {e}   F = {f}      χ = V − E + F = {v - e + f}")
        return (title,)

    anim = FuncAnimation(fig, update, frames=len(steps), interval=frame_ms, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def animate_flood(slice2d, start=2.5, end=-2.5, interval=0.125, frame_ms=150):
    """The mask {ν ≥ level} of a slice as the level falls from start to end by interval, with V − E + F traced."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    levels = np.linspace(start, end, int(round(abs(start - end) / interval)) + 1)
    s = np.asarray(slice2d)
    chis = np.array([euler_2d(s >= level)[3] for level in levels])
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(10.5, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})
    img = ax.imshow(np.zeros(s.shape, dtype=bool), cmap="Greys_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("")
    line, = axc.plot([], [], "o-", color="#27a", ms=3)
    axc.axhline(0, color="#888", lw=0.8)
    pad = 0.08 * (chis.max() - chis.min() or 1)
    axc.set(xlabel="level ν", ylabel="χ = V − E + F", title="χ at every level so far",
            xlim=(min(start, end), max(start, end)), ylim=(chis.min() - pad, chis.max() + pad))
    axc.grid(alpha=0.3)

    def update(i):
        img.set_data(s >= levels[i])
        line.set_data(levels[:i + 1], chis[:i + 1])
        title.set_text(f"ν ≥ {levels[i]:5.2f}      χ = V − E + F = {chis[i]:4d}")
        return img, line, title

    anim = FuncAnimation(fig, update, frames=len(levels), interval=frame_ms, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def isosurfaces(nu, thresholds, sub=64):
    """Marching-cubes isosurfaces of a sub-cube at each level, through scikit-image."""
    from skimage.measure import marching_cubes
    import matplotlib.pyplot as plt
    vol = np.ascontiguousarray(np.asarray(nu)[:sub, :sub, :sub], dtype=np.float32)
    fig = plt.figure(figsize=(4.4 * len(thresholds), 4.4))
    for i, t in enumerate(thresholds, 1):
        ax = fig.add_subplot(1, len(thresholds), i, projection="3d")
        verts, faces, _, _ = marching_cubes(vol, level=float(t))
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color="#5a8", lw=0.05, alpha=0.95)
        ax.set_xlim(0, sub)
        ax.set_ylim(0, sub)
        ax.set_zlim(0, sub)
        ax.set_title(f"ν ≥ {t:g}")
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_birth_example(q2d, k):
    """A small 2D patch as the level falls: the cells entering at each level, the ±1 tokens per bin, χ summed from the
    top.

    Returns (tokens per bin, χ)."""
    import matplotlib.pyplot as plt
    q = np.asarray(q2d)
    rows, cols = q.shape
    e_down = np.minimum(q[:-1, :], q[1:, :])
    e_right = np.minimum(q[:, :-1], q[:, 1:])
    faces = np.minimum(np.minimum(q[:-1, :-1], q[1:, :-1]), np.minimum(q[:-1, 1:], q[1:, 1:]))

    def bc(a):
        return np.bincount(a.ravel(), minlength=k)

    born_v, born_e, born_f = bc(q), bc(e_down) + bc(e_right), bc(faces)
    hist = born_v - born_e + born_f
    chi = ecc_from_hist(hist)
    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, k)
    for i, level in enumerate(range(k - 1, -1, -1)):
        ax = fig.add_subplot(grid[0, i])

        def shade(birth, colour):
            return colour if birth == level else "#bbb"

        for (r, c), birth in np.ndenumerate(faces):
            if birth >= level:
                ax.fill([c, c + 1, c + 1, c], [r, r, r + 1, r + 1], color=shade(birth, "#5a8"), alpha=0.35, lw=0)
        for edges, (dr, dc) in ((e_down, (1, 0)), (e_right, (0, 1))):
            for (r, c), birth in np.ndenumerate(edges):
                if birth >= level:
                    ax.plot([c, c + dc], [r, r + dr], color=shade(birth, "#c33"), lw=2.5, zorder=2)
        for (r, c), value in np.ndenumerate(q):
            in_set = value >= level
            ax.plot(c, r, "o", mfc=shade(value, "#27a") if in_set else "none", mec="#bbb", ms=19, zorder=3)
            ax.text(c, r, value, ha="center", va="center", color="white" if in_set else "#999", weight="bold", zorder=4)
        ax.set_xlim(-0.6, cols - 0.4)
        ax.set_ylim(rows - 0.4, -0.6)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"level {level} adds {born_v[level]}V - {born_e[level]}E + {born_f[level]}F = {hist[level]:+d}",
                     fontsize=10)
    axh, axc = fig.add_subplot(grid[1, :k // 2]), fig.add_subplot(grid[1, k // 2:])
    bins = np.arange(k)
    axh.bar(bins, born_v, color="#27a", label="+1 per vertex")
    axh.bar(bins, born_f, bottom=born_v, color="#5a8", label="+1 per face")
    axh.bar(bins, -born_e, color="#c33", label="−1 per edge")
    axh.plot(bins, hist, "k_", ms=22, mew=2.5, label="net per bin")
    axh.axhline(0, color="#888", lw=0.8)
    axh.set_xticks(bins)
    axh.set_xlabel("birth bin")
    axh.set_ylabel("tokens")
    axh.set_title("tokens sorted by birth bin")
    axh.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)
    axc.hlines(chi, bins - 0.5, bins + 0.5, color="#27a", lw=2.5)
    axc.plot(bins, chi, "o", color="#27a")
    for b in bins:
        axc.annotate(f"{chi[b]}", (b, chi[b]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    axc.axhline(0, color="#888", lw=0.8)
    axc.margins(y=0.15)
    axc.set_xticks(bins)
    axc.set_xlabel("level (bin)")
    axc.set_ylabel("χ")
    axc.set_title("χ at a level = the net tokens at or above it")
    plt.tight_layout()
    plt.show()
    return hist, chi


def animate_birthdays(nu_small, k=16, frame_ms=700):
    """On a small cube, lower the level bin by bin: the cells born at each level, their ±1 tokens, the curve."""
    import cupy as cp
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    q_dev = quantise(cp.asarray(nu_small), k)
    v, (ex, ey, ez), (fxy, fxz, fyz), c = ecc_cells_cupy(q_dev)

    def bc(a):
        return cp.asnumpy(cp.bincount(a.ravel(), minlength=k))

    born = {"V": bc(v), "E": bc(ex) + bc(ey) + bc(ez), "F": bc(fxy) + bc(fxz) + bc(fyz), "C": bc(c)}
    q = cp.asnumpy(q_dev)
    hist = born["V"] - born["E"] + born["F"] - born["C"]
    chi = ecc_from_hist(hist)
    edges = bin_edges(k)
    fig = plt.figure(figsize=(13, 4.2))
    ax3 = fig.add_subplot(1, 3, 1, projection="3d")
    axh = fig.add_subplot(1, 3, 2)
    axc = fig.add_subplot(1, 3, 3)

    def update(i):
        b = k - 1 - i
        ax3.clear()
        ax3.voxels(q >= b, facecolors="#d9b36b", edgecolor="#555", linewidth=0.3)
        ax3.set_title(f"the set ν ≥ {edges[b]:.2f} (bin {b} and above)")
        ax3.set_axis_off()
        axh.clear()
        shown = np.arange(k) >= b
        axh.bar(edges[shown], hist[shown], width=(edges[1] - edges[0]) * 0.9,
                color=["#c33" if j == b else "#27a" for j in np.arange(k)[shown]])
        axh.axhline(0, color="#888", lw=0.8)
        axh.set_xlim(edges[0], edges[-1] + edges[1] - edges[0])
        axh.set_ylim(hist.min() * 1.15 - 1, hist.max() * 1.15 + 1)
        axh.set_title(f"{born['V'][b]}V - {born['E'][b]}E "
                      f"+{born['F'][b]}F - {born['C'][b]}C = {hist[b]:+d}")
        axh.set_xlabel("birth bin (ν)")
        axc.clear()
        axc.step(edges[shown], chi[shown], where="post", color="#27a")
        axc.axhline(0, color="#888", lw=0.8)
        axc.set_xlim(edges[0], edges[-1] + edges[1] - edges[0])
        axc.set_ylim(chi.min() * 1.15 - 1, chi.max() * 1.15 + 1)
        axc.set_title("χ(ν) = tokens summed from the top")
        axc.set_xlabel("level ν")
        return ()

    anim = FuncAnimation(fig, update, frames=k, interval=frame_ms, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def plot_hist_curve(edges, hist, chi):
    """The token histogram by birth bin beside χ(ν), its cumulative sum from the top."""
    import matplotlib.pyplot as plt
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6))
    left.bar(edges, hist, width=edges[1] - edges[0], color="#27a")
    left.axhline(0, color="#888", lw=0.8)
    left.set(xlabel="birth bin (ν)", ylabel="tokens", title="tokens sorted by birth")
    right.step(edges, chi, where="post", color="#27a")
    right.axhline(0, color="#888", lw=0.8)
    right.set(xlabel="level ν", ylabel="χ", title="χ(ν): the tokens summed from the top")
    plt.tight_layout()
    plt.show()


def plot_overlay(edges, chi_meas, chi_pred, title="", label="measured"):
    """Measured χ against the Gaussian expectation, with their difference underneath.

    Returns (relative difference, zero crossings)."""
    import matplotlib.pyplot as plt
    chi_meas, chi_pred = np.asarray(chi_meas, dtype=np.float64), np.asarray(chi_pred, dtype=np.float64)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    top.plot(edges, chi_meas, "o-", ms=3, label=label)
    top.plot(edges, chi_pred, "--", color="#444", label="Gaussian expectation")
    top.axhline(0, color="#888", lw=0.8)
    top.set(ylabel="χ", title=title)
    top.legend(fontsize=8)
    top.grid(alpha=0.3)
    bottom.plot(edges, chi_meas - chi_pred, color="#c33")
    bottom.axhline(0, color="#888", lw=0.8)
    bottom.set(xlabel="level ν", ylabel="difference")
    bottom.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    rel = float(np.max(np.abs(chi_meas - chi_pred)) / np.max(np.abs(chi_pred)))
    return rel, zero_crossings(edges, chi_meas)


# --- The synthesis ----------------------------------------------------------------------

def plot_curves_with_delta(edges, curves, reference, title=""):
    """Every tool's χ(ν) over the reference, with each tool's difference to it in a strip underneath."""
    import matplotlib.pyplot as plt
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    top.plot(edges, reference, "--", color="#444", lw=1.5, label="CuPy reference")
    for (name, chi), marker in zip(curves.items(), _MARKERS):
        top.plot(edges, chi, marker, ms=4, alpha=0.8, label=name)
        bottom.plot(edges, np.asarray(chi, dtype=np.int64) - np.asarray(reference, dtype=np.int64),
                    marker + "-", ms=3, label=name)
    top.axhline(0, color="#888", lw=0.8)
    top.set(ylabel="χ", title=title)
    top.legend(fontsize=8)
    top.grid(alpha=0.3)
    bottom.axhline(0, color="#888", lw=0.8)
    bottom.set(xlabel="level ν", ylabel="Δχ", ylim=(-1.5, 1.5))
    bottom.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_timelines(traces, title="", highlight=None):
    """One row per tool: GPU operations as bars on a shared time axis, kernels, memcpys and memsets coloured, idle time
    grey.

    `traces` maps a label to a list of GPU operations or to a `trace_call` record. With `highlight`, the kernels whose
    name contains it are drawn in their own colour and named in the legend."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    names = list(traces)
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(names) + 1.4))
    span = 0.0
    for row, name in enumerate(names):
        rows = traces[name]
        if isinstance(rows, dict):
            rows = rows["rows"]
        end = max((s + d) for s, d, _, _ in rows) * 1e3 if rows else 0.0
        # the GPU is idle wherever no operation covers the grey bar
        ax.broken_barh([(0.0, end)], (row - 0.35, 0.7), color="#bbb", lw=0)
        for kind in _KIND_COLOUR:
            bars = [(s * 1e3, max(d * 1e3, MIN_BAR_MS)) for s, d, _, k in rows if k == kind]
            if bars:
                ax.broken_barh(bars, (row - 0.35, 0.7), color=_KIND_COLOUR[kind], lw=0)
        if highlight:
            bars = [(s * 1e3, max(d * 1e3, MIN_BAR_MS)) for s, d, op, k in rows if k == "kernel" and highlight in op]
            if bars:
                ax.broken_barh(bars, (row - 0.35, 0.7), color=_HIGHLIGHT_COLOUR, lw=0)
        span = max(span, end)
        busy = sum(d for _, d, _, _ in rows) * 1e3
        kernels = sum(1 for *_, kind in rows if kind == "kernel")
        ax.text(end, row + 0.42, f"  kernels {kernels}, busy {busy:.2f} ms", fontsize=8, va="bottom", color="#444")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, len(names) - 0.2)
    ax.set_xlim(0, span * 1.35)
    ax.invert_yaxis()
    ax.set(xlabel="time since the first GPU operation [ms]", title=title)
    handles = [Patch(color=c, label=k) for k, c in _KIND_COLOUR.items()]
    if highlight:
        handles.append(Patch(color=_HIGHLIGHT_COLOUR, label=f"{highlight} kernels"))
    handles.append(Patch(color="#bbb", label="GPU idle"))
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()


def plot_breakdown(parts, title=""):
    """Stacked horizontal bars per tool: kernel time, copies and memsets, and what is left of the wall time."""
    import matplotlib.pyplot as plt
    names = list(parts)
    keys = [("kernel_s", "kernels", "#27a"), ("copies_s", "copies and memsets", "#c33"),
            ("other_s", "launch gaps, CPU, profiler", "#bbb")]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 1.4))
    left = np.zeros(len(names))
    for key, label, colour in keys:
        vals = np.array([parts[n].get(key, 0.0) * 1e3 for n in names])
        ax.barh(names, vals, left=left, color=colour, label=label)
        left += vals
    ax.invert_yaxis()
    ax.set(xlabel="one call [ms]", title=title)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()


def plot_traffic(sweeps, stages, gbs, title=""):
    """Gvoxel/s against N for every label of `sweeps`, with the memory-traffic estimate of its stage dashed at `gbs`
    GB/s.

    `stages` maps a label to its stage name. Each dashed line is named by the labels it belongs to."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    colours = ("#27a", "#c33", "#5a8", "#e90", "#80c", "#444", "#a63", "#1a9", "#b50")
    estimates = {}
    for (name, points), colour in zip(sweeps.items(), colours):
        ns = [p["n"] for p in points]
        ax.plot(ns, [n ** 3 / p["median_s"] / 1e9 for n, p in zip(ns, points)], "o-", color=colour, label=name)
        if name in stages:
            # Gvoxel/s at the copy bandwidth; the bytes per voxel do not depend on N
            rate = gbs * ns[-1] ** 3 / bytes_model(stages[name], ns[-1])
            estimates.setdefault(round(rate, 3), []).append(name)
    ticks = sorted({p["n"] for points in sweeps.values() for p in points})
    for rate, names in estimates.items():
        ax.plot([ticks[0], ticks[-1]], [rate, rate], "--", color="#888", lw=1)
        ax.text(ticks[-1] * 1.06, rate, ", ".join(names), fontsize=7, va="center", color="#555")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{n}^3" for n in ticks])
    ax.minorticks_off()
    ax.set_xlim(ticks[0] / 1.2, ticks[-1] * 3.4)
    ax.set(xlabel="N (voxels per side)", ylabel="Gvoxel/s [log]", title=title)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()


def plot_first_call(labels, first_s, median_s, title):
    """Grouped bars of the first call and the median on a log time axis."""
    import matplotlib.pyplot as plt
    x, w = np.arange(len(labels)), 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels)), 4))
    ax.bar(x - w / 2, [t * 1e3 for t in first_s], w, label="first call", color="#c33")
    ax.bar(x + w / 2, [t * 1e3 for t in median_s], w, label="median", color="#5a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set(ylabel="time [ms, log]", title=title)
    ax.legend()
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_compile_share(stage, title, compile_label):
    """Stacked bars splitting the first call at every size of the stage's sweep into the run and the rest of the first
    call."""
    import matplotlib.pyplot as plt
    sweep = load_stage(stage)["sweep"]
    labels = [f"N={row['n']}^3" for row in sweep]
    first_ms = np.array([row["first_s"] for row in sweep]) * 1e3
    run_ms = np.array([row["median_s"] for row in sweep]) * 1e3
    compile_ms = np.maximum(first_ms - run_ms, 0.0)
    run_share = run_ms / (run_ms + compile_ms)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, run_share, width=0.55, color="#27a", label="run (time per call)")
    ax.bar(x, 1.0 - run_share, width=0.55, bottom=run_share, color="#c33", label=compile_label)
    for xi, (share, ms) in enumerate(zip(run_share, run_ms)):
        ax.text(xi, 1.02, f"{(1.0 - share) * 100:.0f}% extra\n{ms:,.1f} ms run", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, .25, .5, .75, 1.])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set(ylabel="share of the first-call time", title=title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()
