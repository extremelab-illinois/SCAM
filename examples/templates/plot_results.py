# SPDX-License-Identifier: MIT
"""Standalone plotter for SCAM CSV output.

Usage
-----
    python3 examples/templates/plot_results.py <results_dir> [options]

    # Static PNG only (default)
    python3 examples/templates/plot_results.py results/tacot_arcjet_template/

    # PNG + video (mp4 if ffmpeg is available, otherwise gif)
    python3 examples/templates/plot_results.py results/tacot_arcjet_template/ --video

    # Control video frame rate and speed
    python3 examples/templates/plot_results.py results/ablation1_template/ --video --fps 20 --speed 4
    
    # Slab animation
    python3 examples/templates/plot_results.py results/ablation2_template/ --slab --speed 30 2>&1

Options
-------
--video         Produce an animation alongside the static PNG.
--fps N         Frames per second in the output video (default: 15).
--speed N       How many simulation seconds each real second covers (default: auto).
                Overrides --fps to honour the requested playback speed.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_time_history(out: Path):
    csv = out / "time_history.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No time_history.csv in {out}")
    with open(csv) as fh:
        header = fh.readline().strip().split(",")
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    def col(name):
        return data[:, header.index(name)]

    tc_cols = [h for h in header if h.startswith("tc_") and h.endswith("_K")]
    return header, data, col, tc_cols


def _load_profiles(out: Path, filename: str):
    csv = out / filename
    if not csv.exists():
        return None, None, None
    with open(csv) as fh:
        raw_header = fh.readline().strip().split(",")
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    depth_m = data[:, 0]       # [n_nodes]
    profiles = data[:, 1:]     # [n_nodes, n_snapshots]
    snap_times = []
    for h in raw_header[1:]:
        try:
            snap_times.append(float(h.replace("t=", "").replace("s", "")))
        except ValueError:
            snap_times.append(np.nan)
    return depth_m, profiles, np.array(snap_times)


# ---------------------------------------------------------------------------
# Static figure
# ---------------------------------------------------------------------------

def make_static_figure(out: Path) -> Path:
    header, data, col, tc_cols = _load_time_history(out)

    t       = col("time_s")
    T_wall  = col("T_wall_K")
    s_total = col("s_total_m")
    q_cond  = col("q_cond_W_m2")

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(out.name, fontsize=13)

    # Panel 1: temperatures
    ax = axes[0]
    ax.plot(t, T_wall, "k-", lw=1.5, label="T_wall")
    for tc_name in tc_cols:
        depth = tc_name.split("_at_")[1].replace("mm_K", "") + " mm"
        ax.plot(t, col(tc_name), "--", lw=1, label=f"TC @ {depth}")
    ax.set_ylabel("Temperature [K]")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)

    # Panel 2: recession
    ax = axes[1]
    recession_mm = (s_total - s_total[0]) * 1e3
    ax.plot(t, recession_mm, "b-", lw=1.5)
    ax.set_ylabel("Recession [mm]")
    ax.grid(True, lw=0.4)

    # Panel 3: conduction heat flux
    ax = axes[2]
    ax.plot(t, q_cond * 1e-3, "r-", lw=1.5, label="q_cond")
    ax.set_ylabel("q_cond [kW/m²]")
    ax.set_xlabel("Time [s]")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)

    plt.tight_layout()
    png = out / "results.png"
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def make_video(out: Path, fps: int, speed: float | None, beta_level: float = 0.01) -> Path:
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    header, th_data, col, tc_cols = _load_time_history(out)
    t_hist   = col("time_s")
    T_wall   = col("T_wall_K")
    s_total  = col("s_total_m")

    depth_m, T_profiles, snap_times = _load_profiles(out, "temperature_profiles.csv")
    if depth_m is None:
        raise FileNotFoundError(f"No temperature_profiles.csv in {out}")
    _, beta_profiles, _ = _load_profiles(out, "char_fraction_profiles.csv")
    # Per-snapshot actual node positions — correct depth axis for ALE cases.
    # Falls back to the fixed first-snapshot grid if the file is absent.
    _, depth_profiles, _ = _load_profiles(out, "depth_profiles.csv")

    n_snaps = T_profiles.shape[1]

    # Determine fps from speed if requested
    if speed is not None:
        t_sim = snap_times[-1] - snap_times[0]
        duration_s = t_sim / speed          # real-time video duration [s]
        fps = max(1, round(n_snaps / duration_s))

    # Depth axis: distance from the hot face [mm]
    # y_nodes are from original front face; depth = y - y[0]
    depth_mm = (depth_m - depth_m[0]) * 1e3

    # Temperature axis limits from all valid data
    T_all = T_profiles[np.isfinite(T_profiles)]
    T_min, T_max = T_all.min(), T_all.max()
    T_pad = (T_max - T_min) * 0.05
    s_recession_mm = (s_total - s_total[0]) * 1e3
    s_max_mm = max(s_recession_mm.max() * 1.2, 0.1)

    # --- Figure layout: two panels ---
    fig, (ax_prof, ax_hist) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.0, 1.6]},
    )

    # Left panel — temperature profile
    ax_prof.set_xlim(depth_mm[0], depth_mm[np.isfinite(depth_mm)].max())
    ax_prof.set_ylim(T_min - T_pad, T_max + T_pad)
    ax_prof.set_xlabel("Depth from hot surface [mm]", fontsize=16)
    ax_prof.set_ylabel("Temperature [K]", fontsize=16)
    ax_prof.tick_params(which="both", direction="out", labelsize=15,
                        top=True, labeltop=False,
                        right=True, labelright=False)
    (line_prof,) = ax_prof.plot([], [], "k-", lw=2)
    time_label   = ax_prof.text(
        0.97, 0.97, "", transform=ax_prof.transAxes,
        ha="right", va="top", fontsize=17, fontweight="bold",
    )

    # Recession marker on profile (vertical line at current surface position)
    vline_surf = ax_prof.axvline(x=0, color="r", lw=1.5, ls="--", label="surface")
    # β isoline (leading edge of pyrolysis wave); only drawn when data is available
    vline_beta = ax_prof.axvline(
        x=np.nan, color="b", lw=1.5, ls="--",
        label=f"β = {beta_level}" if beta_profiles is not None else "_nolegend_",
    )
    # Legend just below the time counter (top-right)
    ax_prof.legend(fontsize=14, loc="upper right",
                   bbox_to_anchor=(1.0, 0.88), framealpha=0.8)

    # Right panel — time history with cursor
    t_pad_x = t_hist[-1] * 0.02
    ax_hist.set_xlim(-t_pad_x, t_hist[-1])
    ax_hist.set_ylim(T_min - T_pad, T_max + T_pad)
    ax_hist.plot(t_hist, T_wall, "k-", lw=1.2, label="T_wall")
    for tc_name in tc_cols:
        depth_lbl = tc_name.split("_at_")[1].replace("mm_K", "") + " mm"
        ax_hist.plot(t_hist, col(tc_name), "--", lw=0.9, label=f"TC @ {depth_lbl}")
    ax_hist.set_xlabel("Time [s]", fontsize=16)
    ax_hist.set_ylabel("Temperature [K]", fontsize=16)
    ax_hist.tick_params(which="both", direction="out", labelsize=15,
                        top=True, labeltop=False)
    ax_hist.legend(fontsize=14, loc="upper right")
    cursor = ax_hist.axvline(x=t_hist[0], color="r", lw=1.2, ls="--")

    # Inset recession strip inside right panel (twin x-axis at bottom)
    ax_rec = ax_hist.twinx()
    ax_rec.plot(t_hist, s_recession_mm, "b-", lw=1, alpha=0.6)
    ax_rec.set_ylabel("Recession [mm]", color="b", fontsize=15)
    ax_rec.tick_params(axis="y", labelcolor="b", labelsize=14, direction="out")
    ax_rec.set_ylim(0, s_max_mm)
    rec_dot, = ax_rec.plot([], [], "bo", ms=5)

    plt.tight_layout()

    def _frame_index_for_snap(i):
        """Find the index in t_hist closest to snap_times[i]."""
        return int(np.argmin(np.abs(t_hist - snap_times[i])))

    def _frame_depth_mm(i: int) -> np.ndarray:
        """Actual node depths [mm] at frame i; falls back to fixed grid."""
        if depth_profiles is not None:
            return (depth_profiles[:, i] - depth_m[0]) * 1e3
        return depth_mm

    def _beta_isoline_depth_mm(i: int) -> float:
        """Return depth [mm] where β crosses beta_level, or nan if unavailable.

        If the surface node already exceeds beta_level (e.g. requesting β=1
        when the ALE surface node is at β=0.97), the isoline has been consumed
        by recession — we return the surface depth so the line stays coincident
        with the surface rather than disappearing.
        """
        if beta_profiles is None:
            return np.nan
        beta = beta_profiles[:, i]
        d = _frame_depth_mm(i)
        mask = np.isfinite(beta) & np.isfinite(d)
        if mask.sum() < 2:
            return np.nan
        d = d[mask]
        b = beta[mask]
        # Isoline consumed by recession: requested level exceeds the surface
        # node's β (e.g. --beta 1.0 when ALE interpolation gives β_surface=0.97).
        # Extrapolate linearly from the first two nodes; if the crossing would
        # fall before the surface, clamp it to the surface position.
        if beta_level > b[0]:
            db01 = b[1] - b[0]
            if db01 != 0:
                d_extrap = d[0] + (beta_level - b[0]) / db01 * (d[1] - d[0])
            else:
                d_extrap = d[0]
            return float(max(d_extrap, d[0]))
        crossings = np.where(np.diff(np.sign(b - beta_level)))[0]
        if len(crossings) == 0:
            return np.nan
        j = crossings[0]
        db = b[j + 1] - b[j]
        if db == 0:
            return float(d[j])
        return float(d[j] + (beta_level - b[j]) / db * (d[j + 1] - d[j]))

    def init():
        line_prof.set_data([], [])
        time_label.set_text("")
        cursor.set_xdata([t_hist[0]])
        rec_dot.set_data([], [])
        vline_surf.set_xdata([0])
        vline_beta.set_xdata([np.nan])
        return line_prof, time_label, cursor, rec_dot, vline_surf, vline_beta

    def update(i):
        T = T_profiles[:, i]
        d = _frame_depth_mm(i)
        mask = np.isfinite(T) & np.isfinite(d)
        line_prof.set_data(d[mask], T[mask])

        t_snap = snap_times[i]
        time_label.set_text(f"t = {t_snap:.1f} s")
        cursor.set_xdata([t_snap])

        hi = _frame_index_for_snap(i)
        rec_dot.set_data([t_hist[hi]], [s_recession_mm[hi]])

        surf_depth_mm = (s_total[hi] - s_total[0]) * 1e3
        vline_surf.set_xdata([surf_depth_mm])
        vline_beta.set_xdata([_beta_isoline_depth_mm(i)])

        return line_prof, time_label, cursor, rec_dot, vline_surf, vline_beta

    anim = FuncAnimation(
        fig, update, frames=n_snaps, init_func=init,
        blit=True, interval=1000 / fps,
    )

    # Try mp4 first, fall back to gif
    mp4 = out / "results.mp4"
    gif = out / "results.gif"
    try:
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(str(mp4), writer=writer)
        print(f"Saved: {mp4}")
        plt.close(fig)
        return mp4
    except Exception:
        pass

    try:
        writer = PillowWriter(fps=fps)
        anim.save(str(gif), writer=writer)
        print(f"Saved: {gif}  (ffmpeg not available; saved as GIF)")
        plt.close(fig)
        return gif
    except Exception as exc:
        raise RuntimeError(
            "Could not save video. Install ffmpeg for mp4 or pillow for gif."
        ) from exc


# ---------------------------------------------------------------------------
# Slab animation
# ---------------------------------------------------------------------------

def make_slab_video(out: Path, fps: int, speed: float | None) -> Path:
    """2-D slab visualization: T, β_total, β per component as colored contours.

    The 1-D profiles are extruded to a thin slab (width = total_depth / 4).
    Depth axis is in the original material frame: y=0 = original hot face,
    increasing downward.  The ablated zone (y < s_total) is shown in light gray.
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    import matplotlib.colors as mcolors
    import matplotlib.gridspec as gridspec

    header, th_data, col, tc_cols = _load_time_history(out)
    t_hist      = col("time_s")
    T_wall_hist = col("T_wall_K")
    s_total     = col("s_total_m")

    depth_m, T_profiles, snap_times = _load_profiles(out, "temperature_profiles.csv")
    if depth_m is None:
        raise FileNotFoundError(f"No temperature_profiles.csv in {out}")
    _, beta_profiles, _  = _load_profiles(out, "char_fraction_profiles.csv")
    _, depth_profiles, _ = _load_profiles(out, "depth_profiles.csv")

    # Per-component β files: comp0_beta_profiles.csv, comp1_beta_profiles.csv, …
    comp_betas, comp_names = [], []
    for c_idx in range(10):
        _, cb, _ = _load_profiles(out, f"comp{c_idx}_beta_profiles.csv")
        if cb is None:
            break
        comp_betas.append(cb)
        comp_names.append(f"comp {c_idx}")

    n_snaps = T_profiles.shape[1]
    if speed is not None:
        t_sim      = snap_times[-1] - snap_times[0]
        fps        = max(1, round(n_snaps / (t_sim / speed)))

    # Fixed depth grid in the original material frame [mm]
    back_face_mm  = depth_m[-1] * 1e3          # fixed back-face position [mm]
    n_display     = 400                          # rows in the rendered image
    y_fixed_mm    = np.linspace(0.0, back_face_mm, n_display)
    width_mm      = back_face_mm / 4.0          # slab width = depth/4 → slender slab

    # Color limits
    T_all        = T_profiles[np.isfinite(T_profiles)]
    T_min, T_max = T_all.min(), T_all.max()

    # β_total colormap: tan = virgin → dark-brown = char
    cmap_beta = mcolors.LinearSegmentedColormap.from_list(
        "virgin_char",
        [(0.95, 0.90, 0.72), (0.25, 0.14, 0.05)],
        N=256,
    )
    comp_cmaps = ["Blues", "Oranges", "Greens", "Purples"]
    ablated_color = (0.82, 0.82, 0.82)          # light gray for consumed zone

    # Build panel list: (title, data_array, cmap, vmin, vmax, bar_label)
    panels = [("Temperature [K]",   T_profiles,    "plasma",    T_min, T_max, "T [K]")]
    if beta_profiles is not None:
        panels.append(("β  total",  beta_profiles, cmap_beta,   0.0,   1.0,   "β"))
    for c_idx, cb in enumerate(comp_betas):
        cmap_c = comp_cmaps[c_idx % len(comp_cmaps)]
        panels.append((f"β  {comp_names[c_idx]}", cb, cmap_c, 0.0, 1.0, f"β_{c_idx}"))

    n_panels = len(panels)

    # Colorbars go BELOW each slab (horizontal) so nothing crowds the inter-panel
    # gaps.  This lets wspace be very tight (slabs close) without any overlap.
    # bottom=0.25 gives room for: x-label → gap → colorbar → colorbar label.
    # top=0.84 keeps the time-counter header band clear.
    fig = plt.figure(figsize=(3.2 * n_panels + 6.0, 12.0))
    gs  = gridspec.GridSpec(
        1, n_panels + 2,
        width_ratios=[1.0] * n_panels + [0.4, 4.0],
        wspace=0.08, left=0.07, right=0.94, bottom=0.20, top=0.90,
    )
    slab_axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    ax_hist   = fig.add_subplot(gs[0, n_panels + 1])   # skip spacer col n_panels

    # --- Set up slab panels ---
    images = []
    for i, (title, data_arr, cmap, vmin, vmax, bar_label) in enumerate(panels):
        ax  = slab_axes[i]
        ax.set_title(title, fontsize=18, pad=6)
        ax.set_xlim(0, width_mm)
        ax.set_ylim(back_face_mm, 0)            # surface (y=0) at TOP
        ax.set_xlabel("Width [mm]", fontsize=16)
        if i == 0:
            ax.set_ylabel("Depth from hot surface [mm]", fontsize=17)
        # Outer ticks on all four sides; hide labels on right/top/non-first left
        ax.tick_params(which="both", direction="out", labelsize=16,
                       top=True, labeltop=False,
                       right=True, labelright=False,
                       labelleft=(i == 0))

        cm = (plt.get_cmap(cmap).copy() if isinstance(cmap, str) else
              mcolors.LinearSegmentedColormap(cmap.name, cmap._segmentdata))
        cm.set_bad(color=ablated_color)

        blank = np.full((n_display, 2), np.nan)
        im = ax.imshow(
            blank,
            extent=[0, width_mm, back_face_mm, 0],
            origin="upper",
            aspect="auto",
            cmap=cm,
            vmin=vmin, vmax=vmax,
            interpolation="bilinear",
        )
        images.append(im)
        # Horizontal colorbar below the panel — eliminates right-side crowding
        cb = plt.colorbar(im, ax=ax, orientation="horizontal", location="bottom",
                          pad=0.14, fraction=0.06, shrink=0.92)
        cb.ax.tick_params(labelsize=14, direction="out")
        cb.set_label(bar_label, fontsize=16)

    # Surface recession line (red dashed) on every slab panel — initially hidden
    surf_lines = [
        ax.axhline(y=0, color="r", lw=1.0, ls="--", alpha=0.8)
        for ax in slab_axes
    ]

    # --- Time-history panel ---
    T_pad = (T_max - T_min) * 0.05
    ax_hist.set_ylim(T_min - T_pad, T_max + T_pad)
    ax_hist.plot(t_hist, T_wall_hist, "k-", lw=1.2, label="T_wall")
    for tc_name in tc_cols:
        lbl = tc_name.split("_at_")[1].replace("mm_K", "") + " mm"
        ax_hist.plot(t_hist, col(tc_name), "--", lw=0.8, label=f"TC @ {lbl}")
    t_pad_x = t_hist[-1] * 0.02
    ax_hist.set_xlim(-t_pad_x, t_hist[-1])
    ax_hist.set_xlabel("Time [s]", fontsize=17)
    ax_hist.set_ylabel("Temperature [K]", fontsize=17)
    ax_hist.legend(fontsize=15, loc="upper right")
    ax_hist.set_title("Time history", fontsize=18)
    ax_hist.tick_params(which="both", direction="out", labelsize=16,
                        top=True, labeltop=False)
    cursor = ax_hist.axvline(x=t_hist[0], color="r", lw=1.2, ls="--")
    s_rec_mm = (s_total - s_total[0]) * 1e3
    ax_rec = ax_hist.twinx()
    ax_rec.plot(t_hist, s_rec_mm, "b-", lw=1.0, alpha=0.65)
    ax_rec.set_ylabel("Recession [mm]", color="b", fontsize=16)
    ax_rec.tick_params(axis="y", labelcolor="b", labelsize=15, direction="out")
    ax_rec.set_ylim(0, max(s_rec_mm.max() * 1.2, 0.1))
    rec_dot, = ax_rec.plot([], [], "bo", ms=4)

    # Counter sits in the header band above all axes (top=0.84 → 1.0 is free)
    time_label = fig.text(
        0.5, 0.96, "", ha="center", va="top", fontsize=18, fontweight="bold",
    )

    # --- Per-frame interpolation onto fixed depth grid ---
    def _on_fixed_grid(data_col: np.ndarray, snap_idx: int) -> np.ndarray:
        """Resample a 1-D profile onto y_fixed_mm; NaN above current surface."""
        vals = data_col[:, snap_idx]
        y_act = (depth_profiles[:, snap_idx] if depth_profiles is not None
                 else np.tile(depth_m, (1,)).ravel()) * 1e3   # mm
        mask = np.isfinite(vals) & np.isfinite(y_act)
        if mask.sum() < 2:
            return np.full(n_display, np.nan)
        y_a, v_a = y_act[mask], vals[mask]
        order = np.argsort(y_a)
        y_a, v_a = y_a[order], v_a[order]
        out_arr = np.interp(y_fixed_mm, y_a, v_a, left=np.nan, right=np.nan)
        out_arr[y_fixed_mm < y_a[0]] = np.nan   # ablated zone → gray
        return out_arr

    def _hist_idx(snap_idx):
        return int(np.argmin(np.abs(t_hist - snap_times[snap_idx])))

    def update(i):
        for j, (_, data_arr, *_) in enumerate(panels):
            grid = _on_fixed_grid(data_arr, i)
            images[j].set_data(np.tile(grid[:, np.newaxis], (1, 2)))

        # Surface recession line position (depth of current surface in mm)
        hi = _hist_idx(i)
        s_mm = s_rec_mm[hi]                     # recession from original surface
        for ln in surf_lines:
            ln.set_ydata([s_mm, s_mm])

        time_label.set_text(f"t = {snap_times[i]:.1f} s")
        cursor.set_xdata([snap_times[i]])
        rec_dot.set_data([t_hist[hi]], [s_rec_mm[hi]])

    anim = FuncAnimation(
        fig, update, frames=n_snaps,
        blit=False, interval=1000 / fps,
    )

    mp4 = out / "results_slab.mp4"
    gif = out / "results_slab.gif"
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(str(mp4), writer=writer)
        print(f"Saved: {mp4}")
        plt.close(fig)
        return mp4
    except Exception:
        pass

    writer = PillowWriter(fps=fps)
    anim.save(str(gif), writer=writer)
    print(f"Saved: {gif}  (ffmpeg not available; saved as GIF)")
    plt.close(fig)
    return gif


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Plot SCAM CSV output (static PNG and optional video).",
    )
    p.add_argument("results_dir", help="Path to the SCAM results directory.")
    p.add_argument(
        "--video", action="store_true",
        help="Profile animation: temperature + char-front isoline.",
    )
    p.add_argument(
        "--slab", action="store_true",
        help="Slab animation: 2-D colored contours of T, β_total, β per component.",
    )
    p.add_argument(
        "--fps", type=int, default=15,
        help="Frames per second for the video (default: 15).",
    )
    p.add_argument(
        "--speed", type=float, default=None,
        help="Simulation seconds per real second of video (overrides --fps).",
    )
    p.add_argument(
        "--beta", type=float, default=0.01, metavar="LEVEL",
        help="Char-fraction isoline for --video (default: 0.01 = leading decomposition front).",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    out = Path(args.results_dir)
    make_static_figure(out)
    if args.video:
        make_video(out, fps=args.fps, speed=args.speed, beta_level=args.beta)
    if args.slab:
        make_slab_video(out, fps=args.fps, speed=args.speed)


if __name__ == "__main__":
    main()
