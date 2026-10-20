# -*- coding: utf-8 -*-
"""Result 1 as scalp topography — per-electrode condition x window interaction.

The montage has 13 channels (no occipital sites), so interpolation is restricted
to the area actually covered by electrodes, and every electrode position is drawn
as a dot so that uncovered regions are visible.
Significant electrodes are marked with a star + bold outline.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe                               # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
import statsmodels.formula.api as smf                             # noqa: E402
from matplotlib.patches import Circle, Polygon                    # noqa: E402
from scipy.interpolate import griddata                            # noqa: E402
from scipy.ndimage import gaussian_filter                         # noqa: E402

warnings.filterwarnings("ignore")
_R = Path(r"D:/SO-Spindle논문")
P = _R if (_R / "outputs/paper").exists() else _R / "이전"
OUT = P / "outputs/paper"
TBL = _R / "Final analysis/tables"
FIG = _R / "Final analysis/figures"
TBL.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.linewidth": .7, "savefig.dpi": 400, "figure.dpi": 140,
})

# 10-20 coordinates (unit circle, nose at +y). 13 channels.
POS = {
    "Fp1": (-.26, .83), "Fp2": (.26, .83),
    "F7": (-.72, .40), "F3": (-.36, .42), "F4": (.36, .42), "F8": (.72, .40),
    "T3": (-.85, .00), "C3": (-.42, .00), "Cz": (.00, .00),
    "C4": (.42, .00), "T4": (.85, .00),
    "P3": (-.36, -.42), "P4": (.36, -.42),
}
ROI_OF = {"Fp1": "Frontal", "Fp2": "Frontal", "F3": "Frontal",
          "F4": "Frontal", "F7": "Temporal", "F8": "Temporal",
          "T3": "Temporal", "T4": "Temporal", "C3": "Central",
          "Cz": "Central", "C4": "Central", "P3": "Parietal",
          "P4": "Parietal"}
DETS = ["staresina_N2", "jy_N2", "staresina_N2N3", "jy_N2N3"]
DL = {"staresina_N2": "Staresina · N2", "jy_N2": "Reference · N2",
      "staresina_N2N3": "Staresina · N2+N3", "jy_N2N3": "Reference · N2+N3"}
STAR = "#E8B51F"            # star colour — yellow


def fdr(p):
    p = np.asarray(p, float)
    o = np.full_like(p, np.nan)
    k = np.isfinite(p)
    m = int(k.sum())
    if m == 0:
        return o
    i = np.argsort(p[k])
    q = np.minimum.accumulate((p[k][i] * m / (np.arange(m) + 1))[::-1])[::-1]
    t = np.empty(m)
    t[i] = q.clip(0, 1)
    o[k] = t
    return o


def per_channel_interaction():
    """Per electrode: prevalence ~ condition * window + (1|participant)."""
    g = pd.read_csv(OUT / "S_Table_channel_metrics.csv")
    rows = []
    for det in DETS:
        for ch in POS:
            t = g[(g.detector == det) & (g.ch == ch)
                  & g.win.isin(["pre", "stim", "post"])].dropna(
                subset=["prevalence"]).copy()
            if t.participant.nunique() < 15:
                continue
            t["arm_b"] = (t.arm == "Treatment").astype(int)
            t["win"] = pd.Categorical(t.win, ["pre", "stim", "post"])
            try:
                m = smf.mixedlm("prevalence ~ arm_b * C(win)", t,
                                groups=t.participant).fit()
            except Exception:
                continue
            for term, lab in (("arm_b:C(win)[T.stim]", "stim vs pre"),
                              ("arm_b:C(win)[T.post]", "post vs pre")):
                if term not in m.params.index:
                    continue
                b, se = float(m.params[term]), float(m.bse[term])
                rows.append(dict(detector=det, ch=ch, ROI=ROI_OF[ch],
                                 contrast=lab, n=t.participant.nunique(),
                                 beta=b, ci_lo=b - 1.96 * se,
                                 ci_hi=b + 1.96 * se,
                                 p=float(m.pvalues[term])))
    T = pd.DataFrame(rows)
    T["q"] = T.groupby(["detector", "contrast"]).p.transform(
        lambda p: fdr(p.values))
    T.to_csv(TBL / "R1_channel_interaction.csv", index=False,
             encoding="utf-8-sig")
    return T


import mne                                                        # noqa: E402

mne.set_log_level("ERROR")
# The MNE standard montage uses the names T7/T8 (recordings use T3/T4)
MON = {"T3": "T7", "T4": "T8"}
SPH = 0.105          # head radius — slightly enlarged so that all electrodes fit inside
_INFO = None


def _info():
    """13-channel EEG Info + standard_1020 montage."""
    global _INFO
    if _INFO is None:
        names = [MON.get(c, c) for c in POS]
        i = mne.create_info(names, sfreq=125.0, ch_types="eeg")
        i.set_montage(mne.channels.make_standard_montage("standard_1020"))
        _INFO = i
    return _INFO


def topo_mne(ax, vals, pvals, vmax, title, show_names=False):
    """MNE standard topomap. Significant electrodes are marked with yellow stars."""
    info = _info()
    order = list(POS)
    z = np.array([vals.get(c, np.nan) for c in order], float)
    mask = np.array([np.isfinite(pvals.get(c + "_p", np.nan))
                     and pvals[c + "_p"] < .05 for c in order])
    z = np.where(np.isfinite(z), z, 0.0)
    mp = dict(marker="*", markerfacecolor=STAR, markeredgecolor="#1A1D26",
              markeredgewidth=1.2, linewidth=0, markersize=20)
    im, _ = mne.viz.plot_topomap(
        z, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-vmax, vmax),
        contours=4, sensors="ko", outlines="head", extrapolate="head",
        image_interp="cubic", res=256, mask=mask, mask_params=mp,
        sphere=(0.0, 0.0, 0.0, SPH),
        names=[MON.get(c, c) for c in order] if show_names else None)
    # Clip the coloured area to exactly the same circle as the head outline
    im.set_clip_path(Circle((0, 0), SPH, transform=ax.transData))
    for ln in ax.lines:                       # tidy electrode marker sizes
        if ln.get_marker() == "*":
            # It is a Line2D, so markers get joined unless the line is switched off
            ln.set_linestyle("none")
            ln.set_markersize(20)
            ln.set_markeredgewidth(1.2)
            ln.set_markeredgecolor("#1A1D26")
            ln.set_zorder(9)
        elif ln.get_marker() in ("o", "."):
            ln.set_markersize(2.0)
            ln.set_markerfacecolor("#1A1D26")
            ln.set_markeredgewidth(0)
    # Move labels above the dots. Starred electrodes are moved further to clear the star (20 pt).
    smask = dict(zip([MON.get(c, c) for c in order], mask))
    for t in ax.texts:
        x0, y0 = t.get_position()
        t.set_position((x0, y0 + (.0205 if smask.get(t.get_text().strip(),
                                                     False) else .0085)))
        t.set_fontsize(5.2)
        t.set_color("#2B3138")
        t.set_ha("center")
        t.set_path_effects([pe.withStroke(linewidth=1.7, foreground="white")])
        t.set_zorder(11)
    ax.set_title(title, fontsize=8.2, pad=3)
    return im


def head(ax):
    ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=1.3, color="#1A1D26",
                        zorder=6))
    ax.add_patch(Polygon([(-.10, .995), (0, 1.12), (.10, .995)], closed=True,
                         fill=False, lw=1.3, color="#1A1D26", zorder=6))
    for s in (-1, 1):
        ax.add_patch(Circle((s * 1.02, .02), .055, fill=False, lw=1.2,
                            color="#1A1D26", zorder=6))
    ax.set_xlim(-1.26, 1.26); ax.set_ylim(-1.18, 1.26)
    ax.set_aspect("equal"); ax.axis("off")


def topo_hull(ax, vals, qs, vmax, title, show_names=False):
    """(Unused) In-house implementation that draws only inside the electrode convex hull.

    RBF multiquadric behaves erratically when extrapolating, so linear interpolation
    plus nearest-electrode filling is used and everything outside the convex hull is masked.
    """
    chs = [c for c in POS if c in vals and np.isfinite(vals[c])]
    x = np.array([POS[c][0] for c in chs])
    y = np.array([POS[c][1] for c in chs])
    z = np.array([vals[c] for c in chs])

    N = 320
    gx, gy = np.meshgrid(np.linspace(-1.05, 1.05, N),
                         np.linspace(-1.05, 1.05, N))
    pts = np.column_stack([x, y])
    # No extrapolation. Interpolate only inside the area enclosed by electrodes (convex hull)
    # and leave the rest empty — there are no occipital electrodes.
    zi = griddata(pts, z, (gx, gy), method="cubic")
    inside = np.isfinite(zi)
    sm = gaussian_filter(np.where(inside, zi, 0.0), 3.0)
    wt = gaussian_filter(inside.astype(float), 3.0)
    zi = np.where(wt > 1e-6, sm / np.maximum(wt, 1e-6), np.nan)
    zi = np.ma.masked_where(~inside | (gx ** 2 + gy ** 2 > .995), zi)

    im = ax.contourf(gx, gy, zi, levels=np.linspace(-vmax, vmax, 33),
                     cmap="RdBu_r", vmin=-vmax, vmax=vmax, zorder=1,
                     extend="both")
    im.set_edgecolor("face")                        # prevent striping in vector output
    ax.contour(gx, gy, zi, levels=[0], colors="#55606B", linewidths=.55,
               linestyles="dashed", zorder=2)
    for c in POS:
        xx, yy = POS[c]
        ok = c in vals and np.isfinite(vals[c])
        # Stars mark uncorrected P < .05 only (no electrode survives FDR at the
        # electrode level — stated in the caption)
        sig = ok and qs.get(c + "_p", 1) < .05
        ax.scatter([xx], [yy], s=40 if sig else 14,
                   facecolor=STAR if sig else ("white" if ok else "#DDD8D0"),
                   edgecolor="#1A1D26", lw=1.1 if sig else .7, zorder=7)
        if sig:
            ax.text(xx, yy - .004, "★", fontsize=8.4, ha="center",
                    va="center", color=STAR, zorder=8,
                    path_effects=[pe.withStroke(linewidth=1.6,
                                                foreground="#1A1D26")])
        if show_names:
            # Push labels outward from the centre so they do not overlap
            rr = np.hypot(xx, yy) or 1.0
            off = .155
            lx = xx + (xx / rr) * off * (.55 if abs(xx) < .2 else 1.0)
            ly = yy + (yy / rr) * off * (.9 if abs(yy) < .2 else 1.0)
            if abs(yy) < .05:                      # midline row goes below
                ly = yy - .155
            ax.text(lx, ly, c, fontsize=5.4, ha="center", va="center",
                    color="#44495C", zorder=9,
                    path_effects=[pe.withStroke(linewidth=1.8,
                                                foreground="white")])
    head(ax)
    ax.set_title(title, fontsize=8.2, pad=2)
    return im


# The one actually used is the MNE standard topomap (fills the whole head)
topo = topo_mne


def add_cbar(f, im, ax, vmax, label, frac=.62, pad=.052, nticks=5):
    """Horizontal colour bar below the head plot, same specification for all figures."""
    p = ax.get_position()
    w = (p.x1 - p.x0) * frac
    cax = f.add_axes([p.x0 + ((p.x1 - p.x0) - w) / 2, p.y0 - pad, w, .022])
    cb = f.colorbar(im, cax=cax, orientation="horizontal",
                    ticks=np.round(np.linspace(-vmax, vmax, nticks), 2))
    cb.set_label(label, fontsize=8.6, labelpad=4)
    cb.ax.tick_params(labelsize=8.0, length=3.2, width=.9, pad=2)
    cb.outline.set_linewidth(.9)
    return cb

CONTR = (("stim vs pre", "During − before"),
         ("post vs pre", "After − before"))


def _draw(T, dets, fname, title, sub, vmax, _unused=None):
    """Compute the layout in inches so that each head plot is square."""
    nrow = len(dets)
    WIDE, L, Rr = 5.8, .085, .775            # total width, left/right margins (fractions)
    HEAD_IN, FOOT_IN = 1.15, .50             # header / footer height (inches)
    col_in = WIDE * (Rr - L) / 2             # width of one column (inches)
    rows_in = nrow * col_in * 1.02 + (nrow - 1) * .04
    tall = HEAD_IN + FOOT_IN + rows_in
    f = plt.figure(figsize=(WIDE, tall))
    top = 1 - HEAD_IN / tall
    gs = f.add_gridspec(nrow, 2, wspace=.02, hspace=.04,
                        left=L, right=Rr, top=top, bottom=FOOT_IN / tall)
    y1, y2, y3 = (1 - .20 / tall, 1 - .52 / tall, 1 - .80 / tall)
    for r, det in enumerate(dets):
        for c, (contr, lab) in enumerate(CONTR):
            t = T[(T.detector == det) & (T.contrast == contr)]
            ax = f.add_subplot(gs[r, c])
            im = topo(ax, dict(zip(t.ch, t.beta)),
                      {k + "_p": v for k, v in zip(t.ch, t.p)},
                      vmax, lab if r == 0 else "",
                      show_names=(r == 0 and c == 0))
            if c == 0:                       # row labels in figure coordinates
                p = ax.get_position()
                f.text(L - .022, (p.y0 + p.y1) / 2, DL[det], rotation=90,
                       va="center", ha="center", fontsize=8,
                       color="#1A1D26")
    cbh = min(.42, rows_in * .50 / tall)
    cax = f.add_axes([.800, (1 - cbh) / 2 - .02, .019, cbh])
    cb = f.colorbar(im, cax=cax)
    cb.set_label(r"Interaction $\beta$", fontsize=7, labelpad=3)
    cb.ax.tick_params(labelsize=6.2)
    f.text(L, y1, title, fontsize=10.5, ha="left", va="top")
    f.text(L, y2, sub, fontsize=6.8, color="#6B7378", ha="left", va="top")
    lowq = T[T.detector.isin(dets)].q.min()
    f.text(L, y3,
           "★ (yellow) marks P < .05 uncorrected. No electrode survives FDR "
           "correction (lowest q = %.3f); these stars show uncorrected "
           "p-values only." % lowq,
           fontsize=6.8, color="#96353A", ha="left", va="top")
    f.text(L, .012,
           "Black dots are the 13 recorded derivations. The montage has no "
           "occipital electrode, so posterior values are interpolated and "
           "should not be read\nas measurements. Region-level estimates, "
           "which pool slow oscillations across electrodes, do survive "
           "correction and are given in the main trajectory figure.",
           fontsize=6.3, color="#6B7378", ha="left", va="bottom")
    f.savefig(FIG / (fname + ".png"), bbox_inches="tight")
    f.savefig(FIG / (fname + ".pdf"), bbox_inches="tight")
    plt.close(f)
    print("%-34s electrodes P<.05 %d  q<.05 %d"
          % (fname, int((T[T.detector.isin(dets)].p < .05).sum()),
             int((T[T.detector.isin(dets)].q < .05).sum())))


def fig_topo(T):
    vmax = float(np.nanpercentile(np.abs(T.beta.values), 97)) or .05
    # Main text — Staresina · N2 only
    _draw(T, ["staresina_N2"], "Fig1b_topography_main",
          "Where on the scalp the trajectory changes",
          "Per-electrode condition × window interaction · "
          "Staresina detector, N2", vmax)
    # Supplementary — the remaining three configurations
    _draw(T, ["jy_N2", "staresina_N2N3", "jy_N2N3"],
          "FigS_topography_detectors",
          "The same map under the other detector configurations",
          "Per-electrode condition × window interaction", vmax)
    s = T[(T.detector == "staresina_N2") & (T.contrast == "stim vs pre")]
    print(s.sort_values("p")[["ch", "ROI", "beta", "p", "q"]].round(4)
          .to_string(index=False))


if __name__ == "__main__":
    T = per_channel_interaction()
    fig_topo(T)
    print("\n[saved] %s" % FIG)
