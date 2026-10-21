# -*- coding: utf-8 -*-
"""Figure 6 restructured — 'spatial spread' as the lead.

Changes
  a ← spatial extent → coupling-probability curve (strongest determinant moved to the front)
  b   slope difference
  c ← amplitude quintiles (weakest variable moved to the back)
  d   forest of six metrics, the two spread metrics moved to the top so they are read first
  e–i unchanged
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402
import paperstyle as PS                                           # noqa: E402
import figures_mechanism as FM                                    # noqa: E402
import figures_timecourse as FT                                   # noqa: E402
import figures_fig8_full as F8                                    # noqa: E402

R = LP.PROJECT_ROOT
TBL = R / "Final analysis/tables"
OUT = R / "Final analysis/manuscript/figures/Figure_06_spatial_extent_and_coupling"
STIM, NOST, INK, INK2 = PS.C["stim"], PS.C["nost"], PS.C["ink"], PS.C["ink2"]

# Two spread metrics on top — in the forest, ORDER[0] is placed at the top
SPREAD_FIRST = ["Derivations involved per wave",
                "Fraction of widespread waves",
                "Amplitude, coupled - uncoupled (uV)",
                "Slope, coupled waves (uV/s)",
                "Slope, uncoupled waves (uV/s)",
                "Slow-oscillation amplitude (uV)"]


def main():
    PS.use()
    B = pd.read_csv(TBL / "M7_time_course_bins.csv")
    L = pd.read_csv(TBL / "M7_lag_correlation.csv")

    f = plt.figure(figsize=(7.09, 6.6))
    gs = f.add_gridspec(3, 6, height_ratios=[1, 1, 1.05], wspace=1.75,
                        hspace=.60, left=.105, right=.975, top=.950, bottom=.105)
    # ── row 1: determinants, strongest first
    axa = f.add_subplot(gs[0, 0:2]); FM.panel_extent(axa)
    axb = f.add_subplot(gs[0, 2:4]); FM.panel_slope(axb)
    # ── metrics moved by stimulation go to the right of row 1
    axd = f.add_subplot(gs[0, 4:6])
    _old = FM.ORDER
    FM.ORDER = SPREAD_FIRST
    try:
        FM.panel_effects(axd, stars=False, p_x=9.6, xlim=(-4.3, 9.8))
    finally:
        FM.ORDER = _old
    axe = f.add_subplot(gs[1, 0:3])
    FT.series(axe, B, "extent", "Change in derivations per SO\nfrom before",
              "Spatial extent, aligned to onset", baseline=True, robust=True,
              min_n=12)
    axf = f.add_subplot(gs[1, 3:6])
    FT.series(axf, B, "prev", "Change in coupling from before (pp)",
              "Coupling, aligned to onset", scale=100, baseline=True,
              robust=True, min_n=12)
    # ── row 3: changes and their relationship
    axg = f.add_subplot(gs[2, 0:3])
    ch = F8.change_panel(axg, B, "extent", "Change in derivations per SO",
                         "Spatial extent: change from before")
    axh = f.add_subplot(gs[2, 3:6])
    cg = F8.change_panel(axh, B, "prev", "Change in coupling (pp)",
                         "Coupling: change from before", scale=100)
    # g, h: remove between-condition Mann-Whitney P and header — keep only stars (Wilcoxon vs 0)
    for ax_ in (axg, axh):
        for t in list(ax_.texts):
            tx = t.get_text()
            if tx.startswith("$P$ = ") or tx == "stimulation vs no stimulation":
                t.remove()
        # label directly inside the panel which bar is which condition
        for yy, lab, col in ((.985, "Stimulation", STIM),
                             (.900, "No stimulation", NOST)):
            ax_.text(.025, yy, lab, transform=ax_.transAxes, ha="left",
                     va="top", fontsize=6.3, color=col,
                     fontweight="bold")

    # b: narrow cell, so stars instead of P
    for t in axb.texts:
        txt = t.get_text()
        if "$P$" in txt:
            val = txt.split("\n")[0]
            t.set_text(val + "\n" + ("***" if "< .001" in txt else "**"))
    # d: remove direction labels, shrink text
    for t in list(axd.texts):
        if t.get_text() in ("smaller, shallower", "more widespread"):
            t.remove()
    axd.set_xticks([-4, -2, 0, 2, 4])
    for t in axd.get_yticklabels():
        t.set_fontsize(6.3)
    axd.set_xlabel("Interaction, after vs before\n(estimate ÷ s.e.)",
                   linespacing=1.3)
    # bold the two spread rows — make the reading order visible as well
    labs = axd.get_yticklabels()
    for t in labs:
        if t.get_text() in ("Derivations involved", "Fraction widespread"):
            t.set_fontweight("bold"); t.set_color(PS.C["up"])
    # shorten long axis labels so they do not overlap neighbouring panels
    axa.set_xlabel("Derivations with the same SO (\u00b1 400 ms)", fontsize=6.8)
    axb.set_ylabel("Slope difference,\ncoupled \u2212 uncoupled (\u00b5V s\u207b\u00b9)",
                   linespacing=1.3, fontsize=6.8)
    axf.set_ylabel("Change in coupling (pp)")
    for a_ in (axa, axb, axd, axe, axf, axg, axh):
        a_.set_title("")

    # c: shorten y labels and shift the axis right to avoid colliding with panel b
    _short = {"SO slope, coupled": "Slope, coupled",
              "SO slope, uncoupled": "Slope, uncoupled",
              "Slow-oscillation amplitude": "SO amplitude"}
    axd.set_yticklabels([_short.get(t.get_text(), t.get_text())
                         for t in axd.get_yticklabels()], fontsize=6.3)
    for t in axd.get_yticklabels():
        if t.get_text() in ("Derivations involved", "Fraction widespread"):
            t.set_fontweight("bold"); t.set_color(PS.C["up"])
    _pd = axd.get_position()
    axd.set_position([_pd.x0 + .030, _pd.y0, _pd.width - .030, _pd.height])
    f.canvas.draw()
    pa, pc = axa.get_position(), axd.get_position()
    keys = [Line2D([], [], color=FM.ROI[r], lw=2.0, label=r) for r in FM.ROIS]
    keys += [Line2D([], [], color=STIM, marker="o", ms=4.5, lw=1.5, mec="white",
                    label="Stimulation"),
             Line2D([], [], color=NOST, marker="o", ms=4.5, lw=1.5, mec="white",
                    label="No stimulation")]
    f.legend(handles=keys, loc="lower center", ncol=6,
             bbox_to_anchor=((pa.x0 + pc.x1) / 2, .012), fontsize=6.6,
             handlelength=1.3, handletextpad=.4, columnspacing=1.2, borderpad=0)
    for ax, lt in zip((axa, axb, axd, axe, axf, axg, axh), "abcdefg"):
        PS.tag(f, ax, lt, dx=.058, dy=.030)

    f.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    f.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(f)
    print("saved", OUT)
    print("extent changes", ch); print("coupling changes", cg)


if __name__ == "__main__":
    main()
