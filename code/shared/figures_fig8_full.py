# -*- coding: utf-8 -*-
"""Figure 8 (combined) — which SOs recruit spindles · what stimulation changes ·
whether that is time-locked to stimulation.

Row 1  a amplitude    b slope    c spatial extent            (which SOs couple)
Row 2  d six-metric direction   e coupling time course   f spatial-extent time course  (what stimulation changes, over time)
Row 3  g coupling change  h spatial-extent change  i lag correlation     (per-condition changes side by side)

Panels up to d reuse functions from figures_mechanism; e·f·i reuse those from figures_timecourse.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import matplotlib.transforms as mtransforms                       # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from scipy import stats                                           # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_paths as LP  # noqa: E402
import paperstyle as PS                                           # noqa: E402
import figures_mechanism as FM                                    # noqa: E402
import figures_timecourse as FT                                   # noqa: E402

R = LP.PROJECT_ROOT
TBL = R / "Final analysis/tables"
FIG = R / "Final analysis/figures"
STIM, NOST, INK, INK2 = PS.C["stim"], PS.C["nost"], PS.C["ink"], PS.C["ink2"]
RNG = np.random.default_rng(20261003)


def change_panel(ax, B, y, ylab, title, scale=1.0):
    """Within-recording changes per condition (during−before, after−before) side by side — magnitudes comparable at a glance."""
    Bb = B.copy()
    Bb["phase"] = np.where(Bb.bin < 0, 0, np.where(Bb.bin < FT.STIM_END, 1, 2))
    W = (Bb.groupby(["file", "arm", "phase"])[y].mean().unstack("phase")
         .dropna()) * scale
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    xs = {("Treatment", 1): 0 - .17, ("Sham", 1): 0 + .17,
          ("Treatment", 2): 1 - .17, ("Sham", 2): 1 + .17}
    vals = {}
    for arm, c in (("Treatment", STIM), ("Sham", NOST)):
        w = W.xs(arm, level="arm")
        for ph in (1, 2):
            d = (w[ph] - w[0]).values
            vals[(arm, ph)] = d
            x = xs[(arm, ph)]
            bs = RNG.choice(d, (4000, d.size), True).mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            ax.bar([x], [d.mean()], width=.30, color=c, alpha=.85, lw=0,
                   zorder=2)
            ax.plot([x, x], [lo, hi], color=INK, lw=1.3, zorder=4,
                    solid_capstyle="round")
            p = stats.wilcoxon(d).pvalue
            ax.text(x, max(hi, 0), ("%+.2f" if scale == 1 else "%+.1f") % d.mean()
                    + ("\n***" if p < .001 else "\n**" if p < .01
                       else "\n*" if p < .05 else "\nns"),
                    ha="center", va="bottom", fontsize=6.4, color=c,
                    linespacing=1.3, fontweight="bold")
    for ph, xg in ((1, 0), (2, 1)):
        pb = stats.mannwhitneyu(vals[("Treatment", ph)], vals[("Sham", ph)]).pvalue
        ax.text(xg, .955, "$P$ = %.2f" % pb, transform=tr, ha="center",
                va="top", fontsize=6.2, color=INK2)
    ax.text(.5, 1.015, "stimulation vs no stimulation", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.0, color=INK2, style="italic")
    ax.axhline(0, color=INK, lw=.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["During\n− before", "After\n− before"],
                       linespacing=1.3)
    ax.set_xlim(-.6, 1.6)
    lo, hi = ax.get_ylim(); ax.set_ylim(min(lo, 0), hi + (hi - lo) * .40)
    ax.set_ylabel(ylab)
    ax.set_title(title, pad=16)
    return {k: (v.mean(), len(v)) for k, v in vals.items()}


def main():
    PS.use()
    B = pd.read_csv(TBL / "M7_time_course_bins.csv")
    L = pd.read_csv(TBL / "M7_lag_correlation.csv")

    f = plt.figure(figsize=(7.09, 8.0))
    gs = f.add_gridspec(3, 3, wspace=.58, hspace=.46, left=.095, right=.975,
                        top=.950, bottom=.085)
    axa = f.add_subplot(gs[0, 0]); FM.panel_amp(axa)
    axb = f.add_subplot(gs[0, 1]); FM.panel_slope(axb)
    axc = f.add_subplot(gs[0, 2]); FM.panel_extent(axc)
    axd = f.add_subplot(gs[1, 0])
    FM.panel_effects(axd, stars=False, p_x=9.6, xlim=(-4.3, 9.8))
    axe = f.add_subplot(gs[1, 1])
    FT.series(axe, B, "prev", "Change in coupling from before (pp)",
              "Coupling, aligned to onset", scale=100, baseline=True,
              robust=True, min_n=12)
    axf = f.add_subplot(gs[1, 2])
    FT.series(axf, B, "extent", "Change in derivations per SO\nfrom before",
              "Spatial extent, aligned to onset", baseline=True, robust=True,
              min_n=12)
    axg = f.add_subplot(gs[2, 0])
    cg = change_panel(axg, B, "prev", "Change in coupling (pp)",
                      "Coupling: change from before", scale=100)
    axh = f.add_subplot(gs[2, 1])
    ch = change_panel(axh, B, "extent", "Change in derivations per SO",
                      "Spatial extent: change from before")
    axi = f.add_subplot(gs[2, 2]); FT.lagpanel(axi, L)
    axi.set_title("Coupling × extent, within recording", pad=18)
    axi.set_ylabel("Spearman $r$")
    axi.set_xticklabels(["extent\nleads", "same\nbin", "coupling\nleads"],
                        fontsize=6.4, linespacing=1.3)

    # b: narrow cell, so stars instead of P values
    for t in axb.texts:
        txt = t.get_text()
        if "$P$" in txt:
            val = txt.split("\n")[0]
            star = "***" if "< .001" in txt else "**"
            t.set_text(val + "\n" + star)
    # d: narrow cell, so remove direction labels
    for t in list(axd.texts):
        if t.get_text() in ("smaller, shallower", "more widespread"):
            t.remove()
    axi.set_title("Coupling × extent,\nwithin recording", pad=22,
                  linespacing=1.3)
    axi.set_title("Coupling × extent,\nwithin recording", pad=30,
                  linespacing=1.3)
    axb.set_title("Coupled waves rise more steeply", pad=30)
    # panel d sits in a narrow cell, so shrink axis and text
    axd.set_xticks([-4, -2, 0, 2, 4])
    axd.set_title("Stimulation shifts the\npopulation in two directions",
                  pad=6, linespacing=1.3)
    for t in axd.get_yticklabels():
        t.set_fontsize(6.3)
    axd.set_xlabel("Interaction, after vs before\n(estimate ÷ s.e.)",
                   linespacing=1.3)

    for a_ in (axa, axb, axc, axd, axe, axf, axg, axh, axi):
        a_.set_title("")
    f.canvas.draw()
    pa, pc = axa.get_position(), axc.get_position()
    keys = [Line2D([], [], color=FM.ROI[r], lw=2.0, label=r) for r in FM.ROIS]
    keys += [Line2D([], [], color=STIM, marker="o", ms=4.5, lw=1.5,
                    mec="white", label="Stimulation"),
             Line2D([], [], color=NOST, marker="o", ms=4.5, lw=1.5,
                    mec="white", label="No stimulation")]
    f.legend(handles=keys, loc="lower center", ncol=6,
             bbox_to_anchor=((pa.x0 + pc.x1) / 2, .012),
             fontsize=6.6, handlelength=1.3, handletextpad=.4,
             columnspacing=1.2, borderpad=0)
    for ax, lt in zip((axa, axb, axc, axd, axe, axf, axg, axh, axi),
                      "abcdefghi"):
        PS.tag(f, ax, lt, dx=.058, dy=.030)

    f.savefig(FIG / "FigM_mechanism_full.png", bbox_inches="tight")
    f.savefig(FIG / "FigM_mechanism_full.pdf", bbox_inches="tight")
    plt.close(f)
    print("FigM_mechanism_full (a–i)  outliers removed in e/f:", getattr(FT.series, "n_removed", None))
    print("coupling changes", cg); print("extent changes", ch)


if __name__ == "__main__":
    main()
