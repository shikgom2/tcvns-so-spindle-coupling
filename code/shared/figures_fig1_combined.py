# -*- coding: utf-8 -*-
"""Figure 1 — trajectories, four-region summary and electrode topography in one figure (new style).

a  frontal trajectory        b  parietal trajectory        c  four-region interaction β
d  during − before topography   e  after − before topography   + legend and colour bar

Brain double-column width (180 mm = 7.09 in).
The 'uncorrected' status of the star markers is stated in the caption, so it is not written in the figure.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_paths as LP  # noqa: E402
import figures_topography as TP                                   # noqa: E402
import paperstyle as PS                                           # noqa: E402

_R = LP.PROJECT_ROOT
P = _R if (_R / "outputs/paper").exists() else _R / "이전"
OUT = P / "outputs/paper"
TBL = _R / "Final analysis/tables"
FIG = _R / "Final analysis/figures"

C = PS.C
STIM, NOST = C["stim"], C["nost"]
ROIS = PS.ROIS
DET = "staresina_N2"
WINS = ("pre", "stim", "post")
WLAB = ["Before", "During", "After"]
RNG = np.random.default_rng(20261003)


def traj(ax, g, roi, I):
    """Per-window trajectory for one region — white-edged line + glossy dots + stimulation-window band."""
    t = g[(g.detector == DET) & (g.ROI == roi)]
    ax.axvspan(.5, 1.5, color=STIM, alpha=.07, lw=0, zorder=0)
    for arm, c in (("Sham", NOST), ("Treatment", STIM)):
        m, e = [], []
        for w in WINS:
            v = (t[(t.arm == arm) & (t.win == w)]
                 .groupby("participant").prevalence.mean().dropna().values)
            m.append(v.mean() * 100)
            e.append(v.std(ddof=1) / np.sqrt(v.size) * 100)
        m, e = np.array(m), np.array(e)
        ax.errorbar([0, 1, 2], m, yerr=e, fmt="none", ecolor=c, elinewidth=1.3,
                    capsize=3.0, capthick=1.1, alpha=.85, zorder=3)
        PS.line(ax, [0, 1, 2], m, c, lw=2.2, zorder=4)
        PS.dot(ax, [0, 1, 2], m, c, s=34, zorder=6)
    r = I[(I.detector == DET) & (I.ROI == roi)
          & (I.contrast == "stim vs pre")].iloc[0]
    ax.text(.035, .975,
            r"$\beta$ = %+.3f  $\bigstar$" % r.beta + "\n$q$ = %.3f" % r.q,
            transform=ax.transAxes, ha="left", va="top", fontsize=7.3,
            color=STIM, fontweight="bold", linespacing=1.5)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(WLAB)
    ax.set_xlim(-.38, 2.38)
    PS.title(ax, roi)


def forest(ax, I):
    t = (I[(I.detector == DET) & (I.contrast == "stim vs pre")]
         .set_index("ROI").reindex(ROIS))
    y = np.arange(len(ROIS))[::-1]
    ax.set_xlim(-.062, .192)
    ax.set_ylim(-.6, len(ROIS) - .4)
    for i, roi in enumerate(ROIS):
        r = t.loc[roi]
        sig = r.q < .05
        c = STIM if sig else C["ns"]
        PS.grad_ci(ax, r.ci_lo, r.ci_hi, y[i], c, center=r.beta, h=.30)
        PS.dot(ax, [r.beta], [y[i]], c, s=36 if sig else 26, zorder=6,
               halo=sig)
        ax.text(r.ci_hi + .008, y[i],
                "%+.3f%s" % (r.beta, "  \u2605" if sig else ""),
                fontsize=7.3, va="center",
                color=c if sig else C["ink2"],
                fontweight="bold" if sig else "normal")
    ax.axvline(0, color=C["ink"], lw=.9, ls=(0, (3, 2)), zorder=2)
    ax.set_yticks(y[::-1]); ax.set_yticklabels(ROIS[::-1])
    for tick, roi in zip(ax.get_yticklabels(), ROIS[::-1]):
        if t.loc[roi, "q"] < .05:
            tick.set_fontweight("bold"); tick.set_color(STIM)
    ax.set_xticks([-0.05, 0, 0.05, 0.10, 0.15])
    ax.set_xlabel(r"Interaction $\beta$")
    PS.title(ax, "All four regions")


def main():
    PS.use()
    g = pd.read_csv(OUT / "S_Table_session_metrics.csv")
    I = pd.read_csv(OUT / "S_Table_arm_x_phase.csv")
    Cc = pd.read_csv(TBL / "R1_channel_interaction.csv")
    Cc = Cc[Cc.detector == DET]

    f = plt.figure(figsize=(7.09, 5.15))
    gs = f.add_gridspec(2, 3, height_ratios=[1, 1.22],
                        width_ratios=[1, 1, 1.10],
                        wspace=.42, hspace=.34,
                        left=.084, right=.978, top=.930, bottom=.090)

    # ── a, b trajectories
    axa = f.add_subplot(gs[0, 0])
    traj(axa, g, "Frontal", I)
    axa.set_ylabel("Coupling prevalence (%)")
    axb = f.add_subplot(gs[0, 1], sharey=axa)
    traj(axb, g, "Parietal", I)
    plt.setp(axb.get_yticklabels(), visible=False)
    lo, hi = axa.get_ylim()
    axa.set_ylim(lo - (hi - lo) * .02, hi + (hi - lo) * .30)

    # ── c forest
    axc = f.add_subplot(gs[0, 2])
    forest(axc, I)

    # ── d, e topographies
    vmax = float(np.nanpercentile(np.abs(Cc.beta.values), 97)) or .05
    axd = f.add_subplot(gs[1, 0])
    axe = f.add_subplot(gs[1, 1])
    for ax, contr, lab, names in ((axd, "stim vs pre", "During − before",
                                   True),
                                  (axe, "post vs pre", "After − before",
                                   False)):
        t = Cc[Cc.contrast == contr]
        im = TP.topo(ax, dict(zip(t.ch, t.beta)),
                     {k + "_p": v for k, v in zip(t.ch, t.p)},
                     vmax, "", show_names=names)
        PS.title(ax, lab, pad=4)

    # ── bottom right: legend + star + vertical colour bar (aligned to the head-map height)
    axl = f.add_subplot(gs[1, 2]); axl.axis("off")
    for ax in (axa, axb, axc):
        PS.panel(ax)
        ax.grid(False); ax.set_axisbelow(True)
    f.canvas.draw()
    pl = axl.get_position()
    pdd = axd.get_position()
    handles = [Line2D([], [], color=STIM, marker="o", ms=5.0, lw=2.2,
                      mec="white", mew=.9, label="Stimulation"),
               Line2D([], [], color=NOST, marker="o", ms=5.0, lw=2.2,
                      mec="white", mew=.9, label="No stimulation")]
    f.legend(handles=handles, loc="upper left",
             bbox_to_anchor=(pl.x0 + .030, pl.y1 + .010), fontsize=7.4,
             handletextpad=.6, labelspacing=.55, borderpad=0)
    star = Line2D([], [], color="none", marker="*", ms=10,
                  markerfacecolor=TP.STAR, markeredgecolor=C["ink"],
                  markeredgewidth=.9, label="$P$ < .05")
    f.legend(handles=[star], loc="upper left",
             bbox_to_anchor=(pl.x0 + .030, pl.y1 - .100), fontsize=7.2,
             handletextpad=.45, borderpad=0)

    # Colour bar: within the vertical extent of the head map (d), shifted up to line up with the legend
    ch = pdd.y1 - pdd.y0
    cax = f.add_axes([pl.x0 + .052, pdd.y0 + ch * .04, .020, ch * .56])
    cb = f.colorbar(im, cax=cax,
                    ticks=np.round(np.linspace(-vmax, vmax, 5), 2))
    cb.set_label(r"Interaction $\beta$", labelpad=5)
    cb.ax.tick_params(labelsize=6.8, length=2.6, width=.8, pad=2)
    cb.outline.set_linewidth(.8)
    cb.outline.set_edgecolor("#C9C3B8")

    for ax, lt in ((axa, "a"), (axb, "b"), (axc, "c"),
                   (axd, "d"), (axe, "e")):
        PS.tag(f, ax, lt, dx=.058, dy=.024)

    out = FIG / "Fig1_main_combined"
    f.savefig(out.with_suffix(".png"), bbox_inches="tight")
    f.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(f)
    print("Fig1_main_combined  (restyled)")


if __name__ == "__main__":
    main()
