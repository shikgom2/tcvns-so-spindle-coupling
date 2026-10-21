# -*- coding: utf-8 -*-
"""Split the old Figure 1 in two — do not mix two families with different estimands.

Figure 1  condition x window interaction (difference of differences)
          a frontal trajectory  b parietal trajectory  c four-region interaction  d during-pre topography  e post-pre topography

Figure 2  between-condition contrast in the stimulation window (level difference)
          a frontal distribution  b parietal distribution  c Hedges g  d event-level OR
          e per-electrode map  f per-electrode dumbbell
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
from scipy import stats                                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402

warnings.filterwarnings("ignore")
import paperstyle as PS                                           # noqa: E402
import figures_raincloud as RC                                    # noqa: E402
import figures_topography as TP                                   # noqa: E402
import figures_fig1_combined as F1                                # noqa: E402
import figures_result2 as F4                                      # noqa: E402
import figures_fig1_merged as FM                                  # noqa: E402

R = LP.PROJECT_ROOT
OUTDIR = R / "Final analysis/manuscript/figures/v14oct2026"
C = PS.C
STIM, NOST, NS_ = C["stim"], C["nost"], C["ns"]
NOMC = "#B4651A"
ROIS = PS.ROIS
DET = "staresina_N2"
INKC = "#6B7378"


def dumbbell_v2(ax, Cc):
    """Per-electrode condition contrast — arrows, aligned g column, region bands."""
    import matplotlib.transforms as mtransforms
    from matplotlib.patches import Rectangle

    order = ROIS
    s = Cc.copy()
    s["rk"] = s.ROI.map({r: i for i, r in enumerate(order)})
    s = s.sort_values(["rk", "g"], ascending=[True, False]).reset_index(drop=True)
    y, ys, seps = 0.0, [], []
    for i in range(len(s)):
        if i and s.ROI[i] != s.ROI[i - 1]:
            y += .9
            seps.append(y - .45)
        ys.append(y)
        y += 1
    s["y"] = ys

    xmin = float(min(s.mean_trt.min(), s.mean_sham.min()))
    xmax = float(max(s.mean_trt.max(), s.mean_sham.max()))
    x_lo, x_hi = xmin - 1.1, xmax + 3.9
    x_g = xmax + 1.9                       # the g column is aligned at the same x for all rows
    ax.set_xlim(x_lo, x_hi)

    for gx in range(int(np.ceil(xmin / 5.0) * 5), int(xmax) + 1, 5):
        ax.axvline(gx, color="#EDE9E3", lw=.6, zorder=0)

    for _, r in s.iterrows():
        sig = r.q < .05
        col = STIM if sig else "#AFB5B9"
        if sig:
            ax.add_patch(Rectangle((x_lo, r.y - .46), x_hi - x_lo, .92,
                                   facecolor=STIM, alpha=.055, lw=0, zorder=0))
        ax.annotate("", xy=(r.mean_trt, r.y), xytext=(r.mean_sham, r.y),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=2.1 if sig else 1.5,
                                    shrinkA=3.0, shrinkB=2.0,
                                    mutation_scale=9.0 if sig else 7.0,
                                    alpha=.95 if sig else .75), zorder=3)
        ax.scatter([r.mean_sham], [r.y], s=17, color=NOST, zorder=5, lw=.7,
                   edgecolor="white")
        ax.text(x_g, r.y, "%+.2f" % r.g, va="center", ha="right",
                fontsize=6.8 if sig else 6.4,
                color=STIM if sig else "#8C9296",
                fontweight="bold" if sig else "normal")
        if sig:
            ax.text(x_g + .40, r.y, "★", va="center", ha="left",
                    fontsize=5.8, color=STIM)

    for sp_ in seps:
        ax.axhline(sp_, color="#E2DDD5", lw=.7, zorder=1)
    ax.set_yticks(s.y)
    ax.set_yticklabels(s.ch, fontsize=6.8)
    for tick, (_, r) in zip(ax.get_yticklabels(), s.iterrows()):
        if r.q < .05:
            tick.set_fontweight("bold"); tick.set_color(STIM)

    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for roi in order:
        yy = s[s.ROI == roi].y
        ax.plot([-.075, -.075], [yy.min() - .30, yy.max() + .30], transform=tr,
                color=PS.ROI.get(roi, "#9AA0A4"), lw=2.6,
                solid_capstyle="round", clip_on=False, zorder=4)
        ax.text(-.092, yy.mean(), roi, transform=tr, ha="center", va="center",
                fontsize=6.2, color=PS.ROI.get(roi, "#6B7378"),
                rotation=90, clip_on=False)

    ax.text(x_g, s.y.min() - 0.92, "Hedges' $g$", ha="right", va="center",
            fontsize=6.4, color=INKC, style="italic")
    ax.invert_yaxis()
    ax.set_xlabel("Coupling prevalence, stimulation window (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=1.5)
    return s


def _load():
    g = pd.read_csv(F1.OUT / "S_Table_session_metrics.csv")
    I = pd.read_csv(F1.OUT / "S_Table_arm_x_phase.csv")
    Cc = pd.read_csv(F1.TBL / "R1_channel_interaction.csv")
    Cc = Cc[Cc.detector == DET]
    B = pd.read_csv(F4.OUT / "S_Table_between_condition.csv")
    BS = B[(B.window == "stim") & (B.detector == DET)].set_index("ROI").reindex(ROIS)
    CB = F4.channel_between()
    return g, I, Cc, BS, CB


def figure1(g, I, Cc):
    """Interaction — difference of differences."""
    f = plt.figure(figsize=(7.09, 5.5))
    outer = f.add_gridspec(2, 1, height_ratios=[1.0, 1.12], hspace=.24,
                           left=.085, right=.975, top=.945, bottom=.065)
    r0 = outer[0].subgridspec(1, 3, width_ratios=[1, 1, 1.10], wspace=.44)
    r1 = outer[1].subgridspec(1, 3, width_ratios=[1, 1, 1.10], wspace=.20)

    axa = f.add_subplot(r0[0]); FM.traj_norm(axa, g, "Frontal", I)
    axa.set_ylabel("Coupling prevalence,\nchange from own baseline (pp)",
                   linespacing=1.3)
    axb = f.add_subplot(r0[1], sharey=axa); FM.traj_norm(axb, g, "Parietal", I)
    plt.setp(axb.get_yticklabels(), visible=False)
    lo, hi = axa.get_ylim()
    axa.set_ylim(lo - (hi - lo) * .02, hi + (hi - lo) * .30)
    axc = f.add_subplot(r0[2]); F1.forest(axc, I)

    vmax = float(np.nanpercentile(np.abs(Cc.beta.values), 97)) or .05
    axd = f.add_subplot(r1[0]); axe = f.add_subplot(r1[1])
    for ax, contr, lab, names in ((axd, "stim vs pre", "During \u2212 before", True),
                                  (axe, "post vs pre", "After \u2212 before", False)):
        t = Cc[Cc.contrast == contr]
        im = TP.topo(ax, dict(zip(t.ch, t.beta)),
                     {k + "_p": v for k, v in zip(t.ch, t.p)}, vmax, "",
                     show_names=names)
        ax.set_title(lab, pad=6)
    axl = f.add_subplot(r1[2]); axl.axis("off")

    f.canvas.draw()
    pr = axd.get_position(); y0, hh = pr.y0, pr.y1 - pr.y0
    ww = hh * 5.5 / 7.09
    L, GAP = .085, .040
    axd.set_position([L, y0, ww, hh])
    axe.set_position([L + ww + GAP, y0, ww, hh])

    f.canvas.draw()
    pdd, pee = axd.get_position(), axe.get_position()
    x_leg = pee.x1 + .055
    handles = [Line2D([], [], color=STIM, marker="o", ms=5, lw=2.0, mec="white",
                      label="Stimulation"),
               Line2D([], [], color=NOST, marker="o", ms=5, lw=2.0, mec="white",
                      label="No stimulation")]
    f.legend(handles=handles, loc="upper left",
             bbox_to_anchor=(x_leg, pee.y1 + .004), fontsize=7.2,
             handletextpad=.6, labelspacing=.55, borderpad=0)
    star = Line2D([], [], color="none", marker="*", ms=10,
                  markerfacecolor=TP.STAR, markeredgecolor=C["ink"],
                  markeredgewidth=.9, label="$P$ < .05")
    f.legend(handles=[star], loc="upper left",
             bbox_to_anchor=(x_leg, pee.y1 - .075), fontsize=7.0,
             handletextpad=.45, borderpad=0)
    ch = pdd.y1 - pdd.y0
    cax = f.add_axes([x_leg + .020, pdd.y0 + ch * .04, .018, ch * .50])
    cb = f.colorbar(im, cax=cax, ticks=np.round(np.linspace(-vmax, vmax, 5), 2))
    cb.set_label("Interaction " + chr(946), labelpad=5)
    cb.ax.tick_params(labelsize=6.8, length=2.6, width=.8, pad=2)
    cb.outline.set_linewidth(.8)

    for ax, lt in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d"), (axe, "e")):
        PS.tag(f, ax, lt, dx=.052, dy=.016)
    out = OUTDIR / "Figure_1_condition_by_window_interaction"
    f.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    f.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(f)
    print("saved", out.name)


def figure2(g, BS, CB):
    """Between-condition contrast in the stimulation window — level difference."""
    f = plt.figure(figsize=(7.09, 6.0))
    outer = f.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=.42,
                           left=.085, right=.975, top=.945, bottom=.075)
    r0 = outer[0].subgridspec(1, 4, width_ratios=[1, 1, .80, .80], wspace=.62)
    r1 = outer[1].subgridspec(1, 2, width_ratios=[1, 2.35], wspace=.22)

    tstim = g[(g.detector == DET) & (g.win == "stim")]
    rain = []
    for k, reg in enumerate(("Frontal", "Parietal")):
        ax = f.add_subplot(r0[k]); rain.append(ax)
        dat = {}
        for arm in ("Treatment", "Sham"):
            dat[arm] = (tstim[(tstim.ROI == reg) & (tstim.arm == arm)]
                        .groupby("participant").prevalence.mean()
                        .dropna().values * 100)
        u = stats.mannwhitneyu(dat["Treatment"], dat["Sham"])
        RC.raincloud(ax, dat, reg,
                     "Coupling prevalence,\nstimulation window (%)",
                     ptxt=("$P$ < .001" if u.pvalue < .001
                           else "$P$ = %.3f" % u.pvalue))
        ax.set_title(reg, pad=8)
        ax.set_xticklabels(["Stimulation", "No\nstimulation"],
                           fontsize=6.0, linespacing=1.2)
        ax.yaxis.label.set_size(7.0)
        if k:
            ax.set_ylabel("")
    axa, axb = rain
    axc = f.add_subplot(r0[2]); FM.forest_g(axc, BS)
    axc.set_title("Between conditions", pad=8)
    axc.set_xlabel("Hedges' $g$")
    axd = f.add_subplot(r0[3]); FM.forest_or(axd, BS)
    axd.set_title("Event level", pad=8)
    axd.set_xlabel("Odds ratio, per slow oscillation")

    axe = f.add_subplot(r1[0])
    vg = float(np.nanpercentile(np.abs(CB.g.values), 98))
    img = TP.topo(axe, dict(zip(CB.ch, CB.g)),
                  {k + "_p": v for k, v in zip(CB.ch, CB.q)}, vg, "",
                  show_names=False)
    axe.set_title("Electrode map", pad=6)
    axf = f.add_subplot(r1[1]); dumbbell_v2(axf, CB)
    axf.set_xlabel("Coupling prevalence, stimulation window (%)")
    axf.set_title("Every electrode: no stimulation \u2192 stimulation", pad=6)

    f.canvas.draw()
    pee = axe.get_position()
    cax = f.add_axes([pee.x0 + (pee.x1 - pee.x0) * .12, pee.y0 - .030,
                      (pee.x1 - pee.x0) * .76, .011])
    cb = f.colorbar(img, cax=cax, orientation="horizontal",
                    ticks=np.round(np.linspace(-vg, vg, 5), 2))
    cb.set_label("Hedges' $g$ per electrode", fontsize=6.6, labelpad=3)
    cb.ax.tick_params(labelsize=6.2, length=2.4, pad=1.6)
    cb.outline.set_linewidth(.8)

    keys = [Line2D([], [], marker="o", ls="none", ms=5.4, markerfacecolor=STIM,
                   markeredgecolor=STIM, label="q < .05"),
            Line2D([], [], marker="o", ls="none", ms=5.4, markerfacecolor=NOMC,
                   markeredgecolor=NOMC, label="P < .05"),
            Line2D([], [], marker="o", ls="none", ms=5.4, markerfacecolor=NS_,
                   markeredgecolor=NS_, label="not significant")]
    pa_, pb_ = axa.get_position(), axb.get_position()
    f.legend(handles=keys, loc="upper center",
             bbox_to_anchor=((pa_.x0 + pb_.x1) / 2 - .015, pa_.y0 - .068), ncol=3,
             fontsize=6.4, handletextpad=.35, columnspacing=1.3, borderpad=0)

    for ax, lt in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d"),
                   (axe, "e"), (axf, "f")):
        PS.tag(f, ax, lt, dx=.052, dy=.016)
    out = OUTDIR / "Figure_2_between_condition_during_stimulation"
    f.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    f.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(f)
    print("saved", out.name)


def main():
    PS.use()
    g, I, Cc, BS, CB = _load()
    figure1(g, I, Cc)
    figure2(g, BS, CB)


if __name__ == "__main__":
    main()
