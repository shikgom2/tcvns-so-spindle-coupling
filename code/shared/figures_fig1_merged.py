# -*- coding: utf-8 -*-
"""Figure 1 (merged) — coupling prevalence rises with stimulation: within-recording interaction + between-condition contrast in the stimulation window.

Row 1  a frontal trajectory   b parietal trajectory   c interaction β for the four regions
Row 2  d during−before topography   e after−before topography   (legend · colour bar)
Row 3  f per-participant values, stimulation window   g Hedges g   h event-level OR   i per-electrode dumbbell

a–e reuse the data and rules of figures_fig1_combined, f–i those of figures_result2, unchanged.
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
from scipy import stats                                           # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_paths as LP  # noqa: E402
import paperstyle as PS                                           # noqa: E402
import figures_raincloud as RC
import figures_topography as TP                                   # noqa: E402
import figures_fig1_combined as F1                                # noqa: E402
import figures_result2 as F4                                      # noqa: E402
import figures_schematic as SCH                                   # noqa: E402

_R = LP.PROJECT_ROOT
FIG = _R / "Final analysis/figures"
C = PS.C
STIM, NOST, NS_ = C["stim"], C["nost"], C["ns"]
NOMC = "#B4651A"
ROIS = PS.ROIS
RNG = np.random.default_rng(20261003)
DET = "staresina_N2"


def panel_f(ax, G, BS):
    """Per-participant coupling prevalence in the stimulation window, both conditions."""
    t = G[(G.detector == DET) & (G.win == "stim")]
    for k, (arm, c) in enumerate((("Sham", NOST), ("Treatment", STIM))):
        xs, ms, los, his = [], [], [], []
        for i, roi in enumerate(ROIS):
            v = (t[(t.arm == arm) & (t.ROI == roi)]
                 .groupby("participant").prevalence.mean().dropna().values)
            x = i + (k - .5) * .30
            ax.scatter(x + RNG.uniform(-.075, .075, v.size), v * 100, s=3.5,
                       color=c, alpha=.30, lw=0, zorder=1)
            bs = RNG.choice(v, (4000, v.size), True).mean(1) * 100
            xs.append(x); ms.append(v.mean() * 100)
            los.append(np.percentile(bs, 2.5)); his.append(np.percentile(bs, 97.5))
        ax.errorbar(xs, ms, yerr=[np.array(ms) - los, np.array(his) - ms],
                    fmt="none", ecolor=c, elinewidth=1.4, capsize=2.8,
                    capthick=1.1, zorder=3)
        PS.dot(ax, xs, ms, c, s=36, zorder=6)
    ax.set_xticks(range(4)); ax.set_xticklabels([r[:4] for r in ROIS])
    ax.set_ylabel("Coupling prevalence,\nstimulation window (%)",
                  linespacing=1.4)
    ax.set_ylim(0, 62)
    r = BS.loc["Parietal"]
    ax.plot([2.85, 2.85, 3.15, 3.15], [44, 45.6, 45.6, 44], color=STIM, lw=.9)
    ax.text(3, 46.6, "$q$ = %.3f" % r.q_mwu, ha="center", va="bottom",
            fontsize=7, color=STIM, fontweight="bold")


def forest_g(ax, BS):
    y = np.arange(4)[::-1]
    for i, roi in enumerate(ROIS):
        r = BS.loc[roi]
        sig = r.q_mwu < .05; nom = (not sig) and r.p_mwu < .05
        col = STIM if sig else (NOMC if nom else NS_)
        ax.plot([r.g_lo, r.g_hi], [y[i], y[i]], color=col, lw=1.9,
                solid_capstyle="round")
        ax.scatter([r.g], [y[i]], s=40, zorder=6, lw=1.2,
                   facecolor=col if (sig or nom) else "white", edgecolor=col)
        ax.text(1.95, y[i], "%+.2f%s" % (r.g, " ★" if sig else ""),
                fontsize=7, va="center", ha="right",
                color=col if sig else "#8C9296",
                fontweight="bold" if sig else "normal")
    ax.axvline(0, color=C["ink"], lw=.9, ls=(0, (3, 2)))
    ax.set_yticks(y[::-1]); ax.set_yticklabels(ROIS[::-1])
    for tick, roi in zip(ax.get_yticklabels(), ROIS[::-1]):
        if BS.loc[roi, "q_mwu"] < .05:
            tick.set_fontweight("bold"); tick.set_color(STIM)
    ax.set_ylim(-.6, 3.4); ax.set_xlim(-.45, 2.0)
    ax.set_xticks([0, .5, 1.0]); ax.set_xlabel("Hedges' $g$")


def forest_or(ax, BS):
    y = np.arange(4)[::-1]
    for i, roi in enumerate(ROIS):
        r = BS.loc[roi]
        sig = r.q_gee < .05 if np.isfinite(r.q_gee) else False
        nom = (not sig) and r.p_gee < .05
        col = STIM if sig else (NOMC if nom else NS_)
        ax.plot([r.OR_lo, r.OR_hi], [y[i], y[i]], color=col, lw=1.9,
                solid_capstyle="round")
        ax.scatter([r.OR], [y[i]], s=40, zorder=6, lw=1.2,
                   facecolor=col if (sig or nom) else "white", edgecolor=col)
        ax.text(2.75, y[i], "%.2f%s" % (r.OR, " ★" if sig else
                                        (" ☆" if nom else "")),
                fontsize=7, va="center", ha="right",
                color=col if (sig or nom) else "#8C9296",
                fontweight="bold" if sig else "normal")
    ax.axvline(1, color=C["ink"], lw=.9, ls=(0, (3, 2)))
    ax.set_yticks(y[::-1]); ax.set_yticklabels([])
    ax.set_ylim(-.6, 3.4); ax.set_xlim(.70, 2.8)
    ax.set_xticks([1.0, 1.5, 2.0])
    ax.set_xlabel("Odds ratio, per slow oscillation")


def dumbbell(ax, Cc):
    order = ROIS
    s = Cc.copy(); s["rk"] = s.ROI.map({r: i for i, r in enumerate(order)})
    s = s.sort_values(["rk", "g"], ascending=[True, False]).reset_index(drop=True)
    y = 0; ys = []; seps = []
    for i in range(len(s)):
        if i and s.ROI[i] != s.ROI[i - 1]:
            y += 1; seps.append(y - .5)
        ys.append(y); y += 1
    s["y"] = ys
    for _, r in s.iterrows():
        sig = r.q < .05; c = STIM if sig else NS_
        ax.plot([r.mean_sham, r.mean_trt], [r.y, r.y], color=c, lw=2.0,
                solid_capstyle="round", zorder=2, alpha=.9 if sig else .7)
        ax.scatter([r.mean_sham], [r.y], s=24, color=NOST, zorder=4, lw=.7,
                   edgecolor="white")
        ax.scatter([r.mean_trt], [r.y], s=24, color=STIM, zorder=4, lw=.7,
                   edgecolor="white")
        ax.text(max(r.mean_trt, r.mean_sham) + .35, r.y,
                "%+.2f%s" % (r.g, " ★" if sig else ""), va="center",
                fontsize=6.6, color=STIM if sig else "#8C9296",
                fontweight="bold" if sig else "normal")
    for sp_ in seps:
        ax.axhline(sp_, color="#D7D2C9", lw=.7, zorder=1)
    ax.set_yticks(s.y); ax.set_yticklabels(s.ch, fontsize=6.8)
    for tick, (_, r) in zip(ax.get_yticklabels(), s.iterrows()):
        if r.q < .05:
            tick.set_fontweight("bold"); tick.set_color(STIM)
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for roi in order:
        yy_ = s[s.ROI == roi].y
        ax.text(1.012, yy_.mean(), roi[:4], transform=tr, ha="left",
                va="center", fontsize=6.6, color="#6B7378")
    ax.invert_yaxis()
    xmax = float(max(s.mean_trt.max(), s.mean_sham.max()))
    xmin = float(min(s.mean_trt.min(), s.mean_sham.min()))
    ax.set_xlim(xmin - .6, xmax + 3.4)
    ax.set_xlabel("Coupling prevalence, stimulation window (%)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)


def traj_norm(ax, g, roi, I):
    """a/b: each recording normalised to its own before value (before = 0), then participant mean ± SEM.
    Between-condition during/after differences: participant-level Mann–Whitney."""
    from scipy import stats as _st
    t = g[(g.detector == DET) & (g.ROI == roi)]
    w = t.pivot_table(index=["participant", "session_id", "arm"], columns="win",
                      values="prevalence").reset_index()
    w = w.dropna(subset=["pre"])
    for win in ("stim", "post"):
        w[win] = (w[win] - w.pre) * 100
    w["pre"] = 0.0
    ax.axvspan(.5, 1.5, color=STIM, alpha=.07, lw=0, zorder=0)
    ax.axhline(0, color=C["ink2"], lw=.7, ls=(0, (3, 2)), zorder=1)
    per = {}
    for arm, c in (("Sham", NOST), ("Treatment", STIM)):
        pp = w[w.arm == arm].groupby("participant")[["pre", "stim", "post"]].mean()
        per[arm] = pp
        m = pp.mean().values; e = (pp.std(ddof=1) / np.sqrt(pp.notna().sum())).values
        ax.errorbar([0, 1, 2], m, yerr=e, fmt="none", ecolor=c, elinewidth=1.3,
                    capsize=3.0, capthick=1.1, alpha=.85, zorder=3)
        PS.line(ax, [0, 1, 2], m, c, lw=2.2, zorder=4)
        PS.dot(ax, [0, 1, 2], m, c, s=34, zorder=6)
    r = I[(I.detector == DET) & (I.ROI == roi) & (I.contrast == "stim vs pre")].iloc[0]
    sig = r.q < .05
    ax.text(.035, .975, (r"$\beta$ = %+.3f" % r.beta) + (r"  $\bigstar$" if sig else "")
            + "\n$q$ = %.3f" % r.q, transform=ax.transAxes, ha="left", va="top",
            fontsize=7.3, color=STIM if sig else C["ink2"],
            fontweight="bold" if sig else "normal", linespacing=1.5)
    pv = {}
    for win in ("stim", "post"):
        a = per["Treatment"][win].dropna(); b = per["Sham"][win].dropna()
        pv[win] = _st.mannwhitneyu(a, b).pvalue
    ax.text(.97, .80, "During $P$ = %.3f\nAfter $P$ = %.3f" % (pv["stim"], pv["post"]),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.4,
            color=C["ink2"], linespacing=1.5)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Before", "During", "After"])
    ax.set_xlim(-.38, 2.38)
    PS.title(ax, roi)


def main():
    PS.use()
    g = pd.read_csv(F1.OUT / "S_Table_session_metrics.csv")
    I = pd.read_csv(F1.OUT / "S_Table_arm_x_phase.csv")
    Cc = pd.read_csv(F1.TBL / "R1_channel_interaction.csv")
    Cc = Cc[Cc.detector == DET]
    B = pd.read_csv(F4.OUT / "S_Table_between_condition.csv")
    BS = B[(B.window == "stim") & (B.detector == DET)].set_index("ROI").reindex(ROIS)
    CB = F4.channel_between()

    f = plt.figure(figsize=(7.09, 10.6))
    outer = f.add_gridspec(4, 1, height_ratios=[1.0, 1.08, 1.0, 1.08],
                           hspace=.52, left=.085, right=.975, top=.962,
                           bottom=.052)
    r0 = outer[0].subgridspec(1, 3, width_ratios=[1, 1, 1.10], wspace=.44)
    r1 = outer[1].subgridspec(1, 3, width_ratios=[1, 1, 1.10], wspace=.20)
    r2 = outer[2].subgridspec(1, 4, width_ratios=[1, 1, .80, .80],
                              wspace=.62)
    r3 = outer[3].subgridspec(1, 2, width_ratios=[1, 2.35], wspace=.22)

    # ── a, b, c
    axa = f.add_subplot(r0[0]); traj_norm(axa, g, "Frontal", I)
    axa.set_ylabel("Coupling prevalence,\nchange from own baseline (pp)", linespacing=1.3)
    axb = f.add_subplot(r0[1], sharey=axa); traj_norm(axb, g, "Parietal", I)
    plt.setp(axb.get_yticklabels(), visible=False)
    lo, hi = axa.get_ylim(); axa.set_ylim(lo - (hi - lo) * .02, hi + (hi - lo) * .30)
    axc = f.add_subplot(r0[2]); F1.forest(axc, I)

    # ── d, e  interaction topographies
    vmax = float(np.nanpercentile(np.abs(Cc.beta.values), 97)) or .05
    axd = f.add_subplot(r1[0]); axe = f.add_subplot(r1[1])
    for ax, contr, lab, names in ((axd, "stim vs pre", "During − before", True),
                                  (axe, "post vs pre", "After − before", False)):
        t = Cc[Cc.contrast == contr]
        im = TP.topo(ax, dict(zip(t.ch, t.beta)),
                     {k + "_p": v for k, v in zip(t.ch, t.p)}, vmax, "",
                     show_names=names)
        ax.set_title(lab, pad=6)
    axl = f.add_subplot(r1[2]); axl.axis("off")
    f.canvas.draw()
    pr = axd.get_position(); y0, hh = pr.y0, pr.y1 - pr.y0
    ww = hh * 10.6 / 7.09
    L, GAP = .085, .040
    axd.set_position([L, y0, ww, hh]); axe.set_position([L + ww + GAP, y0, ww, hh])

    # ── f, g  between-condition participant distributions,  h  Hedges g,  i  event level
    tstim = g[(g.detector == DET) & (g.win == "stim")]
    rain = []
    for k, reg in enumerate(("Frontal", "Parietal")):
        ax = f.add_subplot(r2[k])
        rain.append(ax)
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
        print("  raincloud %-9s n=%d/%d  med %.1f vs %.1f  P=%.4f"
              % (reg, dat["Treatment"].size, dat["Sham"].size,
                 np.median(dat["Treatment"]), np.median(dat["Sham"]),
                 u.pvalue))
    axf, axg2 = rain
    axg = f.add_subplot(r2[2]); forest_g(axg, BS)
    axg.set_title("Between conditions", pad=8)
    axh = f.add_subplot(r2[3]); forest_or(axh, BS)
    axh.set_title("Event level", pad=8)
    axg.set_xlabel("Hedges' $g$")
    axh.set_xlabel("Odds ratio, per slow oscillation")

    # ── i  per-electrode g topography,  j  dumbbell
    axi = f.add_subplot(r3[0])
    vg = float(np.nanpercentile(np.abs(CB.g.values), 98))
    img = TP.topo(axi, dict(zip(CB.ch, CB.g)),
                  {k + "_p": v for k, v in zip(CB.ch, CB.q)}, vg, "",
                  show_names=False)
    axi.set_title("Electrode map", pad=6)
    axj = f.add_subplot(r3[1]); dumbbell(axj, CB)
    axj.set_xlabel("Coupling prevalence, stimulation window (%)")
    axj.set_title("Every electrode: no stimulation → stimulation"
                  "   (label = Hedges' $g$)", pad=6)

    f.canvas.draw()
    pdd = axd.get_position(); pee = axe.get_position()
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
             bbox_to_anchor=(x_leg, pee.y1 - .052), fontsize=7.0,
             handletextpad=.45, borderpad=0)
    ch = pdd.y1 - pdd.y0
    cax = f.add_axes([x_leg + .020, pdd.y0 + ch * .02, .018, ch * .46])
    cb = f.colorbar(im, cax=cax, ticks=np.round(np.linspace(-vmax, vmax, 5), 2))
    cb.set_label("Interaction " + chr(946), labelpad=5)
    cb.ax.tick_params(labelsize=6.8, length=2.6, width=.8, pad=2)
    cb.outline.set_linewidth(.8)

    # Colour bar for panel i (horizontal, below the head)
    pii = axi.get_position()
    cax2 = f.add_axes([pii.x0 + (pii.x1 - pii.x0) * .12, pii.y0 - .022,
                       (pii.x1 - pii.x0) * .76, .010])
    cb2 = f.colorbar(img, cax=cax2, orientation="horizontal",
                     ticks=np.round(np.linspace(-vg, vg, 5), 2))
    cb2.set_label("Hedges' $g$ per electrode", fontsize=6.6, labelpad=3)
    cb2.ax.tick_params(labelsize=6.2, length=2.4, pad=1.6)
    cb2.outline.set_linewidth(.8)

    # Condition legend below f · significance legend below g/h
    keys = [Line2D([], [], marker="o", ls="none", ms=5.4, markerfacecolor=STIM,
                   markeredgecolor=STIM, label="q < .05"),
            Line2D([], [], marker="o", ls="none", ms=5.4, markerfacecolor=NOMC,
                   markeredgecolor=NOMC, label="P < .05"),
            Line2D([], [], marker="o", ls="none", ms=5.4, markerfacecolor=NS_,
                   markeredgecolor=NS_, label="not significant")]
    pg, ph = axg.get_position(), axh.get_position()
    f.legend(handles=keys, loc="upper center",
             bbox_to_anchor=((pg.x0 + ph.x1) / 2, pg.y0 - .042), ncol=3,
             fontsize=6.4, handletextpad=.35, columnspacing=1.2, borderpad=0)

    for ax, lt in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d"), (axe, "e"),
                   (axf, "f"), (axg2, "g"), (axg, "h"), (axh, "i"),
                   (axi, "j"), (axj, "k")):
        PS.tag(f, ax, lt, dx=.052, dy=.016)

    out = FIG / "Fig1_merged"
    f.savefig(out.with_suffix(".png"), bbox_inches="tight")
    f.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(f)
    print("Fig1_merged (a–k)")


if __name__ == "__main__":
    main()
