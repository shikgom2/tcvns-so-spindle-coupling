# -*- coding: utf-8 -*-
"""Result 2 main figure — Staresina · N2 only.

a per-participant values (four regions, two conditions)
b Hedges g + CI for the four regions; significant cells in bold + ★
c event-level odds ratios
d per-electrode effect topography (yellow star at uncorrected P<.05)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe                               # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_paths as LP  # noqa: E402

warnings.filterwarnings("ignore")
import figures_topography as TP
import paperstyle as PS                                   # noqa: E402

_R = LP.PROJECT_ROOT
P = _R if (_R / "outputs/paper").exists() else _R / "이전"
OUT = P / "outputs/paper"
TBL = _R / "Final analysis/tables"
FIG = _R / "Final analysis/figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.linewidth": .7, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.major.width": .7,
    "ytick.major.width": .7, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "legend.frameon": False,
    "savefig.dpi": 400, "figure.dpi": 140,
})
STIM, NOST = "#1D6B8C", "#B4651A"
SIG, NS_ = PS.C["stim"], PS.C["ns"]
NOMC = "#B4651A"            # passes only before correction: orange
ROIS = ["Frontal", "Temporal", "Central", "Parietal"]
DET = "staresina_N2"
RNG = np.random.default_rng(20261003)


def fdr(p):
    p = np.asarray(p, float)
    o = np.full_like(p, np.nan)
    k = np.isfinite(p)
    m = int(k.sum())
    i = np.argsort(p[k])
    q = np.minimum.accumulate((p[k][i] * m / (np.arange(m) + 1))[::-1])[::-1]
    t = np.empty(m)
    t[i] = q.clip(0, 1)
    o[k] = t
    return o


def hedges(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                / (na + nb - 2))
    return (a.mean() - b.mean()) / s * (1 - 3 / (4 * (na + nb) - 9))


def channel_between():
    """Per-electrode between-group effect in the stimulation window (Hedges g + Mann-Whitney)."""
    g = pd.read_csv(OUT / "S_Table_channel_metrics.csv")
    t = g[(g.detector == DET) & (g.win == "stim")].dropna(
        subset=["prevalence"])
    rows = []
    for ch in TP.POS:
        s = t[t.ch == ch]
        a = s[s.arm == "Treatment"].groupby("participant").prevalence.mean()
        b = s[s.arm == "Sham"].groupby("participant").prevalence.mean()
        if len(a) < 10 or len(b) < 10:
            continue
        rows.append(dict(ch=ch, ROI=TP.ROI_OF[ch], n_trt=len(a),
                         n_sham=len(b), g=hedges(a.values, b.values),
                         mean_trt=float(a.mean() * 100),
                         mean_sham=float(b.mean() * 100),
                         p=float(stats.mannwhitneyu(a, b).pvalue)))
    C = pd.DataFrame(rows)
    C["q"] = fdr(C.p.values)
    C.to_csv(TBL / "R2_channel_between.csv", index=False,
             encoding="utf-8-sig")
    return C


def main():
    PS.use()
    B = pd.read_csv(OUT / "S_Table_between_condition.csv")
    BS = B[(B.window == "stim") & (B.detector == DET)].set_index(
        "ROI").reindex(ROIS)
    G = pd.read_csv(OUT / "S_Table_session_metrics.csv")
    C = channel_between()

    f = plt.figure(figsize=(9.6, 6.4))
    gs = f.add_gridspec(2, 3, width_ratios=[1.35, 1, 1],
                        height_ratios=[1, 1.02], wspace=.42, hspace=.48,
                        left=.075, right=.975, top=.905, bottom=.120)

    # a — per-participant values
    ax = ax_a = f.add_subplot(gs[0, 0])
    t = G[(G.detector == DET) & (G.win == "stim")]
    for k, (arm, lab, c) in enumerate((("Sham", "No stimulation", NOST),
                                       ("Treatment", "Stimulation", STIM))):
        xs, ms, los, his = [], [], [], []
        for i, roi in enumerate(ROIS):
            v = (t[(t.arm == arm) & (t.ROI == roi)]
                 .groupby("participant").prevalence.mean().dropna().values)
            x = i + (k - .5) * .30
            ax.scatter(x + RNG.uniform(-.075, .075, v.size), v * 100, s=3.5,
                       color=c, alpha=.30, lw=0, zorder=1)
            bs = RNG.choice(v, (4000, v.size), True).mean(1) * 100
            xs.append(x); ms.append(v.mean() * 100)
            los.append(np.percentile(bs, 2.5))
            his.append(np.percentile(bs, 97.5))
        ax.errorbar(xs, ms, yerr=[np.array(ms) - los, np.array(his) - ms],
                    fmt="none", ecolor=c, elinewidth=1.4, capsize=2.8,
                    capthick=1.1, zorder=3)
        PS.dot(ax, xs, ms, c, s=36, zorder=6)
        ax.plot([], [], "o", color=c, ms=5, mec="white", mew=.8, label=lab)
    ax.set_xticks(range(4)); ax.set_xticklabels(ROIS, fontsize=7.2)
    ax.set_ylabel("Coupling prevalence (%)")
    ax.set_ylim(0, 62)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.19), ncol=2,
              fontsize=7, handletextpad=.4, columnspacing=1.2)
    r = BS.loc["Parietal"]
    ytop = 44
    ax.plot([3 - .15, 3 - .15, 3 + .15, 3 + .15], [ytop, ytop + 1.6, ytop + 1.6, ytop],
            color=STIM, lw=.9, zorder=5)
    ax.text(3, ytop + 2.6, "$q$ = %.3f" % r.q_mwu, ha="center", va="bottom",
            fontsize=7, color=STIM, fontweight="bold")

    # b — Hedges g for the four regions
    ax = ax_b = f.add_subplot(gs[0, 1])
    y = np.arange(4)[::-1]
    for i, roi in enumerate(ROIS):
        r = BS.loc[roi]
        sig = r.q_mwu < .05
        nom = (not sig) and r.p_mwu < .05
        col = SIG if sig else (NOMC if nom else NS_)
        PS.grad_ci(ax, r.g_lo, r.g_hi, y[i], col, center=r.g, h=.30)
        ax.scatter([r.g], [y[i]], s=40, zorder=6, lw=1.2,
                   facecolor=col if (sig or nom) else "white", edgecolor=col)
        ax.text(1.72, y[i], "%+.2f%s" % (r.g, " ★" if sig else ""),
                fontsize=7, va="center", ha="right",
                color=col if sig else "#8C9296",
                fontweight="bold" if sig else "normal")
    ax.axvline(0, color=PS.C["ink"], lw=.9, ls=(0, (3, 2)))
    ax.set_yticks(y[::-1]); ax.set_yticklabels(ROIS[::-1], fontsize=7.2)
    for tick, roi in zip(ax.get_yticklabels(), ROIS[::-1]):
        if BS.loc[roi, "q_mwu"] < .05:
            tick.set_fontweight("bold"); tick.set_color(SIG)
    ax.set_ylim(-.6, 3.4); ax.set_xlim(-.45, 1.74)
    ax.set_xticks([0, .5, 1.0])
    ax.set_xlabel("Hedges' $g$")

    # c — event-level OR
    ax = ax_c = f.add_subplot(gs[0, 2])
    for i, roi in enumerate(ROIS):
        r = BS.loc[roi]
        sig = r.q_gee < .05 if np.isfinite(r.q_gee) else False
        nom = (not sig) and r.p_gee < .05
        col = SIG if sig else (NOMC if nom else NS_)
        PS.grad_ci(ax, r.OR_lo, r.OR_hi, y[i], col, center=r.OR, h=.30)
        # filled marker = passes correction, open marker = passes only before correction
        ax.scatter([r.OR], [y[i]], s=40, zorder=6, lw=1.2,
                   facecolor=col if (sig or nom) else "white", edgecolor=col)
        ax.text(2.42, y[i], "%.2f%s" % (r.OR, " ★" if sig else
                                        (" ☆" if nom else "")),
                fontsize=7, va="center", ha="right",
                color=col if (sig or nom) else "#8C9296",
                fontweight="bold" if sig else "normal")
    ax.axvline(1, color=PS.C["ink"], lw=.9, ls=(0, (3, 2)))
    ax.set_yticks(y[::-1]); ax.set_yticklabels([])
    ax.set_ylim(-.6, 3.4); ax.set_xlim(.70, 2.45)
    ax.set_xticks([1.0, 1.5, 2.0])
    ax.set_xlabel("Odds ratio, per slow oscillation")

    # shared symbol key for b and c — below panel c
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], marker="o", ls="none", ms=5.4,
                   markerfacecolor=SIG, markeredgecolor=SIG,
                   label="q < .05"),
            Line2D([], [], marker="o", ls="none", ms=5.4,
                   markerfacecolor=NOMC, markeredgecolor=NOMC,
                   label="P < .05"),
            Line2D([], [], marker="o", ls="none", ms=5.4,
                   markerfacecolor=NS_, markeredgecolor=NS_,
                   label="not significant")]
    pb = ax_b.get_position(); pc = ax.get_position()
    f.legend(handles=keys, loc="upper center",
             bbox_to_anchor=((pb.x0 + pc.x1) / 2, pc.y0 - .074),
             ncol=3, fontsize=6.3, handletextpad=.35, columnspacing=1.4,
             borderpad=0)

    # d — per-electrode topography
    ax = f.add_subplot(gs[1, 0])
    vmax = float(np.nanpercentile(np.abs(C.g.values), 98))
    im = TP.topo(ax, dict(zip(C.ch, C.g)),
                 {k + "_p": v for k, v in zip(C.ch, C.q)},
                 vmax, "", show_names=False)   # panel e shows all electrode names
    ax_d = ax

    # e — per-electrode dumbbell: no-stimulation mean -> stimulation mean, grouped by region, sorted by g
    ax = f.add_subplot(gs[1, 1:])
    order = ["Frontal", "Temporal", "Central", "Parietal"]
    s = C.copy(); s["rk"] = s.ROI.map({r: i for i, r in enumerate(order)})
    s = s.sort_values(["rk", "g"], ascending=[True, False]).reset_index(drop=True)
    y = 0; ys = []; seps = []
    for i, r in s.iterrows():
        if i and s.ROI[i] != s.ROI[i - 1]:
            y += 1; seps.append(y - .5)
        ys.append(y); y += 1
    s["y"] = ys
    for _, r in s.iterrows():
        sig = r.q < .05
        c = SIG if sig else NS_
        ax.plot([r.mean_sham, r.mean_trt], [r.y, r.y], color=c, lw=2.0,
                solid_capstyle="round", zorder=2, alpha=.9 if sig else .7)
        ax.scatter([r.mean_sham], [r.y], s=26, color=NOST, zorder=4, lw=.7,
                   edgecolor="white")
        ax.scatter([r.mean_trt], [r.y], s=26, color=STIM, zorder=4, lw=.7,
                   edgecolor="white")
        ax.text(max(r.mean_trt, r.mean_sham) + .35, r.y,
                "%+.2f%s" % (r.g, " ★" if sig else ""), va="center",
                fontsize=6.4, color=SIG if sig else "#8C9296",
                fontweight="bold" if sig else "normal")
    for sp_ in seps:
        ax.axhline(sp_, color="#D7D2C9", lw=.7, zorder=1)
    ax.set_yticks(s.y); ax.set_yticklabels(s.ch, fontsize=6.6)
    for tick, (_, r) in zip(ax.get_yticklabels(), s.iterrows()):
        if r.q < .05:
            tick.set_fontweight("bold"); tick.set_color(SIG)
    for roi in order:
        yy_ = s[s.ROI == roi].y
        ax.text(1.01, yy_.mean(), roi, transform=mtransforms.blended_transform_factory(ax.transAxes, ax.transData),
                ha="left", va="center", fontsize=6.6, color="#6B7378",
                rotation=90)
    ax.invert_yaxis()
    ax.set_xlabel("Coupling prevalence, stimulation window (%)   ·   label = Hedges' $g$")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    xmax = float(max(s.mean_trt.max(), s.mean_sham.max()))
    xmin = float(min(s.mean_trt.min(), s.mean_sham.min()))
    ax.set_xlim(xmin - .6, xmax + 2.6)

    pdp = ax_d.get_position()
    cax = f.add_axes([pdp.x0 + (pdp.x1 - pdp.x0) * .20, pdp.y0 - .062,
                      (pdp.x1 - pdp.x0) * .58, .014])
    cb = f.colorbar(im, cax=cax, orientation="horizontal",
                    ticks=np.round(np.linspace(-vmax, vmax, 5), 2))
    cb.set_label("Hedges' $g$ per electrode", fontsize=6.8, labelpad=3)
    cb.ax.tick_params(labelsize=6.2, length=2.6, pad=1.6)

    # place panel letters in figure coordinates (so they do not overlap axis labels)
    for a_, lt in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d"),
                   (ax, "e")):
        pp = a_.get_position()
        f.text(max(pp.x0 - .050, .004), pp.y1 + .014, lt, fontsize=11,
               fontweight="bold", va="bottom", ha="left")

    for a_ in (ax_a, ax_b, ax_c, ax):
        PS.panel(a_); a_.grid(False); a_.set_axisbelow(True)
    f.savefig(FIG / "Fig2_between.png", bbox_inches="tight")
    f.savefig(FIG / "Fig2_between.pdf", bbox_inches="tight")
    plt.close(f)
    print("Fig2_between  (Staresina N2 only)")
    print(C.sort_values("p")[["ch", "ROI", "g", "p", "q"]].round(4)
          .to_string(index=False))


if __name__ == "__main__":
    main()
