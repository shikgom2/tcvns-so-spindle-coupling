# -*- coding: utf-8 -*-
"""Figure 8 — which slow oscillations recruit spindles, and in which direction does stimulation push them.

a  coupling probability by amplitude quintile           (gradient area + glossy dots)
b  slope of coupled vs uncoupled slow oscillations      (half violin + dots + interval)
c  coupling probability by spatial extent               (gradient area + interval band)
d  condition x window interaction for 6 measures        (gradient bands instead of bars)

Detector Staresina (0.30-1.25 Hz, 0.8-2.0 s, mean + 1 SD), stages N2+N3.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt                                   # noqa: E402
import matplotlib.transforms as mtransforms                       # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from scipy import stats                                           # noqa: E402

warnings.filterwarnings("ignore")
import paperstyle as PS                                           # noqa: E402

R = Path(r"D:/SO-Spindle논문")
TBL = R / "Final analysis/tables"
FIG = R / "Final analysis/figures"
RNG = np.random.default_rng(20261003)
ROIS, C, ROI = PS.ROIS, PS.C, PS.ROI

ORDER = ["Slow-oscillation amplitude (uV)",
         "Slope, uncoupled waves (uV/s)",
         "Slope, coupled waves (uV/s)",
         "Amplitude, coupled - uncoupled (uV)",
         "Fraction of widespread waves",
         "Derivations involved per wave"]
SHORT = {"Slow-oscillation amplitude (uV)": "SO amplitude",
         "Slope, uncoupled waves (uV/s)": "SO slope, uncoupled",
         "Slope, coupled waves (uV/s)": "SO slope, coupled",
         "Amplitude, coupled - uncoupled (uV)": "Amplitude gap",
         "Fraction of widespread waves": "Fraction widespread",
         "Derivations involved per wave": "Derivations involved"}


def ci(v, n=4000):
    v = np.asarray(v, float)
    return np.percentile(RNG.choice(v, (n, v.size), True).mean(1),
                         [2.5, 97.5])


# --------------------------------------------------------------------- a
def panel_amp(ax):
    D = pd.read_csv(TBL / "M1_amp_curve.csv")
    D = D[D.ROI.isin(ROIS)]
    allm = D.groupby("quintile").prev.mean() * 100
    ax.set_xlim(.62, 5.38)
    # per-region lines faint in the background; grand mean in front with a bootstrap 95 % band
    for roi in ROIS:
        t = D[D.ROI == roi].groupby("quintile").prev.mean() * 100
        ax.plot(t.index, t.values, color=ROI[roi], lw=1.3, alpha=.45,
                label=roi, zorder=3, solid_capstyle="round")
    lo, hi = [], []
    for q, g in D.groupby("quintile"):
        v = g.prev.values * 100
        bs = RNG.choice(v, (3000, v.size), True).mean(1)
        lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
    GOLD = "#D9A520"                      # grand mean: pale yellow-gold
    ax.fill_between(allm.index, lo, hi, color="#F2D98A", alpha=.45, lw=0,
                    zorder=4)
    PS.line(ax, allm.index, allm.values, GOLD, lw=2.4, zorder=6)
    ax.scatter(allm.index, allm.values, s=22, color=GOLD, alpha=.55,
               zorder=7, lw=.6, edgecolor="white")
    ax.set_ylim(allm.min() - 3.2, allm.max() + 2.4)
    rr = (D.groupby(["session", "ROI"])
          .apply(lambda t: stats.spearmanr(t.quintile, t.prev).statistic
                 if t.quintile.nunique() > 2 else np.nan).dropna())
    pv = float(stats.wilcoxon(rr.values).pvalue)
    ax.text(.035, .965, "median $\\rho$ = %+.2f\n%s   (%d cells)"
            % (np.median(rr.values),
               "$P$ < .001" if pv < .001 else "$P$ = %.3f" % pv, len(rr)),
            transform=ax.transAxes, va="top", ha="left", fontsize=6.9,
            color=C["ink"], linespacing=1.45)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Slow-oscillation amplitude  (within-recording quintile)")
    ax.set_ylabel("Coupling probability (%)")
    PS.title(ax, "Bigger waves couple more often")


# --------------------------------------------------------------------- b
def panel_slope(ax):
    A = pd.read_csv(TBL / "M1_amp_gating.csv")
    A = A[A.ROI.isin(ROIS)].dropna(subset=["slope_gap"])
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, roi in enumerate(ROIS):
        v = A[A.ROI == roi].slope_gap.values
        PS.halfviolin(ax, v, i + .10, ROI[roi], width=.40, side=1)
        ax.scatter(i - .14 + RNG.uniform(-.10, .10, v.size), v, s=4.6,
                   color=ROI[roi], alpha=.34, lw=0, zorder=2)
        lo, hi = ci(v)
        ax.plot([i - .14, i - .14], [lo, hi], color=ROI[roi], lw=2.6,
                zorder=4, solid_capstyle="round")
        PS.dot(ax, [i - .14], [v.mean()], ROI[roi], s=34, zorder=6)
        p = float(stats.wilcoxon(v).pvalue)
        ax.text(i, 1.012, "%+.1f\n%s" % (v.mean(), "$P$ < .001" if p < .001
                                         else "$P$ = %.3f" % p),
                transform=tr, ha="center", va="bottom", fontsize=6.5,
                color=ROI[roi], linespacing=1.45, clip_on=False)
    ax.axhline(0, color=C["ink"], lw=.9, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels([r[:4] for r in ROIS])
    ax.set_xlim(-.62, 3.62)
    ax.set_ylabel("Slope difference,  coupled − uncoupled\n(µV s$^{-1}$)",
                  linespacing=1.5)
    PS.title(ax, "Coupled waves rise more steeply", pad=22)


# --------------------------------------------------------------------- c
def panel_extent(ax):
    T = pd.read_csv(TBL / "T_M2_coupling_by_extent.csv")
    x = T.Derivations.values.astype(float)
    y = T.Coupling_pct.values
    lo = np.array([float(s.split(" to ")[0]) for s in T.CI])
    hi = np.array([float(s.split(" to ")[1]) for s in T.CI])
    ax.set_xlim(x.min() - .45, x.max() + .45)
    ax.set_ylim(lo.min() - 1.6, hi.max() + 1.2)
    PS.grad_area(ax, x, y, C["stim"], base=ax.get_ylim()[0], alpha=.46)
    ax.fill_between(x, lo, hi, color=C["stim"], alpha=.20, lw=0, zorder=3)
    PS.line(ax, x, y, C["stim"], lw=2.4, zorder=5)
    PS.dot(ax, x, y, C["stim"], s=30, zorder=6)
    rho, pv = stats.spearmanr(x, y)
    ax.text(.035, .965, r"$\rho$ = %+.2f,  " % rho
            + ("$P$ < .001" if pv < .001 else "$P$ = %.3f" % pv),
            transform=ax.transAxes, va="top", ha="left", fontsize=6.9,
            color=C["ink"])
    ax.annotate("%.1f %%" % y[-1], xy=(x[-1], y[-1]), xytext=(-2, 9),
                textcoords="offset points", ha="right", fontsize=6.8,
                color=C["stim"], fontweight="bold")
    ax.annotate("%.1f %%" % y[0], xy=(x[0], y[0]), xytext=(2, -11),
                textcoords="offset points", ha="left", fontsize=6.8,
                color=C["stim"], fontweight="bold")
    ax.set_xlabel("Derivations showing the same slow oscillation\n"
                  "(within ± 400 ms)", linespacing=1.5)
    ax.set_ylabel("Coupling probability (%)")
    PS.title(ax, "Widespread waves couple more often")


# --------------------------------------------------------------------- d
def panel_effects(ax, stars=True, p_x=6.2, xlim=(-4.6, 6.3)):
    T = pd.read_csv(TBL / "T_M3_stimulation_effects.csv")
    T = T[T.Contrast == "post"]
    yy = np.arange(len(ORDER))[::-1]
    ax.set_xlim(*xlim)
    ax.set_ylim(-.70, len(ORDER) - .18)
    for i, m in enumerate(ORDER):
        g = T[T.Measure == m]
        pool = g[g.Region == "All four"].iloc[0]
        col = C["up"] if pool.beta > 0 else C["down"]
        zp = pool.beta / pool.se
        PS.grad_barh(ax, min(0, zp - 1.96), max(0, zp + 1.96), yy[i], col,
                     h=.30, flip=zp < 0)
        reg = g[g.Region != "All four"]
        z = (reg.beta / reg.se).values
        ax.scatter(z, yy[i] + RNG.uniform(-.055, .055, z.size), s=11,
                   color="white", alpha=.90, lw=.5, edgecolor=col,
                   zorder=5)
        if stars:
            for _, r in reg[reg.P < .05].iterrows():
                ax.scatter([r.beta / r.se], [yy[i]], s=58, marker="*",
                           color=C["star"], edgecolor=C["ink"], lw=.5,
                           zorder=7)
        PS.dot(ax, [zp], [yy[i]], col, s=36, zorder=6)
        ax.text(p_x, yy[i], "$P$ = %.3f" % pool.P, ha="right", va="center",
                fontsize=6.4, color=col if pool.P < .10 else C["ink2"],
                fontweight="bold" if pool.P < .10 else "normal")
    ax.axvline(0, color=C["ink"], lw=1.0, zorder=4)
    ax.set_yticks(yy[::-1])
    ax.set_yticklabels([SHORT[m] for m in ORDER][::-1], fontsize=7.0)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xlabel("Condition × window interaction, after versus before\n"
                  "(estimate ÷ standard error)", linespacing=1.5)
    PS.title(ax, "Stimulation shifts the population in two directions")
    ax.text(-4.45, len(ORDER) - .44, "smaller, shallower", fontsize=6.4,
            color=C["down"], ha="left", va="center", style="italic")
    ax.text(4.5, len(ORDER) - .44, "more widespread", fontsize=6.4,
            color=C["up"], ha="right", va="center", style="italic")


def main():
    PS.use()
    f = plt.figure(figsize=(7.09, 6.0))
    gs = f.add_gridspec(2, 2, wspace=.44, hspace=.92, left=.102,
                        right=.972, top=.868, bottom=.098)
    axa = f.add_subplot(gs[0, 0]); panel_amp(axa)
    axb = f.add_subplot(gs[0, 1]); panel_slope(axb)
    axc = f.add_subplot(gs[1, 0]); panel_extent(axc)
    axd = f.add_subplot(gs[1, 1]); panel_effects(axd)
    for ax in (axa, axb, axc, axd):
        PS.panel(ax)
        ax.grid(False)
        ax.set_axisbelow(True)

    f.canvas.draw()
    pa, pb = axa.get_position(), axb.get_position()
    keys = [Line2D([], [], color=ROI[r], lw=2.2, label=r) for r in ROIS]
    keys.append(Line2D([], [], marker="*", ls="none", ms=8,
                       markerfacecolor=C["star"], markeredgecolor=C["ink"],
                       markeredgewidth=.5, label="region at $P$ < .05"))
    f.legend(handles=keys, loc="upper center",
             bbox_to_anchor=((pa.x0 + pb.x1) / 2, pa.y1 + .122), ncol=5,
             fontsize=6.9, handlelength=1.5, handletextpad=.45,
             columnspacing=1.5, borderpad=0)
    for ax, lt in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")):
        PS.tag(f, ax, lt, dy=.064 if ax in (axa, axb) else .032)

    f.savefig(FIG / "FigM_mechanism.png", bbox_inches="tight")
    f.savefig(FIG / "FigM_mechanism.pdf", bbox_inches="tight")
    plt.close(f)
    print("FigM_mechanism  (restyled)")


if __name__ == "__main__":
    main()
