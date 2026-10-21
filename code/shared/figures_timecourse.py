# -*- coding: utf-8 -*-
"""Stimulation — coupling — spatial extent on a single time axis.

a  coupling prevalence aligned to stimulation onset (5-min bins, two conditions)
b  slow-oscillation spatial extent on the same axis (number of electrodes showing it simultaneously)
c  within-recording paired comparison: before → during → after stimulation, coupling and spatial extent, two conditions
d  within-recording correlation between the coupling and spatial-extent time series (lag -1, 0, +1)

Inputs: M7_time_course_bins.csv, M7_lag_correlation.csv (mech_time_course.py)
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

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_paths as LP  # noqa: E402
import paperstyle as PS                                           # noqa: E402

R = LP.PROJECT_ROOT
TBL = R / "Final analysis/tables"
FIG = R / "Final analysis/figures"
STIM, NOST, INK, INK2 = PS.C["stim"], PS.C["nost"], PS.C["ink"], PS.C["ink2"]
RNG = np.random.default_rng(20261003)
STIM_END = 20.0            # median stimulation-window length (min) — matches R4_stim_span


def series(ax, B, y, ylab, title, scale=1.0, baseline=False, robust=False,
           min_n=8):
    """If baseline=True, subtract each recording's pre-stimulation (bin<0) mean before averaging.
    If robust=True, drop values with MAD-based |z| > 3.5 within each bin×condition (outliers)."""
    if baseline:
        pre = B[B.bin < 0].groupby("file")[y].mean().rename("pre")
        B = B.merge(pre, on="file", how="inner")
        B = B.assign(**{y: B[y] - B.pre})
        ax.axhline(0, color=INK, lw=.7, ls=":", zorder=1)
    if robust:
        def _keep(v):
            med = v.median(); mad = (v - med).abs().median() * 1.4826
            return (v - med).abs() <= 3.5 * mad if mad > 0 else v == v
        keep = B.groupby(["arm", "bin"])[y].transform(_keep)
        series.n_removed = int((~keep).sum())
        B = B[keep]
    for arm, c, lab in (("Sham", NOST, "No stimulation"),
                        ("Treatment", STIM, "Stimulation")):
        g = B[B.arm == arm].groupby("bin")[y]
        m, se, n = g.mean() * scale, g.sem() * scale, g.size()
        k = n >= min_n
        ax.errorbar(m.index[k], m[k], yerr=se[k], fmt="o-", ms=3.8, lw=1.4,
                    capsize=2.2, color=c, mec="white", mew=.6, label=lab,
                    zorder=3)
    ax.axvspan(0, STIM_END, color=STIM, alpha=.07, lw=0, zorder=0)
    ax.axvline(0, color=INK, lw=.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(-15, 35)
    ax.set_xlabel("Time from stimulation onset (min)")
    ax.set_ylabel(ylab)
    ax.set_title(title, pad=8)


def paired(ax, B, y, ylab, title, scale=1.0):
    """Within recording: pre → during → post. Mean trajectory per condition + faint lines for individual recordings."""
    Bb = B.copy()
    Bb["phase"] = np.where(Bb.bin < 0, 0, np.where(Bb.bin < STIM_END, 1, 2))
    W = (Bb.groupby(["file", "arm", "phase"])[y].mean().unstack("phase")
         .dropna()) * scale
    out = {}
    for arm, c, dx in (("Sham", NOST, .12), ("Treatment", STIM, -.12)):
        w = W.xs(arm, level="arm")
        if len(w) < 5:
            continue
        for _, r in w.iterrows():
            ax.plot(np.arange(3) + dx, r.values, color=c, alpha=.10, lw=.6,
                    zorder=1)
        m = w.mean().values
        ci = np.array([np.percentile(RNG.choice(w[p].values,
                                                 (3000, len(w)), True).mean(1),
                                      [2.5, 97.5]) for p in (0, 1, 2)])
        ax.errorbar(np.arange(3) + dx, m, yerr=[m - ci[:, 0], ci[:, 1] - m],
                    fmt="o-", ms=5, lw=2.0, capsize=2.6, color=c, mec="white",
                    mew=.8, zorder=4)
        p1 = stats.wilcoxon(w[1] - w[0]).pvalue
        p2 = stats.wilcoxon(w[2] - w[0]).pvalue
        out[arm] = (len(w), w[1].mean() - w[0].mean(), p1,
                    w[2].mean() - w[0].mean(), p2)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Before", "During", "After"])
    ax.set_xlim(-.45, 2.45)
    ax.axvspan(.5, 1.5, color=STIM, alpha=.07, lw=0, zorder=0)
    ax.set_ylabel(ylab)
    ax.set_title(title, pad=8)
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    lines = []
    for arm, c in (("Treatment", STIM), ("Sham", NOST)):
        if arm in out:
            n, d1, p1, d2, p2 = out[arm]
            fp = lambda v: "< 0.001" if v < .001 else "= %.3f" % v
            lines.append((c, "%s  during %+.2f ($P$ %s) · after %+.2f "
                             "($P$ %s)  n = %d"
                          % ("Stim." if arm == "Treatment" else "No stim.",
                             d1, fp(p1), d2, fp(p2), n)))
    for i, (c, s) in enumerate(lines):
        ax.text(.02, .975 - .085 * i, s, transform=tr, ha="left", va="top",
                fontsize=5.9, color=c)
    return out


def lagpanel(ax, L):
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (arm, c, lab) in enumerate((("Treatment", STIM, "Stimulation"),
                                       ("Sham", NOST, "No stimulation"))):
        for j, lag in enumerate((-1, 0, 1)):
            v = L[(L.arm == arm) & (L.lag == lag)].r.dropna().values
            if len(v) < 5:
                continue
            x = j + (i - .5) * .32
            ax.scatter(x + RNG.uniform(-.07, .07, v.size), v, s=5, color=c,
                       alpha=.35, lw=0, zorder=2)
            lo, hi = np.percentile(RNG.choice(v, (3000, v.size), True).mean(1),
                                   [2.5, 97.5])
            ax.plot([x, x], [lo, hi], color=c, lw=2.0, zorder=4,
                    solid_capstyle="round")
            ax.scatter([x], [v.mean()], s=28, color=c, zorder=5, lw=.7,
                       edgecolor="white", label=lab if j == 0 else None)
            p = stats.wilcoxon(v).pvalue
            ax.text(x, 1.01 + .07 * i, "$P$ = %.2f" % p, transform=tr,
                    ha="center", va="bottom", fontsize=5.9, color=c,
                    clip_on=False)
    ax.axhline(0, color=INK, lw=.7, ls=(0, (3, 2)))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["extent leads\n(lag −1)", "same bin\n(lag 0)",
                        "coupling leads\n(lag +1)"], fontsize=6.6,
                       linespacing=1.4)
    ax.set_xlim(-.6, 2.6)
    ax.set_ylabel("Spearman $r$ within recording\n(coupling × spatial extent)",
                  linespacing=1.4)
    ax.set_title("Do the two time series move together?", pad=20)


def main():
    PS.use()
    B = pd.read_csv(TBL / "M7_time_course_bins.csv")
    L = pd.read_csv(TBL / "M7_lag_correlation.csv")

    f = plt.figure(figsize=(7.09, 5.6))
    gs = f.add_gridspec(2, 2, wspace=.40, hspace=.66, left=.10, right=.975,
                        top=.90, bottom=.10)
    axa = f.add_subplot(gs[0, 0])
    series(axa, B, "prev", "Coupling prevalence (%)",
           "Coupling, aligned to stimulation onset", scale=100)
    axb = f.add_subplot(gs[0, 1])
    series(axb, B, "extent", "Derivations per slow oscillation",
           "Spatial extent, aligned to stimulation onset")
    axc = f.add_subplot(gs[1, 0])
    oc = paired(axc, B, "prev", "Coupling prevalence (%)",
                "Within recording: coupling", scale=100)
    axd = f.add_subplot(gs[1, 1])
    oe = paired(axd, B, "extent", "Derivations per slow oscillation",
                "Within recording: spatial extent")
    axc.set_ylim(0, 32); axd.set_ylim(1.2, 4.0)

    h, l = axa.get_legend_handles_labels()
    pa = axa.get_position(); pb = axb.get_position()
    f.legend(h, l, loc="upper center", ncol=2,
             bbox_to_anchor=((pa.x0 + pb.x1) / 2, pa.y1 + .085),
             fontsize=7, handletextpad=.5, columnspacing=1.8)
    for ax, lt in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")):
        PS.tag(f, ax, lt, dx=.062, dy=.030)
    f.savefig(FIG / "FigM_timecourse.png", bbox_inches="tight")
    f.savefig(FIG / "FigM_timecourse.pdf", bbox_inches="tight")
    plt.close(f)

    # lag correlation in a separate figure (supplementary)
    f2, ax = plt.subplots(figsize=(3.6, 2.8),
                          gridspec_kw=dict(left=.20, right=.97, top=.80,
                                           bottom=.24))
    lagpanel(ax, L)
    ax.legend(loc="lower right", fontsize=6.2, handletextpad=.4)
    f2.savefig(FIG / "FigS_timecourse_lag.png", bbox_inches="tight")
    f2.savefig(FIG / "FigS_timecourse_lag.pdf", bbox_inches="tight")
    plt.close(f2)
    print("FigM_timecourse · FigS_timecourse_lag")
    print("coupling", oc); print("extent", oe)


if __name__ == "__main__":
    main()
