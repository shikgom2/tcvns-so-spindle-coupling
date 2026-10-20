# -*- coding: utf-8 -*-
"""Reference-paper-style figure — raincloud + SO-locked event histogram + rose plot.

a,b  coupling-prevalence raincloud per region (half-violin + box + points), condition comparison
c–f  histogram of spindle-peak probability relative to the SO trough, with the mean SO waveform overlaid
g–j  coupling-phase rose plot + mean vector

Usage: python figures_raincloud.py [strict|loose]
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
from scipy import stats                                           # noqa: E402

warnings.filterwarnings("ignore")
_R = Path(r"D:/SO-Spindle논문")
P = _R if (_R / "outputs/paper").exists() else _R / "이전"
OUT = P / "outputs/paper"
TBL = _R / "Final analysis/tables"
FIG = _R / "Final analysis/figures"
ARGS = [a for a in sys.argv[1:] if a != "norain"]
TAG = ARGS[0] if ARGS else "strict"
# norain: drop a,b (between-condition raincloud), which move to Figure 2
NORAIN = "norain" in sys.argv

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.linewidth": .7, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.major.width": .7,
    "ytick.major.width": .7, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "legend.frameon": False,
    "savefig.dpi": 400, "figure.dpi": 140,
})
import paperstyle as PS
STIM, NOST = PS.C["stim"], PS.C["nost"]
COL = {"Treatment": STIM, "Sham": NOST}
# clouds filled light with only the outline dark — colour scheme of the reference figure
FILL = {"Treatment": "#86B8CE", "Sham": "#EBB184"}
LAB = {"Treatment": "Stimulation", "Sham": "No stimulation"}
REG = [("frontal", "Frontal"), ("parietal", "Parietal")]
DET = "staresina_N2"
RNG = np.random.default_rng(20261003)


def half_violin(ax, v, x, edge, fill, side=-1, width=.34):
    """Half violin — light fill + dark outline."""
    k = stats.gaussian_kde(v)
    rng = float(v.max() - v.min())
    yy = np.linspace(v.min() - rng * .08, v.max() + rng * .08, 220)
    dd = k(yy)
    dd = dd / dd.max() * width
    ax.fill_betweenx(yy, x, x + side * dd, color=fill, alpha=.92, lw=0,
                     zorder=1)
    ax.plot(x + side * dd, yy, color=edge, lw=1.0, zorder=2)


def bracket(ax, x0, x1, y, txt, color="#44495C"):
    """Significance bracket — above the two conditions, as in the reference figure."""
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * .028
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color=color, lw=.8,
            clip_on=False, zorder=6)
    ax.text((x0 + x1) / 2, y + h * 1.15, txt, ha="center", va="bottom",
            fontsize=7.2, color=color, clip_on=False, zorder=6)


def raincloud(ax, data, title, ylab, ptxt=None):
    """Raincloud per condition — cloud + box + points."""
    top = 0.0
    for i, arm in enumerate(("Treatment", "Sham")):
        v = np.asarray(data[arm], float)
        v = v[np.isfinite(v)]
        top = max(top, float(v.max()))
        half_violin(ax, v, i - .06, COL[arm], FILL[arm], side=-1)
        ax.boxplot([v], positions=[i + .17], widths=.17,
                   showfliers=False, patch_artist=True,
                   medianprops=dict(color="#1A1D26", lw=1.2),
                   boxprops=dict(facecolor="white", edgecolor="#2B3138",
                                 lw=.9),
                   whiskerprops=dict(color="#2B3138", lw=.9),
                   capprops=dict(color="#2B3138", lw=.9))
        ax.scatter(i + .40 + RNG.uniform(-.055, .055, v.size), v, s=6.0,
                   color=COL[arm], alpha=.60, lw=0, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([LAB["Treatment"], LAB["Sham"]], fontsize=7.4)
    ax.set_xlim(-.55, 1.72)
    ax.set_ylim(0, top * 1.26)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=8.6, pad=16)
    if ptxt:
        bracket(ax, .05, 1.05, top * 1.07, ptxt)


def main():
    PS.use()
    d = np.load(TBL / ("SO_event_coupling_%s.npz" % TAG))
    g = pd.read_csv(OUT / "S_Table_session_metrics.csv")
    t = g[(g.detector == DET) & (g.win == "stim")]
    ctr = d["centres"]

    f = plt.figure(figsize=(9.8, 5.7 if NORAIN else 8.0))
    nrow = 2 if NORAIN else 3
    gs = f.add_gridspec(nrow, 4,
                        height_ratios=[1, .98] if NORAIN else [1, 1, .98],
                        wspace=.46, hspace=.70, left=.085, right=.975,
                        top=.935, bottom=.090)
    R0 = 0 if NORAIN else 1
    axes = []

    if not NORAIN:
        # a, b — coupling-prevalence raincloud
        for k, (reg, Reg) in enumerate(REG):
            ax = f.add_subplot(gs[0, k * 2:(k + 1) * 2])
            axes.append(ax)
            dat = {}
            for arm in ("Treatment", "Sham"):
                dat[arm] = (t[(t.ROI == Reg) & (t.arm == arm)]
                            .groupby("participant").prevalence.mean()
                            .dropna().values * 100)
            u = stats.mannwhitneyu(dat["Treatment"], dat["Sham"])
            raincloud(ax, dat, "%s  ·  coupling prevalence" % Reg,
                      "Coupling (%)",
                      ptxt=("$P$ < .001" if u.pvalue < .001
                            else "$P$ = %.3f" % u.pvalue))

    # c–f — event histogram + mean SO waveform
    for k, (reg, Reg) in enumerate(REG):
        for j, arm in enumerate(("Treatment", "Sham")):
            ax = f.add_subplot(gs[R0, k * 2 + j])
            axes.append(ax)
            h = d["%s_%s_hist" % (reg, arm)]
            if not len(h):
                continue
            m = h.mean(0)
            e = h.std(0, ddof=1) / np.sqrt(h.shape[0])
            for cx, mv_ in zip(ctr, m):
                PS.grad_bar(ax, cx, mv_, COL[arm], w=.099, radius=.010)
            ax.errorbar(ctr, m, yerr=e, fmt="none", ecolor="#44495C",
                        lw=.5, capsize=1.1)
            ax.axhline(m.mean(), color=PS.C["ink"], lw=.7, ls=(0, (3, 2)))
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Time from SO trough (s)", fontsize=7)
            if j == 0:
                ax.set_ylabel("$P$(spindle peak)")
            ax.set_title("%s · %s" % (Reg, LAB[arm]), fontsize=8.0, pad=14)
            # overlay the mean SO waveform
            w = np.load(TBL / "SO_locked_regions.npz")
            key = "%s_%s_wave" % (reg, arm)
            if key in w:
                tw, wv = w["t"], w[key].mean(0)
                ax2 = ax.twinx()
                PS.line(ax2, tw, wv, "#5A1B1B", lw=1.5, zorder=5)
                ax2.set_ylim(wv.min() * 2.6, wv.max() * 1.25)
                ax2.set_yticks([])
                ax2.spines["right"].set_visible(False)

    # g–j — phase rose plot
    for k, (reg, Reg) in enumerate(REG):
        for j, arm in enumerate(("Treatment", "Sham")):
            ax = f.add_subplot(gs[R0 + 1, k * 2 + j], projection="polar")
            axes.append(ax)
            ph = d["%s_%s_ph" % (reg, arm)]
            if not len(ph):
                continue
            nb = 24
            edges = np.linspace(-np.pi, np.pi, nb + 1)
            cnt, _ = np.histogram(ph, bins=edges)
            cnt = cnt / cnt.sum()
            ax.bar(edges[:-1] + np.pi / nb, cnt, width=2 * np.pi / nb,
                   color=COL[arm], alpha=.82, lw=.4, edgecolor="white")
            mv = np.exp(1j * ph).mean()
            ax.annotate("", xy=(np.angle(mv), np.abs(mv) * cnt.max() * 3.2),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", lw=1.8,
                                        color="#1A1D26"))
            ax.set_theta_zero_location("E")
            ax.set_yticklabels([])
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
            ax.set_xticklabels(["0", "π/2", "π", "3π/2"], fontsize=6.4)
            ax.tick_params(pad=-2)
            ax.grid(color="#D7D2C9", lw=.5)
            pp = ax.get_position()
            f.text((pp.x0 + pp.x1) / 2, pp.y0 - .030,
                   "%s · %s\npooled MRL %.2f at %+.0f°"
                   % (Reg, LAB[arm], np.abs(mv),
                      np.degrees(np.angle(mv))),
                   ha="center", va="top", fontsize=7.2, color="#44495C")

    for ax, lt in zip(axes, "abcdefgh" if NORAIN else "abcdefghij"):
        p = ax.get_position()
        f.text(max(p.x0 - .050, .002), p.y1 + .018, lt, fontsize=11,
               fontweight="bold", va="bottom", ha="left")

    nrec = {arm: int(d["%s_%s_hist" % ("frontal", arm)].shape[0])
            for arm in ("Treatment", "Sham")}
    for a_ in axes:
        if a_.name != "polar":
            PS.panel(a_); a_.grid(False); a_.set_axisbelow(True)
    out = FIG / ("FigR_coupling_eventlevel_%s%s.png"
             % (TAG, "_norain" if NORAIN else ""))
    f.savefig(out, bbox_inches="tight")
    f.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(f)
    print("%s   recordings  stim %d · no-stim %d"
          % (out.name, nrec["Treatment"], nrec["Sham"]))
    for reg, Reg in REG:
        for arm in ("Treatment", "Sham"):
            mr = d["%s_%s_mrl" % (reg, arm)]
            print("  %-9s %-10s n=%3d  MRL %.3f ± %.3f"
                  % (Reg, arm, len(mr), mr.mean() if len(mr) else np.nan,
                     mr.std(ddof=1) if len(mr) > 1 else np.nan))


if __name__ == "__main__":
    main()
