# -*- coding: utf-8 -*-
"""Supplementary Figure 6 — slow-wave activity (SWA) does not differ between conditions.

a-c  within-night change (during stimulation − before), both conditions side by side, three bands
d    absolute power in the stimulation window, both conditions
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

R = LP.PROJECT_ROOT / "Final analysis"
OUT = R / "manuscript/figures/v14oct2026/Supplementary_Figure_6_slow_wave_activity"
T = pd.read_csv(R / "tables/SWA_windows.csv")
REGS = ["Frontal", "Parietal"]
KEY = {"Frontal": "fron", "Temporal": "temp", "Central": "cent", "Parietal": "pari"}
STIM, NOST, INK, INK2 = PS.C["stim"], PS.C["nost"], PS.C["ink"], PS.C["ink2"]
RNG = np.random.default_rng(20261003)


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty_like(p)
    m = len(p); prev = 1.0
    for rank, i in enumerate(o[::-1], 1):
        prev = min(prev, p[i] * m / (m - rank + 1)); q[i] = prev
    return q


def change(band, reg, arm):
    col = "%s_%s" % (KEY[reg], band)
    d = T[T.arm == arm].dropna(subset=[col])
    w = d.groupby(["participant", "win"])[col].mean().unstack("win")
    w = w.dropna(subset=["stim", "pre"])
    return ((np.log(w["stim"]) - np.log(w["pre"])) * 100).values


def level(band, reg, arm):
    col = "%s_%s" % (KEY[reg], band)
    d = T[(T.arm == arm) & (T.win == "stim")].dropna(subset=[col])
    return d.groupby("participant")[col].mean().dropna().values


def panel_change(ax, band, title):
    pw = {}
    for arm in ("Treatment", "Sham"):
        pw[arm] = [change(band, r, arm) for r in REGS]
    qs = bh([stats.wilcoxon(v).pvalue for v in pw["Treatment"]])
    for k, reg in enumerate(REGS):
        tops = {}
        for j, (arm, c) in enumerate((("Treatment", STIM), ("Sham", NOST))):
            v = pw[arm][k]
            x = k + (-.19 if j == 0 else .19)
            bs = RNG.choice(v, (4000, v.size), True).mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            tops[arm] = hi
            ax.bar([x], [v.mean()], width=.33, color=c, alpha=.85, lw=0, zorder=2)
            ax.plot([x, x], [lo, hi], color=INK, lw=1.2, zorder=4,
                    solid_capstyle="round")
        if qs[k] < .05:
            ax.text(k - .19, tops["Treatment"] + 1.5, "∗", ha="center",
                    va="bottom", fontsize=9, color=STIM, fontweight="bold")
        pb = stats.mannwhitneyu(pw["Treatment"][k], pw["Sham"][k]).pvalue
        ax.text(k, max(tops.values()) + 8.0, ("%.3f" % pb).lstrip("0"),
                ha="center", va="bottom", fontsize=5.9, color=INK2)
    ax.axhline(0, color=INK, lw=.8)
    ax.set_xticks(range(len(REGS)))
    ax.set_xticklabels(REGS, fontsize=7.4)
    ax.set_xlim(-.55, 1.55)
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(min(lo_, -6), hi_ + 12)
    ax.set_ylabel("Change during − before (%)")
    ax.set_title(title, fontsize=8.0, pad=10)


def panel_level(ax):
    for k, reg in enumerate(REGS):
        for j, (arm, c) in enumerate((("Treatment", STIM), ("Sham", NOST))):
            v = level("swa", reg, arm)
            x = k + (-.19 if j == 0 else .19)
            bs = np.median(RNG.choice(v, (4000, v.size), True), axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            ax.plot([x, x], [lo, hi], color=c, lw=2.2, zorder=3,
                    solid_capstyle="round")
            ax.scatter([x], [np.median(v)], s=30, color=c, zorder=5, lw=.7,
                       edgecolor="white")
        a, b = level("swa", reg, "Treatment"), level("swa", reg, "Sham")
        ax.text(k, max(np.median(a), np.median(b)) * 1.30,
                ("%.2f" % stats.mannwhitneyu(a, b).pvalue).lstrip("0"),
                ha="center", va="bottom", fontsize=5.9, color=INK2)
    ax.set_xticks(range(len(REGS)))
    ax.set_xticklabels(REGS, fontsize=7.4)
    ax.set_xlim(-.55, 1.55)
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(0, hi_ * 1.12)
    ax.set_ylabel("Slow-wave activity (µV² Hz⁻¹)")
    ax.set_title("Absolute power during stimulation (median)", fontsize=8.0, pad=10)


def main():
    PS.use()
    f = plt.figure(figsize=(7.09, 4.7))
    gs = f.add_gridspec(2, 2, wspace=.34, hspace=.62, left=.085, right=.975,
                        top=.925, bottom=.135)
    axa = f.add_subplot(gs[0, 0]); panel_change(axa, "swa", "Slow-wave activity, 0.5–4 Hz")
    axb = f.add_subplot(gs[0, 1]); panel_change(axb, "so_band", "Slow-oscillation band, 0.5–1 Hz")
    axc = f.add_subplot(gs[1, 0]); panel_change(axc, "sigma", "Sigma, 12–16 Hz")
    axd = f.add_subplot(gs[1, 1]); panel_level(axd)

    keys = [Line2D([], [], color=STIM, lw=6, label="Stimulation"),
            Line2D([], [], color=NOST, lw=6, label="No stimulation"),
            Line2D([], [], color="none", marker="$∗$", markerfacecolor=STIM,
                   markeredgecolor=STIM, ms=7,
                   label="q < .05 against own baseline")]
    f.legend(handles=keys, loc="lower center", ncol=3, fontsize=6.6,
             bbox_to_anchor=(.53, .005), handlelength=1.4, handletextpad=.5,
             columnspacing=1.6, borderpad=0)
    f.text(.53, .062, "Grey values are Mann–Whitney tests comparing the two "
           "conditions directly.", ha="center", fontsize=6.2, color=INK2)
    for ax, lt in zip((axa, axb, axc, axd), "abcd"):
        PS.tag(f, ax, lt, dx=.058, dy=.028)
    f.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    f.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(f)
    print("saved", OUT.name)
    print("=== numbers for the text (Frontal, Parietal only) ===")
    for band in ("swa", "so_band", "sigma"):
        ps = [stats.wilcoxon(change(band, r, "Treatment")).pvalue for r in REGS]
        qs = bh(ps)
        for k, reg in enumerate(REGS):
            a, b = change(band, reg, "Treatment"), change(band, reg, "Sham")
            print("  %-8s %-9s stim %+6.1f%% (q=%.3f, n=%d) | no-stim %+6.1f%% (n=%d) | direct P=%.3f"
                  % (band, reg, a.mean(), qs[k], len(a), b.mean(), len(b),
                     stats.mannwhitneyu(a, b).pvalue))
    print()
    for reg in REGS:
        a, b = level("swa", reg, "Treatment"), level("swa", reg, "Sham")
        print("  absolute SWA %-9s %.1f vs %.1f uV2/Hz   P=%.3f  (n=%d/%d)"
              % (reg, np.median(a), np.median(b), stats.mannwhitneyu(a, b).pvalue, len(a), len(b)))


if __name__ == "__main__":
    main()
