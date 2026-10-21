# -*- coding: utf-8 -*-
"""Figure 3 — SO-locked spindle envelope. Let a statistical test, not the eye, carry the claim.

a, b  baseline-corrected envelope (frontal / parietal), with the mean SO faintly in the background
c, d  per-recording peak amplitude and peak latency — condition comparison

Tests
  · cluster-based permutation test (5,000 shuffles of condition labels), −1.0 to +1.0 s
  · peak amplitude and peak latency: Mann-Whitney + Hedges g
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402

warnings.filterwarnings("ignore")
import paperstyle as PS
_R = LP.PROJECT_ROOT
TBL = _R / "Final analysis/tables"
FIG = _R / "Final analysis/figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2,
    "axes.linewidth": .7, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.major.width": .7,
    "ytick.major.width": .7, "xtick.major.size": 2.4,
    "ytick.major.size": 2.4, "legend.frameon": False,
    "savefig.dpi": 600, "figure.dpi": 150,
})
STIM, NOST, GREY = PS.C["stim"], PS.C["nost"], "#9AA2A8"
RNG = np.random.default_rng(20261003)
BASE = (-1.5, -1.0)          # each recording's own baseline
WIN = (-1.0, 1.0)            # test interval
PEAK = (-1.0, 1.0)           # peak taken over the whole test interval (avoids bias towards the edges)


def hedges(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                / (na + nb - 2))
    if s == 0:
        return np.nan
    return (a.mean() - b.mean()) / s * (1 - 3 / (4 * (na + nb) - 9))


def cluster_test(A, B, t, n_perm=5000, alpha=.05):
    """Cluster-based permutation test that shuffles condition labels.

    Pointwise independent t -> contiguous samples with |t| above threshold form clusters, summed t is the statistic,
    compared against the distribution of the maximum |cluster sum| obtained by shuffling labels.
    """
    k = (t >= WIN[0]) & (t <= WIN[1])
    a, b = A[:, k], B[:, k]
    na, nb = a.shape[0], b.shape[0]
    thr = stats.t.ppf(1 - alpha / 2, na + nb - 2)

    def clusters(x, y):
        tv = stats.ttest_ind(x, y, axis=0).statistic
        sup = np.abs(tv) > thr
        out, i = [], 0
        while i < len(sup):
            if sup[i]:
                j = i
                while j + 1 < len(sup) and sup[j + 1]:
                    j += 1
                out.append((i, j, float(tv[i:j + 1].sum())))
                i = j + 1
            else:
                i += 1
        return tv, out

    tv, obs = clusters(a, b)
    if not obs:
        return dict(n_clusters=0, p_min=np.nan, t=t[k], tval=tv,
                    clusters=[], thr=thr)
    pooled = np.vstack([a, b])
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = RNG.permutation(na + nb)
        _, cl = clusters(pooled[idx[:na]], pooled[idx[na:]])
        null[i] = max((abs(c[2]) for c in cl), default=0.0)
    res = []
    for i0, i1, s in obs:
        p = float((null >= abs(s)).mean())
        res.append(dict(t_start=float(t[k][i0]), t_end=float(t[k][i1]),
                        cluster_stat=s, p=p))
    return dict(n_clusters=len(res), p_min=min(r["p"] for r in res),
                t=t[k], tval=tv, clusters=res, thr=thr)


def load(region):
    d = np.load(TBL / "SO_locked_regions.npz")
    t = d["t"]
    bl = (t >= BASE[0]) & (t <= BASE[1])
    out = {}
    for arm in ("Treatment", "Sham"):
        e = d["%s_%s_env" % (region, arm)]
        out[arm] = e - e[:, bl].mean(1, keepdims=True)
    w = np.vstack([d["%s_Treatment_wave" % region],
                   d["%s_Sham_wave" % region]])
    return t, out, w.mean(0)


def peaks(t, e):
    """Per-recording peak amplitude and peak latency (within 1 s after the SO trough)."""
    k = (t >= PEAK[0]) & (t <= PEAK[1])
    seg, tt = e[:, k], t[k]
    i = seg.argmax(1)
    return seg.max(1), tt[i]


def panel_wave(ax, t, E, wave, res, title):
    k = (t >= WIN[0]) & (t <= WIN[1])
    # mean SO in the background (shape only, no ticks)
    ax2 = ax.twinx()
    ax2.plot(t[k], wave[k], color=GREY, lw=1.1, alpha=.55, zorder=0)
    ax2.set_yticks([]); ax2.set_ylim(wave[k].min() * 2.3,
                                     wave[k].max() * 1.35)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    for arm, c, lab in (("Treatment", STIM, "Stimulation"),
                        ("Sham", NOST, "No stimulation")):
        e = E[arm][:, k]
        m = e.mean(0)
        s = e.std(0, ddof=1) / np.sqrt(e.shape[0])
        ax.fill_between(t[k], m - s, m + s, color=c, alpha=.18, lw=0,
                        zorder=2)
        PS.line(ax, t[k], m, c, lw=1.9, label=lab, zorder=3)
    ax.axvline(0, color=PS.C["ink"], lw=.8, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0, color="#1A1D26", lw=.6, ls=":", zorder=1)
    ax.set_xlim(*WIN)
    ax.set_xlabel("Time from slow-oscillation trough (s)", fontsize=7.2)
    ax.set_title(title, fontsize=8.6, pad=7)
    msg = ("no cluster, $P$ > .05" if res["n_clusters"] == 0
           else "min cluster $P$ = %.3f" % res["p_min"])
    ax.text(.025, .97, msg, transform=ax.transAxes, ha="left", va="top",
            fontsize=7.0, color="#6B7378")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)


def panel_peak(ax, vals, ylab, title, fmt="%.2f", unit=""):
    out = []
    for i, (arm, c) in enumerate((("Treatment", STIM), ("Sham", NOST))):
        v = vals[arm]
        PS.halfviolin(ax, v, i + .10, c, width=.34, side=1, alpha=.22)
        ax.scatter(i - .14 + RNG.uniform(-.11, .11, v.size), v, s=5.5,
                   color=c, alpha=.36, lw=0, zorder=2)
        bs = RNG.choice(v, (4000, v.size), True).mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.plot([i - .14, i - .14], [lo, hi], color=c, lw=2.4, zorder=4,
                solid_capstyle="round")
        PS.dot(ax, [i - .14], [v.mean()], c, s=34, zorder=6)
        out.append(v)
    u = stats.mannwhitneyu(out[0], out[1])
    g = hedges(out[0], out[1])
    # a few large values stretch the axis and squash the bulk, so clip at the 97th percentile
    allv = np.concatenate(out)
    hi = float(np.percentile(allv, 97))
    lo = float(min(allv.min(), 0))
    pad = (hi - lo) * .18
    top = hi + pad * 2.2          # top margin, so the statistics text does not overlap the violins
    n_out = int((allv > top).sum())
    if n_out:
        ax.set_ylim(lo - pad * .3, top)
        ax.text(.98, .03, "%d above axis" % n_out, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.2, color="#8C9296")
    elif allv.max() > hi * 1.6:
        ax.set_ylim(lo - pad * .3, top)
    ax.text(.5, .97, ("$P$ = %.2f · $g$ = %+.2f" % (u.pvalue, g)),
            transform=ax.transAxes, ha="center", va="top", fontsize=7.0,
            color="#6B7378")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Stimulation", "No stim."], fontsize=7.2)
    ax.set_xlim(-.55, 1.55)
    ax.set_ylabel(ylab, fontsize=7.4)
    ax.set_title(title, fontsize=8.6, pad=7)
    return dict(p=float(u.pvalue), g=float(g),
                mean_trt=float(out[0].mean()), mean_sham=float(out[1].mean()),
                n_trt=int(out[0].size), n_sham=int(out[1].size))


def main():
    PS.use()
    rows = []
    f = plt.figure(figsize=(7.09, 4.55))
    gs = f.add_gridspec(2, 2, height_ratios=[1.05, 1],
                        width_ratios=[1, 1], wspace=.33, hspace=.62,
                        left=.085, right=.975, top=.925, bottom=.120)
    axes = []
    store = {}
    for col, region in enumerate(("frontal", "parietal")):
        t, E, wave = load(region)
        res = cluster_test(E["Treatment"], E["Sham"], t)
        store[region] = (t, E, res)
        ax = f.add_subplot(gs[0, col])
        axes.append(ax)
        panel_wave(ax, t, E, wave, res,
                   region.capitalize() + "  ·  baseline-corrected")
        if col == 0:
            ax.set_ylabel("Spindle envelope change (µV)", fontsize=7.4)
        rows.append(dict(region=region, test="cluster permutation",
                         n_trt=E["Treatment"].shape[0],
                         n_sham=E["Sham"].shape[0],
                         n_clusters=res["n_clusters"], p_min=res["p_min"]))
        print("[%s] clusters=%d  p_min=%s" %
              (region, res["n_clusters"], res["p_min"]))
        for c in res["clusters"]:
            print("    %.2f to %.2f s  sum t = %.1f  P = %.3f"
                  % (c["t_start"], c["t_end"], c["cluster_stat"], c["p"]))

    # c, d — peak amplitude and latency (frontal)
    t, E, _ = store["frontal"]
    amp = {a: peaks(t, E[a])[0] for a in ("Treatment", "Sham")}
    lat = {a: peaks(t, E[a])[1] for a in ("Treatment", "Sham")}
    axc = f.add_subplot(gs[1, 0]); axes.append(axc)
    r1 = panel_peak(axc, amp, "Peak envelope change (µV)",
                    "Frontal  ·  peak amplitude")
    axd = f.add_subplot(gs[1, 1]); axes.append(axd)
    r2 = panel_peak(axd, lat, "Latency of peak (s)",
                    "Frontal  ·  peak latency")
    rows += [dict(region="frontal", test="peak amplitude", **r1),
             dict(region="frontal", test="peak latency", **r2)]
    print("peak amplitude  trt %.3f vs sham %.3f  P=%.3f g=%+.2f"
          % (r1["mean_trt"], r1["mean_sham"], r1["p"], r1["g"]))
    print("peak latency    trt %.3f vs sham %.3f  P=%.3f g=%+.2f"
          % (r2["mean_trt"], r2["mean_sham"], r2["p"], r2["g"]))

    h, l = axes[0].get_legend_handles_labels()
    f.legend(h, l, loc="lower center", bbox_to_anchor=(.52, .008), ncol=2,
             fontsize=7.4, handletextpad=.5, columnspacing=2.0)

    for ax, lt in zip(axes, "abcd"):
        p = ax.get_position()
        f.text(max(p.x0 - .058, .004), p.y1 + .022, lt, fontsize=10.5,
               fontweight="bold", va="bottom", ha="left")

    for a_ in axes:
        PS.panel(a_); a_.grid(False); a_.set_axisbelow(True)
    out = FIG / "Fig3_so_locked_tested"
    f.savefig(out.with_suffix(".png"), bbox_inches="tight")
    f.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(f)
    pd.DataFrame(rows).to_csv(TBL / "R_fig3_solocked_stats.csv",
                              index=False, encoding="utf-8-sig")
    print("\nsaved:", out.name)


if __name__ == "__main__":
    main()
