# -*- coding: utf-8 -*-
"""Result 4 main figure — comparison of three windows within stimulation recordings. Staresina · N2 only.

a trajectory across windows (four regions)
b violins of during − before stimulation for the four regions
c after − before stimulation for the four regions
d after − during stimulation for the four regions  (persistence)
e,f,g per-electrode d_z topographies, three contrasts
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe                               # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
import matplotlib.transforms as mtransforms                       # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_paths as LP  # noqa: E402
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
SIG, NS_ = PS.C["stim"], PS.C["ns"]
NOSTC = PS.C["nost"]
STAR = "#E8B51F"
POSC, NEGC = "#2F7A5B", PS.C["down"]
RCOL = PS.ROI
ROIS = ["Frontal", "Temporal", "Central", "Parietal"]
# Main-text values for Result 4 come from the reference detector (Results 1-3 use Staresina).
# The detector is not repeated in main-text sentences, but must always be stated in figure captions.
DET = "staresina_N2"
DETLAB = {"jy_N2": "Reference detector, N2",
          "staresina_N2": "Staresina detector, N2"}
RNG = np.random.default_rng(20261003)
TMAX = 40.0                 # almost all recordings extend only this far (min)
CONTR = [("stim", "pre", "During − before"),
         ("post", "pre", "After − before"),
         ("post", "stim", "After − during")]


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


def perm(d, n=20000):
    d = np.asarray(d, float)
    o = abs(d.mean())
    fl = RNG.choice([-1.0, 1.0], (n, d.size))
    return float((np.abs((fl * d).mean(1)) >= o).mean())


def tag(f, ax, s, dx=.048, dy=.040):
    p = ax.get_position()
    f.text(max(p.x0 - dx, .002), p.y1 + dy, s, fontsize=11,
           fontweight="bold", va="bottom", ha="left")


def win_diff(t, a, b):
    """Per-participant difference between two windows in stimulation recordings."""
    w = t.pivot_table(index="participant", columns="win", values="prevalence")
    if not {a, b} <= set(w.columns):
        return pd.Series(dtype=float)
    return (w[a] - w[b]).dropna()


def chan_diffs(a, b):
    c = pd.read_csv(OUT / "S_Table_channel_metrics.csv")
    c = c[(c.detector == DET) & (c.arm == "Treatment")].dropna(
        subset=["prevalence"])
    rows = []
    for ch in TP.POS:
        d = win_diff(c[c.ch == ch], a, b)
        if len(d) < 10:
            continue
        sd = d.std(ddof=1)
        rows.append(dict(ch=ch, ROI=TP.ROI_OF[ch], n=len(d),
                         dz=float(d.mean() / sd) if sd else np.nan,
                         diff_pp=float(d.mean() * 100), p=perm(d.values)))
    C = pd.DataFrame(rows)
    C["q"] = fdr(C.p.values)
    return C


def violin(ax, g, a, b, title):
    stats_ = []
    for i, roi in enumerate(ROIS):
        t = g[(g.detector == DET) & (g.ROI == roi)
              & (g.arm == "Treatment")].dropna(subset=["prevalence"])
        v = win_diff(t, a, b).values * 100
        if not len(v):
            continue
        sd = v.std(ddof=1)
        bs = RNG.choice(v, (4000, v.size), True).mean(1)
        stats_.append(dict(ROI=roi, n=len(v), mean=v.mean(),
                           ci_lo=float(np.percentile(bs, 2.5)),
                           ci_hi=float(np.percentile(bs, 97.5)),
                           dz=v.mean() / sd if sd else np.nan,
                           p=perm(v / 100)))
        PS.halfviolin(ax, v, i + .08, "#7E96A3", width=.36, side=1,
                      alpha=.28)
        ax.scatter(i - .14 + RNG.uniform(-.11, .11, v.size), v, s=5,
                   color=[POSC if x > 0 else NEGC for x in v], alpha=.45,
                   lw=0, zorder=2)
        _lo, _hi = np.percentile(bs, [2.5, 97.5])
        ax.plot([i - .14, i - .14], [_lo, _hi], color="#1A1D26", lw=2.2,
                zorder=4, solid_capstyle="round")
        PS.dot(ax, [i - .14], [v.mean()], "#1A1D26", s=26, zorder=6,
               halo=False)
    S = pd.DataFrame(stats_)
    S["q"] = fdr(S.p.values)
    lo, top = ax.get_ylim()
    ax.set_ylim(lo, top + (top - lo) * .16)
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, r in S.iterrows():
        sig, nom = r.q < .05, (r.q >= .05 and r.p < .05)
        # No stars on violins — significant values are shown only in bold/darker colour
        ax.text(i, 1.015, "%+.1f" % r["mean"], transform=tr, ha="center",
                va="bottom", fontsize=8.2 if sig else 6.8, clip_on=False,
                color=SIG if sig else ("#B4651A" if nom else "#8C9296"),
                fontweight="bold" if sig else "normal")
    ax.axhline(0, color="k", lw=.8, ls="--", dashes=(3, 2))
    ax.set_xticks(range(4))
    ax.set_xticklabels([r[:4] for r in ROIS], fontsize=7)
    ax.set_title(title, fontsize=8.2, pad=14)
    return S


def time_course(ax):
    """b — distribution of per-recording time trends (slopes). Looks at \"drift\", not level.

    Overlaying per-bin means for the two conditions makes level differences look like
    trends, because each bin contains a different set of recordings. Converting to one
    slope per recording removes that confound, and the tested quantity (differs from 0?)
    is shown directly in the figure.
    """
    TC = pd.read_csv(TBL / "R4_time_course.csv")
    TC["tb"] = np.floor(TC.t_min / 5.0) * 5.0 + 2.5
    TC = TC[TC.tb <= TMAX]
    rows = []
    for (sid, arm), t in TC.groupby(["session", "arm"]):
        if t.tb.nunique() < 3:
            continue
        rows.append(dict(arm=arm, session=sid,
                         slope=float(np.polyfit(t.tb, t.prevalence * 100,
                                                1)[0]) * 10))
    D = pd.DataFrame(rows)
    LIM = 15.0
    # Same grammar as c-e (vertical violins) — values written in the top margin do not overlap
    for i, (arm, col) in enumerate((("Treatment", SIG), ("Sham", NOSTC))):
        v = D[D.arm == arm].slope.values
        w = np.clip(v, -LIM, LIM)
        PS.halfviolin(ax, w, i + .08, col, width=.38, side=1, alpha=.22)
        ax.scatter(i - .14 + RNG.uniform(-.11, .11, w.size), w, s=5.5,
                   color=col, alpha=.40, lw=0, zorder=2)
        bs = RNG.choice(v, (4000, v.size), True).mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.plot([i - .14, i - .14], [lo, hi], color=col, lw=2.4,
                solid_capstyle="round", zorder=4)
        PS.dot(ax, [i - .14], [v.mean()], col, s=32, zorder=6)
        tr = mtransforms.blended_transform_factory(ax.transData,
                                                   ax.transAxes)
        ax.text(i, 1.015, "%+.2f\n$P$ = %.2f"
                % (v.mean(), float(stats.wilcoxon(v).pvalue)), transform=tr,
                ha="center", va="bottom", fontsize=6.4, color=col,
                linespacing=1.4, clip_on=False)
    pb = float(stats.mannwhitneyu(D[D.arm == "Treatment"].slope,
                                  D[D.arm == "Sham"].slope).pvalue)
    ax.axhline(0, color="#1A1D26", lw=.9, ls="--", dashes=(3, 2), zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Stimulation\n($n$ = %d)"
                        % (D.arm == "Treatment").sum(),
                        "No stimulation\n($n$ = %d)"
                        % (D.arm == "Sham").sum()], fontsize=7,
                       linespacing=1.4)
    ax.set_xlim(-.72, 1.72)
    ax.set_ylim(-LIM - 1.4, LIM + 1.4)
    ax.set_ylabel("Drift in coupling prevalence\n(pp per 10 min)",
                  fontsize=7.2, linespacing=1.4)
    ax.text(.5, 1.235, "$P$ = %.2f between conditions" % pb,
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.4,
            color="#6B7378")
    ax.set_title("No drift across the recording", fontsize=8.2, pad=40)


def main():
    PS.use()
    g = pd.read_csv(OUT / "S_Table_session_metrics.csv")
    f = plt.figure(figsize=(7.2, 9.1))
    outer = f.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.15, .80],
                           hspace=.74, left=.088, right=.975, top=.945,
                           bottom=.050)
    r0 = outer[0].subgridspec(1, 3, width_ratios=[1.45, 1, .14],
                              wspace=.40)
    r1 = outer[1].subgridspec(1, 4, width_ratios=[1, 1, 1, .20], wspace=.34)
    r2 = outer[2].subgridspec(1, 4, width_ratios=[1, 1, 1, .20],
                              wspace=.02)
    r3 = outer[3].subgridspec(1, 4, width_ratios=[1, 1, 1, .14], wspace=.30)

    # a — trajectory across windows
    axa = f.add_subplot(r0[0])
    t = g[(g.detector == DET) & (g.arm == "Treatment")]
    for roi in ROIS:
        m, e = [], []
        for w in ("pre", "stim", "post"):
            v = (t[(t.ROI == roi) & (t.win == w)].groupby("participant")
                 .prevalence.mean().dropna().values)
            m.append(v.mean() * 100)
            e.append(v.std(ddof=1) / np.sqrt(v.size) * 100)
        axa.errorbar([0, 1, 2], m, yerr=e, fmt="none", ecolor=RCOL[roi],
                     elinewidth=1.2, capsize=2.6, alpha=.8, zorder=3)
        PS.line(axa, [0, 1, 2], m, RCOL[roi], lw=2.0, zorder=4)
        PS.dot(axa, [0, 1, 2], m, RCOL[roi], s=26, zorder=6, halo=False)
        axa.plot([], [], "o-", color=RCOL[roi], ms=4.5, mec="white", mew=.7,
                 label=roi)
    axa.axvspan(.5, 1.5, color="#1D6B8C", alpha=.055, lw=0, zorder=0)
    axa.set_xticks([0, 1, 2])
    axa.set_xticklabels(["Before", "During", "After"], fontsize=7)
    axa.set_xlim(-.35, 2.35)
    axa.set_ylabel("Coupling prevalence (%)")
    axa.set_title("Stimulation nights", fontsize=8.2, pad=10)
    axa.legend(loc="lower right", ncol=2, fontsize=5.8, handletextpad=.3,
               columnspacing=.7, handlelength=1.0, borderpad=.25,
               labelspacing=.25)

    # b — coupling prevalence over elapsed time
    axb = f.add_subplot(r0[1])
    time_course(axb)

    # c, d, e — violins for the three contrasts
    S, axs = {}, []
    for k, (a, b, lab) in enumerate(CONTR):
        ax = f.add_subplot(r1[k], sharey=axs[0] if k == 1 else None)
        axs.append(ax)
        S[lab] = violin(ax, g, a, b, lab)
        if k == 1:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax.set_ylabel("Change (pp)")

    # f, g, h — per-electrode topographies
    Cs = {lab: chan_diffs(a, b) for a, b, lab in CONTR}
    allv = np.concatenate([c.dz.values for c in Cs.values()])
    vmax = float(np.nanpercentile(np.abs(allv), 98))
    topo_ax = []
    for k, (a, b, lab) in enumerate(CONTR):
        ax = f.add_subplot(r2[k])
        topo_ax.append(ax)
        im = TP.topo(ax, dict(zip(Cs[lab].ch, Cs[lab].dz)),
                     {x + "_p": v for x, v in zip(Cs[lab].ch, Cs[lab].q)},
                     vmax, lab, show_names=False)

    # The three heads are square, so they float to the centre of wide cells — left-align manually
    f.canvas.draw()
    pr = topo_ax[0].get_position()
    y0, hh = pr.y0, pr.y1 - pr.y0
    L, GAP = .052, .012
    ww = hh * 9.1 / 7.2                      # figure-width share of a square head
    for k, ax in enumerate(topo_ax):
        ax.set_position([L + k * (ww + GAP), y0, ww, hh])
    p1 = topo_ax[2].get_position()
    cax = f.add_axes([p1.x1 + .018, y0 + hh * .22, .013, hh * .56])
    cb = f.colorbar(im, cax=cax,
                    ticks=np.round(np.linspace(-vmax, vmax, 5), 2))
    cb.set_label("$d_z$ per electrode", fontsize=7.6, labelpad=4)
    cb.ax.tick_params(labelsize=7.0, length=2.8, width=.8, pad=2)
    cb.outline.set_linewidth(.8)

    # i, j, k — region-level forest. Uses the same estimates and same FDR as c–e.
    fore = []
    yy = np.arange(4)[::-1]
    for k, (a, b, lab) in enumerate(CONTR):
        ax = f.add_subplot(r3[k])
        fore.append(ax)
        t = S[lab].set_index("ROI").reindex(ROIS)
        for i, roi in enumerate(ROIS):
            r = t.loc[roi]
            sig = np.isfinite(r.q) and r.q < .05
            nom = (not sig) and np.isfinite(r.p) and r.p < .05
            col = SIG if sig else ("#B4651A" if nom else NS_)
            PS.grad_ci(ax, r.ci_lo, r.ci_hi, yy[i], col, center=r["mean"],
                       h=.32)
            PS.dot(ax, [r["mean"]], [yy[i]], col, s=34, zorder=6, halo=sig)
            if sig or nom:
                ax.text(r.ci_hi + .45, yy[i], "%+.1f" % r["mean"],
                        fontsize=7.4 if sig else 6.6, va="center", color=col,
                        fontweight="bold" if sig else "normal")
        ax.axvline(0, color="k", lw=.7, ls="--", dashes=(3, 2))
        ax.set_ylim(-.6, 3.5); ax.set_xlim(-3.2, 7.4)
        ax.set_xlabel("Change (pp)", fontsize=7.2)
        if k == 0:
            ax.set_yticks(yy[::-1])
            ax.set_yticklabels(ROIS[::-1], fontsize=7.2)
            for tick, roi in zip(ax.get_yticklabels(), ROIS[::-1]):
                qq = t.loc[roi, "q"]
                if np.isfinite(qq) and qq < .05:
                    tick.set_fontweight("bold"); tick.set_color(SIG)
        else:
            ax.set_yticks(yy[::-1]); ax.set_yticklabels([])
        ax.set_title(lab, fontsize=8.2, pad=6)

    for ax, lt in zip([axa, axb], "ab"):
        tag(f, ax, lt)
    for k, (ax, lt) in enumerate(zip(axs, "cde")):
        tag(f, ax, lt, dx=.048 if k != 1 else .020, dy=.012)
    for ax, lt in zip(topo_ax, "fgh"):
        tag(f, ax, lt, dx=.012, dy=.004)
    for k, (ax, lt) in enumerate(zip(fore, "ijk")):
        tag(f, ax, lt, dx=.056 if k == 0 else .020)

    for a_ in [axa, axb] + axs + fore:
        PS.panel(a_); a_.grid(False)
        a_.set_axisbelow(True)
    f.savefig(FIG / "Fig4_withinnight.png", bbox_inches="tight")
    f.savefig(FIG / "Fig4_withinnight.pdf", bbox_inches="tight")
    plt.close(f)

    out = []
    for lab, s in S.items():
        s = s.assign(contrast=lab)
        out.append(s)
    R = pd.concat(out)
    R.to_csv(TBL / "R4_region_staresina.csv", index=False,
             encoding="utf-8-sig")
    pd.concat([c.assign(contrast=k) for k, c in Cs.items()]).to_csv(
        TBL / "R4_channel_staresina.csv", index=False, encoding="utf-8-sig")
    print("Fig4_withinnight  (%s)" % DETLAB[DET])
    print(R.round(4).to_string(index=False))
    for lab, c in Cs.items():
        print("[%s] electrodes P<.05 %d/13, q<.05 %d/13"
              % (lab, int((c.p < .05).sum()), int((c.q < .05).sum())))


DL4 = {"jy_N2": "Reference · N2", "jy_N2N3": "Reference · N2+N3",
       "staresina_N2": "Staresina · N2",
       "staresina_N2N3": "Staresina · N2+N3"}


def supplementary():
    """Supplementary figure — all four detectors side by side. States which value comes from where."""
    W = pd.read_csv(OUT / "S_Table_withinnight.csv")
    CM = {"Stim − Pre": "During − before", "Post − Pre": "After − before",
          "Post − Stim": "After − during"}
    W["c2"] = W.contrast.map(CM)
    labs = [l for _, _, l in CONTR]
    dets = list(DL4)

    f, axs = plt.subplots(1, 3, figsize=(10.0, 5.2), sharex=True,
                          gridspec_kw=dict(wspace=.09, left=.215,
                                           right=.985, top=.875,
                                           bottom=.150))
    rows = [(d, r) for d in dets for r in ROIS]
    y = np.arange(len(rows))[::-1]
    for ax, lab in zip(axs, labs):
        t = W[W.c2 == lab].set_index(["detector", "ROI"]).reindex(rows)
        for i, (_, r) in enumerate(t.iterrows()):
            sig, nom = r.q < .05, (r.q >= .05 and r.p_perm < .05)
            col = SIG if sig else ("#B4651A" if nom else NS_)
            ax.plot([r.ci_lo * 100, r.ci_hi * 100], [y[i], y[i]], color=col,
                    lw=1.5, solid_capstyle="round")
            ax.scatter([r.mean_diff * 100], [y[i]], s=30 if sig else 22,
                       color=col, zorder=3, lw=.5, edgecolor="white")
            if sig or nom:                      # bold values instead of stars
                ax.text(r.ci_hi * 100 + .5, y[i], "%+.1f" % (r.mean_diff
                                                             * 100),
                        fontsize=7 if sig else 6.2, va="center", color=col,
                        fontweight="bold" if sig else "normal")
        ax.axvline(0, color="k", lw=.7, ls="--", dashes=(3, 2))
        ax.set_ylim(-.8, len(rows) - .2)
        ax.set_title(lab, fontsize=8.5, pad=5)
        ax.set_xlabel("Change (pp)")
        for k in range(1, 4):
            ax.axhline(len(rows) - 4 * k - .5, color="#D7D2C9", lw=.7)
    axs[0].set_yticks(y[::-1])
    axs[0].set_yticklabels([r for _, r in rows][::-1], fontsize=7.4)
    for tick, (d_, r_) in zip(axs[0].get_yticklabels(), rows[::-1]):
        row = W[(W.c2 == labs[0]) & (W.detector == d_) & (W.ROI == r_)]
        if len(row) and row.iloc[0].q < .05:
            tick.set_fontweight("bold")
    # Detector names — two horizontal lines to the left of each block (rotated text overlaps)
    p0 = axs[0].get_position()
    for k, d_ in enumerate(dets):
        yc = p0.y1 - (p0.y1 - p0.y0) * (4 * k + 2) / len(rows)
        f.text(p0.x0 - .082, yc, DL4[d_].replace(" · ", "\n"), fontsize=8.2,
               va="center", ha="right", color="#1A1D26", linespacing=1.35)
    for ax in axs[1:]:
        ax.set_yticks(y[::-1]); ax.set_yticklabels([])
    f.suptitle("Within-night contrasts under all four detector "
               "configurations", fontsize=10.5, x=.045, y=.962, ha="left")
    f.text(.045, .020,
           "Stimulation recordings only · mean change with 95 % bootstrap "
           "interval · values printed where P < .05, in bold where q < .05\n"
           "Every cell surviving correction does so under a reference "
           "configuration; the main-text values for this result are taken "
           "from Reference · N2.",
           fontsize=7.0, color="#6B7378")
    f.savefig(FIG / "FigS4_withinnight_detectors.png", bbox_inches="tight")
    f.savefig(FIG / "FigS4_withinnight_detectors.pdf", bbox_inches="tight")
    plt.close(f)
    print("FigS4_withinnight_detectors   q<.05 cells by detector:")
    print(W.assign(sig=W.q < .05).groupby("detector").sig.sum().to_string())


if __name__ == "__main__":
    main()
    supplementary()
