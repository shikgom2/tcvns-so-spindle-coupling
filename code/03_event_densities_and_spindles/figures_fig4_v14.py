# -*- coding: utf-8 -*-
"""Figure 4 (v8) — drawn with recording-wide thresholds only. The increase in coupled SOs and spindles is the main result.

a      coupled SO density trajectories, 4 regions (REV_redetect_cells_N2N3.csv → summed per region, recording-wide thresholds)
b–d    post-window interaction β: coupled SOs · spindles · uncoupled SOs (REV_fig4_decomposition, recording-wide)
e      coupling prevalence interaction β (recording-wide)
f      SWA change relative to own pre-window in the stimulation group (Wilcoxon); no-stimulation group in grey for reference
g      per-participant SWA trajectories in the stimulation group
Output manuscript/figures/v14oct2026/Figure_4_slow_oscillation_population_v14.(png|pdf), tables/REV_fig4_source_densities_rw_N2N3.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch               # noqa: E402
from matplotlib.ticker import MaxNLocator                          # noqa: E402

sys.path.insert(0, r"D:/SO-Spindle논문/code/paper")
import paperstyle as PS                                           # noqa: E402

R = Path(r"D:/SO-Spindle논문/Final analysis")
T = R / "tables"
OUT = R / "manuscript/figures/v14oct2026/Figure_4_slow_oscillation_population_v14"
REG = {"Frontal": ["Fp1", "Fp2", "F3", "F4"], "Temporal": ["F7", "F8", "T3", "T4"],
       "Central": ["C3", "Cz", "C4"], "Parietal": ["P3", "P4"]}
REGS = list(REG)
RNG = np.random.default_rng(20261003)
STIM, NOST, INK, INK2 = PS.C["stim"], PS.C["nost"], PS.C["ink"], PS.C["ink2"]
DOWN, UP, NS, BAND, SHADE = "#3C8FA0", "#B23A6F", "#9AA0A4", "#F3F2EF", "#ECECEC"
KEY = {"Frontal": "fron", "Parietal": "pari"}


def region_cells_rw(c):
    out = []
    for reg, chs in REG.items():
        g = (c[c.ch.isin(chs)].groupby(["session_id", "participant", "arm", "win"], as_index=False)
             .agg(n_SO=("n_SO_rw", "sum"), n_cp=("n_coupled_rw", "sum"), n_sp=("n_spindle_rw", "sum"), dur=("duration_sec", "first")))
        g = g[(g.dur >= 20) & (g.n_SO >= 1)].copy(); mins = g.dur / 60
        g["unc"] = (g.n_SO - g.n_cp) / mins; g["cp"] = g.n_cp / mins; g["sp"] = g.n_sp / mins; g["prev"] = g.n_cp / g.n_SO; g["ROI"] = reg
        out.append(g)
    return pd.concat(out, ignore_index=True)


def boot_ci(v, n=4000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    b = np.array([RNG.choice(v, v.size, True).mean() for _ in range(n)])
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def sigcol(beta, q):
    if q >= .05:
        return NS
    return UP if beta > 0 else DOWN


def tidy(ax, grid="y"):
    if grid:
        ax.grid(axis=grid, color=PS.C["grid"], lw=.5, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=2, width=.6, labelsize=6.3)
    ax.spines["left"].set_color(INK2); ax.spines["bottom"].set_color(INK2)


def pstr(p):
    return "P < 0.001" if p < .001 else ("P = %.3f" % p if p < .01 else "P = %.2f" % p)


def forest(ax, rw, outcome, title, xlab, pooled=None):
    tidy(ax, grid=None)
    for i, reg in enumerate(REGS):
        yy = -i
        if i % 2 == 0:
            ax.axhspan(yy - .5, yy + .5, color=BAND, lw=0, zorder=0)
        b = rw[(rw.outcome == outcome) & (rw.ROI == reg) & (rw.contrast == "post vs pre")].iloc[0]
        cb = sigcol(b.beta, b.q)
        ax.plot([b.ci_lo, b.ci_hi], [yy, yy], color=cb, lw=1.3, solid_capstyle="round", zorder=3)
        ax.scatter(b.beta, yy, s=24, marker="s", facecolor=cb if b.q < .05 else "white", edgecolor=cb, lw=1.0, zorder=4)
        if b.q < .05:
            off = .04 * np.ptp(ax.get_xlim())
            ax.text(b.ci_hi + off if b.beta > 0 else b.ci_lo - off, yy, "q = %.3f" % b.q, va="center", ha="left" if b.beta > 0 else "right", fontsize=5.4, color=cb)
    if pooled is not None:
        yy = -4.35
        ax.axhline(-3.7, color=INK2, lw=.5, ls=(0, (2, 1.5)), zorder=1)
        b = pooled[(pooled.outcome == outcome) & (pooled.contrast == "post vs pre")].iloc[0]
        cb = sigcol(b.beta, b.p)
        ax.plot([b.ci_lo, b.ci_hi], [yy, yy], color=cb, lw=1.6, solid_capstyle="round", zorder=3)
        ax.scatter(b.beta, yy, s=34, marker="D", facecolor=cb if b.p < .05 else "white", edgecolor=cb, lw=1.0, zorder=4)
        if b.p < .05:
            off = .04 * np.ptp(ax.get_xlim())
            ax.text(b.ci_hi + off if b.beta > 0 else b.ci_lo - off, yy, pstr(b.p), va="center", ha="left" if b.beta > 0 else "right", fontsize=5.4, color=cb)
    ax.axvline(0, color=INK, lw=.7, zorder=2)
    ax.set_yticks([0, -1, -2, -3, -4.35]); ax.set_ylim(-4.95, .6)
    ax.set_title(title, fontsize=7.2, pad=4); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlab, fontsize=6.3); ax.xaxis.set_major_locator(MaxNLocator(4))


def main():
    PS.use()
    c = pd.read_csv(T / "REV_redetect_cells_N2N3.csv")
    g = region_cells_rw(c); g.to_csv(T / "REV_fig4_source_densities_rw_N2N3.csv", index=False)
    rw = pd.read_csv(T / "REV_fig4_decomposition.csv"); rw = rw[rw.scope == "recording-wide"]
    pooled = pd.read_csv(T / "REV_fig4_pooled_regions.csv")
    sw = pd.read_csv(T / "SWA_windows.csv")

    f = plt.figure(figsize=(7.09, 9.6))
    gs = f.add_gridspec(4, 4, height_ratios=[1.1, 1.0, 1.0, 1.0], wspace=.6, hspace=.7, left=.095, right=.985, top=.945, bottom=.075)

    # ── a: coupled SO density, normalised to each recording's own pre-window (log % change)
    axes_a = []; NDROP = []
    for j, reg in enumerate(REGS):
        ax = f.add_subplot(gs[0, j]); axes_a.append(ax); tidy(ax)
        ax.axvspan(.6, 1.4, color=SHADE, lw=0, zorder=0)
        ax.axhline(100, color=INK, lw=.6, zorder=1)
        for arm, col, ls in (("Sham", NOST, "--"), ("Treatment", STIM, "-")):
            x = g[(g.ROI == reg) & (g.arm == arm)].pivot_table(index="session_id", columns="win", values="cp")
            x = x.dropna(subset=["pre"])
            m_, lo, hi = [100.0], [100.0], [100.0]
            for w in ("stim", "post"):
                xx = x[[w, "pre"]].dropna().values
                idx = RNG.integers(0, len(xx), (4000, len(xx)))
                bs = 100 * xx[idx, 0].mean(1) / xx[idx, 1].mean(1)
                m_.append(100 * xx[:, 0].mean() / xx[:, 1].mean()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
                NDROP.append((reg, arm, w, len(xx)))
            ax.fill_between(range(3), lo, hi, color=col, alpha=.11, lw=0, zorder=1)
            ax.plot(range(3), m_, color=col, lw=1.5, ls=ls, zorder=3)
            ax.scatter(range(3), m_, s=16, color=col, edgecolor="white", lw=.6, zorder=4)
        r = rw[(rw.outcome == "coupled") & (rw.ROI == reg) & (rw.contrast == "post vs pre")].iloc[0]
        col_q = UP if r.q < .05 else INK2
        ax.set_ylim(0, 400); ax.set_yticks([0, 100, 200, 300])
        ax.text(.96, .96, "β = %+.2f\nq = %.3f" % (r.beta, r.q), transform=ax.transAxes, va="top", ha="right", fontsize=5.8,
                color=col_q, bbox=dict(boxstyle="round,pad=.25", fc="white", ec=col_q, lw=.5))
        ax.set_xticks(range(3)); ax.set_xticklabels(["Before", "During", "After"]); ax.set_xlim(-.35, 2.35)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(reg, fontsize=7.4, pad=4)
        if j == 0:
            ax.set_ylabel("Coupled SO density" + chr(10) + "(% of own pre-window, N2+N3)", fontsize=6.6)
    f.legend(handles=[Line2D([], [], color=STIM, lw=1.5, marker="o", ms=3, label="Stimulation"),
                      Line2D([], [], color=NOST, lw=1.5, ls="--", marker="o", ms=3, label="No stimulation"),
                      Patch(color=SHADE, label="Stimulation window")],
             loc="upper right", bbox_to_anchor=(.985, .995), ncol=3, fontsize=6, frameon=False, handlelength=1.8, columnspacing=1.4)

    # ── b–d: forests, post vs pre, recording-wide (prevalence itself is in Figure 1 / Table 1)
    gs2 = gs[1, :].subgridspec(1, 3, wspace=.42)
    ax_b = f.add_subplot(gs2[0]); ax_b.set_xlim(-.6, 2.4); forest(ax_b, rw, "coupled", "Coupled SO density", "Interaction β (events min⁻¹)", pooled=pooled)
    ax_b.set_yticklabels(REGS + ["All regions"])
    ax_c = f.add_subplot(gs2[1]); ax_c.set_xlim(-5, 28); forest(ax_c, rw, "spindle", "Spindle density", "Interaction β (events min⁻¹)", pooled=pooled)
    ax_c.set_yticklabels([""] * 5)
    ax_d = f.add_subplot(gs2[2]); ax_d.set_xlim(-8.5, 2.2); forest(ax_d, rw, "uncoupled", "Uncoupled SO density", "Interaction β (events min⁻¹)", pooled=pooled)
    ax_d.set_yticklabels([""] * 5)

    # ── f: SWA change within stimulation recordings
    ax_f = f.add_subplot(gs[2, :2]); tidy(ax_f)
    xs = []; k = 0
    for reg in ("Frontal", "Parietal"):
        for w in ("stim", "post"):
            col = "%s_swa" % KEY[reg]
            for arm in ("Treatment", "Sham"):
                d = sw[sw.arm == arm].dropna(subset=[col]).groupby(["participant", "win"])[col].mean().unstack("win").dropna(subset=[w, "pre"])
                v = (np.log(d[w]) - np.log(d["pre"])) * 100
                m_, lo, hi = boot_ci(v); pw = stats.wilcoxon(v).pvalue
                if arm == "Treatment":
                    ax_f.bar(k - .1, m_, width=.5, color=STIM, lw=0, zorder=3); ax_f.plot([k - .1, k - .1], [lo, hi], color=INK, lw=.9, zorder=4)
                    ax_f.text(k - .1, hi + 3, pstr(pw), ha="center", fontsize=5.7, color=STIM if pw < .05 else INK2, fontweight="bold" if pw < .05 else "normal")
                else:
                    ax_f.plot([k + .3, k + .3], [lo, hi], color=NS, lw=.9, zorder=3)
                    ax_f.scatter(k + .3, m_, s=16, marker="s", facecolor="white", edgecolor=NS, lw=1.0, zorder=4)
            xs.append("%s\n%s" % (reg, "during" if w == "stim" else "after")); k += 1
    ax_f.axhline(0, color=INK, lw=.7, zorder=2); ax_f.set_xticks(range(4)); ax_f.set_xticklabels(xs); ax_f.set_ylim(-45, 92); ax_f.set_xlim(-.6, 3.6); ax_f.set_yticks([-40, -20, 0, 20, 40, 60])
    ax_f.set_ylabel("Slow-wave activity\n(% change from own pre-window)", fontsize=6.6)
    ax_f.set_title("Slow-wave activity rises within stimulation recordings", fontsize=7.2, pad=4)
    ax_f.legend(handles=[Patch(color=STIM, label="Stimulation (P, Wilcoxon vs own pre-window)"),
                         Line2D([], [], marker="s", color=NS, mfc="white", lw=0, ms=4, label="No stimulation (reference)")],
                fontsize=5.5, loc="upper left", borderaxespad=.3, handlelength=1.2, labelspacing=.3)

    # ── g: participant trajectories in stimulation recordings
    axes_g = []
    for j, reg in enumerate(("Frontal", "Parietal")):
        ax = f.add_subplot(gs[2, 2 + j]); axes_g.append(ax); tidy(ax)
        col = "%s_swa" % KEY[reg]
        d = sw[sw.arm == "Treatment"].dropna(subset=[col]).groupby(["participant", "win"])[col].mean().unstack("win").reindex(columns=["pre", "stim", "post"])
        ax.axvspan(.6, 1.4, color=SHADE, lw=0, zorder=0)
        for _, row in d.iterrows():
            ax.plot(range(3), row.values, color=STIM, alpha=.18, lw=.7, zorder=2)
        med = d.median(); ax.plot(range(3), med.values, color=STIM, lw=2.0, zorder=4, solid_capstyle="round")
        ax.scatter(range(3), med.values, s=22, color=STIM, edgecolor="white", lw=.7, zorder=5)
        ax.set_yscale("log"); ax.set_ylim(3, 120); ax.set_yticks([5, 10, 20, 50, 100]); ax.set_yticklabels(["5", "10", "20", "50", "100"])
        ax.set_xticks(range(3)); ax.set_xticklabels(["Before", "During", "After"]); ax.set_xlim(-.35, 2.35)
        ax.set_title("%s, stimulation recordings" % reg, fontsize=7.2, pad=4)
        if j == 0:
            ax.set_ylabel("Slow-wave activity\n(µV² Hz⁻¹, log scale)", fontsize=6.6)
            ax.legend(handles=[Line2D([], [], color=STIM, lw=2, marker="o", ms=3.5, mec="white", label="Median"),
                               Line2D([], [], color=STIM, alpha=.35, lw=.8, label="Participants")], fontsize=5.5, loc="lower right", borderaxespad=.2, handlelength=1.4)
        else:
            ax.tick_params(axis="y", labelleft=False)

    # ── g: SO-locked spindle-band envelope, stimulation recordings, baseline-corrected (−1.5 to −1.0 s)
    z = np.load(T / "SO_locked_windows.npz"); tt = z["t"]; base = (tt >= -1.5) & (tt <= -1.0)
    axes_g2 = []; gs3 = gs[3, :].subgridspec(1, 4, wspace=.42)
    for j, reg in enumerate(REGS):
        ax = f.add_subplot(gs3[j]); axes_g2.append(ax); tidy(ax)
        for win, col, lab in (("before", "#9A9A9A", "Before"), ("during", STIM, "During"), ("after", "#6FB07F", "After")):
            k = "%s|%s|Treatment|env" % (reg, win)
            if k not in z.files:
                continue
            E = z[k]; E = E - np.nanmean(E[:, base], axis=1, keepdims=True)
            m = np.nanmean(E, 0); se = np.nanstd(E, 0) / np.sqrt(np.sum(np.isfinite(E[:, 0])))
            ax.plot(tt, m, color=col, lw=1.1, label=lab, zorder=3); ax.fill_between(tt, m - se, m + se, color=col, alpha=.18, lw=0, zorder=2)
        ax.axhline(0, color=INK2, lw=.5, ls="--", zorder=1); ax.axvline(0, color=INK, lw=.5, ls=":", zorder=1)
        ax.set_xlim(-1, 1); ax.set_xlabel("Time from SO trough (s)", fontsize=6.3); ax.set_title(reg, fontsize=7.2, pad=4)
        if j == 0:
            ax.set_ylabel("SO-locked spindle envelope" + chr(10) + "(µV, baseline-corrected)", fontsize=6.6)
            pass
    f.legend(handles=[Line2D([], [], marker="s", color=INK2, mfc=INK2, lw=0, ms=4, label="Condition × window interaction, after − before (recording-wide thresholds)"),
                      Line2D([], [], marker="D", color=INK2, mfc=INK2, lw=0, ms=4, label="Pooled over regions (region as fixed effect; P)"),
                      Line2D([], [], color=UP, lw=1.6, label="Increase, q < 0.05"), Line2D([], [], color=DOWN, lw=1.6, label="Decrease, q < 0.05"),
                      Line2D([], [], color=NS, lw=1.6, label="q ≥ 0.05")],
             loc="lower center", bbox_to_anchor=(.53, .004), ncol=5, fontsize=5.4, frameon=False, handletextpad=.5, columnspacing=1.4)
    yg = axes_g2[0].get_position().y1
    f.legend(handles=[Line2D([], [], color="#9A9A9A", lw=1.2, label="Before"), Line2D([], [], color=STIM, lw=1.2, label="During"), Line2D([], [], color="#6FB07F", lw=1.2, label="After")],
             loc="lower right", bbox_to_anchor=(.985, yg + .035), ncol=3, fontsize=6, frameon=False, handlelength=1.6, columnspacing=1.2, title="Stimulation recordings", title_fontsize=6)
    for ax_, lt in zip([axes_a[0], ax_b, ax_c, ax_d, ax_f, axes_g[0], axes_g2[0]], "abcdefg"):
        PS.tag(f, ax_, lt, dx=.058, dy=.012, size=10)
    f.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight"); f.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight"); plt.close(f)
    print("saved", OUT.name); print("n sessions per cell:", NDROP)
    for reg in REGS:
        for arm in ("Treatment", "Sham"):
            x = g[(g.ROI == reg) & (g.arm == arm)]
            print(reg, arm, " ".join("%s cp=%.2f sp=%.1f unc=%.2f" % (w, x[x.win == w].cp.mean(), x[x.win == w].sp.mean(), x[x.win == w].unc.mean()) for w in ("pre", "stim", "post")))


if __name__ == "__main__":
    main()
