# -*- coding: utf-8 -*-
"""Figure 4 (merged) — SO-locked spindle envelope (former Fig 4 a–h) + PPC spectra (former Fig 6 a–f)
on one sheet, 14 panels a–n.

 a–c  frontal: waveform+envelope / corrected envelope / difference
 d–f  parietal: same layout
 g,h  per-recording peak amplitude / peak latency
 i    mean PPC spectrum   j  distribution of per-recording peak frequency
 k–n  per-region ridges (frontal·temporal·central·parietal)
Panel functions are used as is from figures_fig2_merged (FM) and figures_ppc_spectra (FP).
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
from matplotlib.colors import Normalize                           # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402
import paperstyle as PS                                           # noqa: E402
import figures_fig2_merged as FM                                  # noqa: E402
import figures_ppc_spectra as FP                                  # noqa: E402
import figures_fig3_solocked as SL                                # noqa: E402

R = LP.PROJECT_ROOT
TBL = R / "Final analysis/tables"
OUT = R / "Final analysis/manuscript/figures/Figure_04_spindle_envelope_and_frequency_specificity"


def main():
    PS.use()
    d = np.load(TBL / "SO_locked_regions.npz")
    D = pd.read_csv(TBL / "R_ppc_spectrum.csv")
    D = D[D.region.isin(FP.REG)]
    vmax = float(np.nanpercentile(D.ppc.values, 99))

    f = plt.figure(figsize=(7.09, 10.9))
    gs = f.add_gridspec(5, 15, height_ratios=[1, 1, 1.05, .95, 1.25],
                        wspace=.9, hspace=.62, left=.085, right=.975,
                        top=.968, bottom=.075)
    axes = []
    amp, lat = {}, {}
    # ── rows 1–2: envelope
    for r, (reg, Reg, chans) in enumerate(FM.REG):
        t, E, wave = SL.load(reg)
        Eabs = {a: d["%s_%s_env" % (reg, a)] for a in ("Treatment", "Sham")}
        res = SL.cluster_test(E["Treatment"], E["Sham"], t)
        ax = f.add_subplot(gs[r, 0:4]); axes.append(ax)
        FM.raw_panel(ax, t, Eabs, wave, Reg, chans)
        ax = f.add_subplot(gs[r, 6:10]); axes.append(ax)
        FM.corrected_panel(ax, t, E, res, "Baseline-corrected")
        ax = f.add_subplot(gs[r, 11:15]); axes.append(ax)
        FM.diff_panel(ax, t, E, "Stimulation − no stimulation")
        for a in ("Treatment", "Sham"):
            pa_, la_ = SL.peaks(t, E[a])
            amp[(Reg, a)] = pa_; lat[(Reg, a)] = la_
    # ── row 3: peaks
    axg = f.add_subplot(gs[2, 0:7]); axes.append(axg)
    FM.raincloud_panel(axg, amp, "Peak envelope change (µV)", "Peak amplitude")
    axh = f.add_subplot(gs[2, 8:15]); axes.append(axh)
    FM.raincloud_panel(axh, lat, "Latency of peak (s)", "Peak latency", clip_q=100)
    # ── row 4: PPC mean + peak histogram
    axi = f.add_subplot(gs[3, 0:7]); axes.append(axi)
    FP.mean_panel(axi, D)
    axj = f.add_subplot(gs[3, 8:15]); axes.append(axj)
    FP.peak_panel(axj, D)
    # ── row 5: ridges
    ridge_axes = []
    for k, reg in enumerate(FP.REG):
        ax = f.add_subplot(gs[4, 4 * k:4 * k + 3]); axes.append(ax); ridge_axes.append(ax)
        FP.ridge(ax, D, reg, vmax, show_y=(k == 0))

    f.canvas.draw()
    # legend for rows 1–3 (under row 3)
    pg = axg.get_position()
    keys = [Line2D([], [], color=FM.WAVE, lw=1.8, label="Slow oscillation"),
            Line2D([], [], color=PS.C["stim"], lw=1.8, label="Stimulation"),
            Line2D([], [], color=PS.C["nost"], lw=1.8, label="No stimulation")]
    f.legend(handles=keys, loc="upper center", bbox_to_anchor=(.53, pg.y0 - .012),
             ncol=3, fontsize=6.8, handlelength=1.6, columnspacing=1.8,
             frameon=False, borderpad=0)
    # region legend under panel i
    pi = axi.get_position()
    rk = [Line2D([], [], color=FP.RCOL[r], lw=1.9, label=r) for r in FP.REG]
    f.legend(handles=rk, loc="upper center",
             bbox_to_anchor=((pi.x0 + pi.x1) / 2, pi.y0 - .030), ncol=4,
             fontsize=6.4, handlelength=1.3, handletextpad=.4, columnspacing=1.1,
             frameon=False, borderpad=0)
    # colour bar under ridges
    pk, pn = ridge_axes[0].get_position(), ridge_axes[-1].get_position()
    cw = .20
    cax = f.add_axes([(pk.x0 + pn.x1) / 2 - cw / 2, pn.y0 - .050, cw, .011])
    cb = f.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap=FP.CMAP),
                    cax=cax, orientation="horizontal",
                    ticks=np.round(np.linspace(0, vmax, 3), 2))
    cb.set_label("Peak PPC of that recording", fontsize=6.4, labelpad=2)
    cb.ax.tick_params(labelsize=6.0, length=2.0, pad=1.2)
    cb.outline.set_linewidth(.7)

    for ax, lt in zip(axes, "abcdefghijklmn"):
        p = ax.get_position()
        f.text(max(p.x0 - .052, .002), p.y1 + .010, lt, fontsize=10.5,
               fontweight="bold", va="bottom", ha="left")
    for a_ in axes:
        PS.panel(a_); a_.grid(False)
    f.savefig(OUT.with_suffix(".png"), dpi=300)
    f.savefig(OUT.with_suffix(".pdf"))
    print("saved", OUT)


if __name__ == "__main__":
    main()
