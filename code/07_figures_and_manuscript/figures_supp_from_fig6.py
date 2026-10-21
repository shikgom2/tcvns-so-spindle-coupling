# -*- coding: utf-8 -*-
"""The two panels taken out of Figure 6, as stand-alone supplementary figures.

  Supplementary Fig. 3  lagged correlation of coupling × spatial spread
  Supplementary Fig. 4  coupling probability by within-recording amplitude quintile
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import pandas as pd                                               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402

warnings.filterwarnings("ignore")
import paperstyle as PS                                           # noqa: E402
import figures_mechanism as FM                                    # noqa: E402
import figures_timecourse as FT                                   # noqa: E402

R = LP.PROJECT_ROOT
TBL = R / "Final analysis/tables"
SF = R / "Final analysis/manuscript/figures"


def main():
    PS.use()

    # ── S3  lagged correlation
    L = pd.read_csv(TBL / "M7_lag_correlation.csv")
    f = plt.figure(figsize=(3.35, 2.9))
    ax = f.add_subplot(111)
    FT.lagpanel(ax, L)
    ax.set_title("")
    ax.set_ylabel("Spearman $r$ within recording")
    ax.set_xticklabels(["spread\nleads", "same\nbin", "coupling\nleads"],
                       fontsize=7.0, linespacing=1.3)
    f.subplots_adjust(left=.20, right=.97, top=.95, bottom=.17)
    out = SF / "S3_coupling_spread_lag_correlation"
    f.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    f.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(f)
    print("saved", out.name)

    # ── S4  amplitude quintiles
    f = plt.figure(figsize=(3.35, 2.9))
    ax = f.add_subplot(111)
    FM.panel_amp(ax)
    ax.set_title("")
    ax.set_xlabel("Slow-oscillation amplitude\n(within-recording quintile)",
                  linespacing=1.3)
    f.subplots_adjust(left=.19, right=.97, top=.95, bottom=.20)
    out = SF / "S4_coupling_by_amplitude_quintile"
    f.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    f.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(f)
    print("saved", out.name)


if __name__ == "__main__":
    main()


def s7_elapsed_time():
    """Supplementary Fig. 7 — the elapsed-time trend panel removed from Figure 2."""
    import figures_result4 as R4
    PS.use()
    f = plt.figure(figsize=(3.6, 3.0))
    ax = f.add_subplot(111)
    R4.time_course(ax)
    ax.set_title("")
    out = SF / "S7_coupling_versus_elapsed_time"
    f.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    f.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(f)
    print("saved", out.name)
