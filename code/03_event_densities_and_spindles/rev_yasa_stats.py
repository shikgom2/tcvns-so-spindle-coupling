# -*- coding: utf-8 -*-
"""YASA benchmark summary: absolute densities (manuscript detector vs YASA), coupling prevalence (two definitions), condition x window interaction.
Input   REV_yasa_cells.csv, REV_redetect_cells.csv
Output  REV_yasa_benchmark_densities.csv, REV_yasa_benchmark_interaction.csv"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
T = Path(r"D:/SO-Spindle논문/Final analysis/tables")
REG = {"Frontal": ["Fp1", "Fp2", "F3", "F4"], "Temporal": ["F7", "F8", "T3", "T4"], "Central": ["C3", "Cz", "C4"], "Parietal": ["P3", "P4"]}


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty_like(p); m = len(p); prev = 1.0
    for rank, i in enumerate(o[::-1], 1):
        prev = min(prev, p[i] * m / (m - rank + 1)); q[i] = prev
    return q


def dens_table(c, nso, nsp, ncp, label, extra=None):
    g = c.groupby(["session_id", "arm", "win"], as_index=False).agg(n_SO=(nso, "sum"), n_sp=(nsp, "sum"), n_cp=(ncp, "sum"), dur=("duration_sec", "first"), nch=("ch", "count"))
    g = g[g.dur >= 20]; m = g.dur / 60 * g.nch
    g["so"] = g.n_SO / m; g["sp"] = g.n_sp / m; g["prev"] = g.n_cp / g.n_SO.clip(lower=1)
    q1 = lambda v: v.quantile(.25); q3 = lambda v: v.quantile(.75)
    s = g.groupby(["arm", "win"]).agg(n=("session_id", "count"), SO_per_min_per_channel_median=("so", "median"), SO_q1=("so", q1), SO_q3=("so", q3),
                                      spindles_per_min_per_channel_median=("sp", "median"), spindle_q1=("sp", q1), spindle_q3=("sp", q3),
                                      coupling_prevalence_median=("prev", "median"), prev_q1=("prev", q1), prev_q3=("prev", q3)).reset_index()
    s.insert(0, "detector", label); return s


def interaction(c, nso, ncp, label):
    rows = []
    for reg, chs in REG.items():
        g = c[c.ch.isin(chs)].groupby(["session_id", "participant", "arm", "win"], as_index=False).agg(n_SO=(nso, "sum"), n_cp=(ncp, "sum"))
        g = g[g.n_SO >= 1].copy(); g["prevalence"] = g.n_cp / g.n_SO; g["trt"] = (g.arm == "Treatment").astype(int); g["win"] = pd.Categorical(g.win, ["pre", "stim", "post"])
        md = smf.mixedlm("prevalence ~ trt*C(win)", g, groups=g.participant).fit(); k = "trt:C(win)[T.stim]"; ci = md.conf_int().loc[k]
        rows.append(dict(analysis=label, ROI=reg, beta=md.params[k], ci_lo=ci[0], ci_hi=ci[1], p=md.pvalues[k], n_part=g.participant.nunique(), n_rows=len(g),
                         mean_prevalence_stim_window=g[g.win == "stim"].prevalence.mean()))
    o = pd.DataFrame(rows); o["q"] = bh(o.p.values); return o


def main():
    y = pd.read_csv(T / "REV_yasa_cells.csv"); ours = pd.read_csv(T / "REV_redetect_cells.csv")
    D = pd.concat([dens_table(ours, "n_SO", "n_spindle", "n_coupled", "Staresina-adapted detector, window-specific thresholds (this study)"),
                   dens_table(ours, "n_SO_rw", "n_spindle_rw", "n_coupled_rw", "Staresina-adapted detector, recording-wide thresholds (this study)"),
                   dens_table(y, "n_SO", "n_spindle", "n_cp_90", "YASA sw_detect + spindles_detect, coupling: spindle peak within SO and |phase| <= 90 deg"),
                   dens_table(y, "n_SO", "n_spindle", "n_cp_any", "YASA sw_detect + spindles_detect, coupling: spindle peak anywhere within SO")], ignore_index=True)
    D.to_csv(T / "REV_yasa_benchmark_densities.csv", index=False)
    I = pd.concat([interaction(ours, "n_SO", "n_coupled", "Staresina-adapted, window-specific (re-detected cells)"),
                   interaction(y, "n_SO", "n_cp_90", "YASA events, |phase| <= 90 deg"),
                   interaction(y, "n_SO", "n_cp_any", "YASA events, any phase within SO")], ignore_index=True)
    I.to_csv(T / "REV_yasa_benchmark_interaction.csv", index=False)
    pd.set_option("display.width", 250); print(D.round(3).to_string(index=False)); print(I.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
