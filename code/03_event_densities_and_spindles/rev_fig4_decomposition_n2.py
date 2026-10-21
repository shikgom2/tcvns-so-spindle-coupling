# -*- coding: utf-8 -*-
"""Event-density decomposition restricted to N2 under recording-wide thresholds (Sensitivity analyses: stage definition).

Same models as Table 2 / figures_fig4_v14.py (which use N2+N3), fitted to the N2-only cells of rev_redetect.py.
Input   REV_redetect_cells.csv (per session x channel x window; columns *_rw are recording-wide counts)
Output  REV_fig4_decomposition_N2.csv        regional and pooled condition x window interactions, BH across regions
        REV_fig4_source_densities_rw_N2.csv  the region x window densities the models are fitted to
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402

warnings.filterwarnings("ignore")
T = LP.PROJECT_ROOT / "Final analysis/tables"
REG = {"Frontal": ["Fp1", "Fp2", "F3", "F4"], "Temporal": ["F7", "F8", "T3", "T4"], "Central": ["C3", "Cz", "C4"], "Parietal": ["P3", "P4"]}


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty_like(p); m = len(p); prev = 1.0
    for rank, i in enumerate(o[::-1], 1):
        prev = min(prev, p[i] * m / (m - rank + 1)); q[i] = prev
    return q


def main():
    c = pd.read_csv(T / "REV_redetect_cells.csv"); G = []
    for reg, chs in REG.items():
        g = c[c.ch.isin(chs)].groupby(["session_id", "participant", "arm", "win"], as_index=False).agg(
            n_SO=("n_SO_rw", "sum"), n_cp=("n_coupled_rw", "sum"), n_sp=("n_spindle_rw", "sum"), dur=("duration_sec", "first"))
        g = g[(g.dur >= 20) & (g.n_SO >= 1)].copy(); m = g.dur / 60
        g["coupled"] = g.n_cp / m; g["spindle"] = g.n_sp / m; g["uncoupled"] = (g.n_SO - g.n_cp) / m; g["prevalence"] = g.n_cp / g.n_SO; g["ROI"] = reg
        G.append(g)
    G = pd.concat(G); G["trt"] = (G.arm == "Treatment").astype(int); G["win"] = pd.Categorical(G.win, ["pre", "stim", "post"])
    rows = []
    for y in ("coupled", "spindle", "uncoupled", "prevalence"):
        for con, k in (("stim vs pre", "trt:C(win)[T.stim]"), ("post vs pre", "trt:C(win)[T.post]")):
            for reg in REG:
                x = G[G.ROI == reg]; md = smf.mixedlm("%s ~ trt*C(win)" % y, x, groups=x.participant).fit(); ci = md.conf_int().loc[k]
                rows.append(dict(scope="recording-wide", stage="N2", outcome=y, contrast=con, ROI=reg, beta=md.params[k], ci_lo=ci[0], ci_hi=ci[1], p=md.pvalues[k], n_part=x.participant.nunique()))
            md = smf.mixedlm("%s ~ trt*C(win) + C(ROI)" % y, G, groups=G.participant).fit(); ci = md.conf_int().loc[k]
            rows.append(dict(scope="recording-wide", stage="N2", outcome=y, contrast=con, ROI="All regions", beta=md.params[k], ci_lo=ci[0], ci_hi=ci[1], p=md.pvalues[k], n_part=G.participant.nunique()))
    O = pd.DataFrame(rows)
    for (y, con), sub in O[O.ROI != "All regions"].groupby(["outcome", "contrast"]):
        O.loc[sub.index, "q"] = bh(sub.p.values)
    O.to_csv(T / "REV_fig4_decomposition_N2.csv", index=False); G.to_csv(T / "REV_fig4_source_densities_rw_N2.csv", index=False)
    print(O[O.contrast == "post vs pre"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
