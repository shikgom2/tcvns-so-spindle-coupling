# -*- coding: utf-8 -*-
"""Coupling-rule sensitivity (Sensitivity analyses: coupling rule).

Primary rule: spindle centre inside the SO cycle and within +-90 deg of the up-state.
Alternative:  spindle centre within +-1 s of the SO trough (co-occurrence).
Both counts come from the same re-detection (redetect_multi.py, columns n_coupled and n_coupled_win1s).
Output  REV_coupling_rule_sensitivity.csv  condition x window interaction per region and rule, BH across regions within rule.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402
T = LP.PROJECT_ROOT / "Final analysis/tables"
EVENTS = LP.PROJECT_ROOT / "이전/redetect_multi/staresina_N2.parquet"
IDMAP = LP.PROJECT_ROOT / "이전/outputs/paper/INTERNAL_session_idmap.csv"       # internal: links event rows to session codes
REG = {"Frontal": ["Fp1", "Fp2", "F3", "F4"], "Temporal": ["F7", "F8", "T3", "T4"], "Central": ["C3", "Cz", "C4"], "Parietal": ["P3", "P4"]}
BL = {"baseline": "pre", "stimulation 1": "stim", "recovery 1": "post"}


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty_like(p); m = len(p); prev = 1.0
    for rank, i in enumerate(o[::-1], 1):
        prev = min(prev, p[i] * m / (m - rank + 1)); q[i] = prev
    return q


def main():
    d = pd.read_parquet(EVENTS).rename(columns={"원ID": "ID"}).dropna(subset=["Order"]); d["Order"] = d.Order.astype(int)
    idm = pd.read_csv(IDMAP)
    m = d.merge(idm[["session_id", "participant", "arm", "cohort", "ID", "Order"]].rename(columns={"arm": "cond"}), on=["cohort", "ID", "Order"], how="inner")
    m["win"] = m.block.map(BL)
    rows = []
    for reg, chs in REG.items():
        g = m[m.ch.isin(chs)].groupby(["session_id", "participant", "cond", "win"], as_index=False).agg(
            n_SO=("n_SO", "sum"), n_cp=("n_coupled", "sum"), n_cp1=("n_coupled_win1s", "sum"))
        g = g[g.n_SO >= 1]; g["prev"] = g.n_cp / g.n_SO; g["prev1"] = g.n_cp1 / g.n_SO; g["ROI"] = reg; rows.append(g)
    G = pd.concat(rows); G["win"] = pd.Categorical(G.win, ["pre", "stim", "post"]); G["trt"] = (G.cond == "Treatment").astype(int)
    out = []
    for y, lab in (("prev", "spindle centre within SO cycle and |phase| <= 90 deg (primary)"),
                   ("prev1", "spindle centre within +/-1 s of SO trough (co-occurrence)")):
        for reg in REG:
            x = G[G.ROI == reg]; md = smf.mixedlm("%s ~ trt*C(win)" % y, x, groups=x.participant).fit(); k = "trt:C(win)[T.stim]"; ci = md.conf_int().loc[k]
            out.append(dict(definition=lab, ROI=reg, beta=md.params[k], ci_lo=ci[0], ci_hi=ci[1], p=md.pvalues[k], n_part=x.participant.nunique(),
                            mean_prev_stim_treatment=x[(x.win == "stim") & (x.trt == 1)][y].mean(),
                            mean_prev_stim_sham=x[(x.win == "stim") & (x.trt == 0)][y].mean()))
    O = pd.DataFrame(out)
    for lab, sub in O.groupby("definition"):
        O.loc[sub.index, "q"] = bh(sub.p.values)
    O.to_csv(T / "REV_coupling_rule_sensitivity.csv", index=False); print(O.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
