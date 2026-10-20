# -*- coding: utf-8 -*-
"""Sleep-staging algorithm sensitivity: primary model when stream B is staged by YASA (P4) instead of NeuroNet (window-specific thresholds, N2).
Input   REV_redetect_cells.csv (A+B, NeuroNet B), REV_redetect_cells_yasaB.csv (B, YASA)
Output  REV_staging_sensitivity.csv"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:/SO-Spindle논문/code/paper")
import rev_avgref_stats as AS                                     # noqa: E402

T = Path(r"D:/SO-Spindle논문/Final analysis/tables")
REGS = ["Frontal", "Temporal", "Central", "Parietal"]


def fit(g, label):
    rows = []
    for reg in REGS:
        x = g[g.ROI == reg].copy(); x["trt"] = (x.arm == "Treatment").astype(int); x["win"] = pd.Categorical(x.win, ["pre", "stim", "post"])
        md = smf.mixedlm("prevalence ~ trt*C(win)", x, groups=x.participant).fit()
        for k, name in (("trt:C(win)[T.stim]", "stim vs pre"), ("trt:C(win)[T.post]", "post vs pre")):
            ci = md.conf_int().loc[k]
            rows.append(dict(analysis=label, contrast=name, ROI=reg, beta=md.params[k], ci_lo=ci[0], ci_hi=ci[1], p=md.pvalues[k], n_part=x.participant.nunique(), n_rows=len(x)))
    out = pd.DataFrame(rows)
    for c, sub in out.groupby("contrast"):
        out.loc[sub.index, "q"] = AS.bh(sub.p.values)
    return out


def main():
    allc = pd.read_csv(T / "REV_redetect_cells.csv"); yb = pd.read_csv(T / "REV_redetect_cells_yasaB.csv")
    a = allc[allc.cohort == "Cohort A"]; b_nn = allc[allc.cohort == "Cohort B"]
    g_nn, g_y = AS.region_table(pd.concat([a, b_nn])), AS.region_table(pd.concat([a, yb]))
    g_nn = g_nn[g_nn.ROI.isin(REGS)]; g_y = g_y[g_y.ROI.isin(REGS)]
    out = pd.concat([fit(g_nn, "A + B (stream B staged by NeuroNet)"), fit(g_y, "A + B (stream B staged by YASA)"),
                     fit(g_nn[g_nn.participant.isin(b_nn.participant.unique())], "B only (NeuroNet)"),
                     fit(g_y[g_y.participant.isin(yb.participant.unique())], "B only (YASA)")], ignore_index=True)
    out.to_csv(T / "REV_staging_sensitivity.csv", index=False)
    pd.set_option("display.width", 250); print(out[out.contrast == "stim vs pre"].round(4).to_string(index=False))
    # N2 minutes per window under the two stagers (stream B)
    for lab, c in (("NeuroNet", b_nn), ("YASA", yb)):
        d = c.groupby(["session_id", "win"]).duration_sec.first().groupby("win").median() / 60
        print(lab, "median N2 min per window:", d.round(1).to_dict(), "| cells", len(c), "| sessions", c.session_id.nunique())


if __name__ == "__main__":
    main()
