# -*- coding: utf-8 -*-
"""Sensitivity of the primary interaction (Sensitivity analyses; Supplementary Table 4, parts 'model specification' and
'detection and criterion').

Part A  model specification: primary replication, random slope for window, stream-specific models
        (from the session-level metrics, N2, window-specific thresholds)  -> REV_primary_sensitivity.csv, REV_T35a_model_specifications.csv
Part B  detection and criterion: window-specific thresholds (same cells), recording-wide thresholds, chance-corrected prevalence,
        opposite half-cycle (from REV_interaction_sensitivity.csv written by rev_analysis.py) -> REV_T35b_detection_and_criterion.csv
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
S1 = LP.PROJECT_ROOT / "Final analysis/Brain submission/supplementary tables/Supplementary_Table_01_S1_session_metrics.csv"
REGS = ["Frontal", "Temporal", "Central", "Parietal"]


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty_like(p); m = len(p); prev = 1.0
    for rank, i in enumerate(o[::-1], 1):
        prev = min(prev, p[i] * m / (m - rank + 1)); q[i] = prev
    return q


def fit(sub, label, re_formula=None):
    out = []
    for reg in REGS:
        x = sub[sub.ROI == reg].dropna(subset=["prevalence"])
        try:
            md = smf.mixedlm("prevalence ~ trt*C(win)", x, groups=x.participant, re_formula=re_formula).fit(reml=True)
            k = "trt:C(win)[T.stim]"; b, p, se, conv = md.params[k], md.pvalues[k], md.bse[k], md.converged
        except Exception as e:                                    # noqa: BLE001
            b = p = se = np.nan; conv = str(e)[:40]
        out.append(dict(analysis=label, ROI=reg, n_part=x.participant.nunique(), n_rows=len(x), beta=b, se=se, p=p, conv=conv))
    o = pd.DataFrame(out); o["q"] = bh(o.p.values); return o


def main():
    s1 = pd.read_csv(S1)
    inv = pd.read_csv(T / "FIF_index.csv").rename(columns={"Session ID": "session_id", "Participant": "participant", "Cohort": "cohort", "Visit order": "visit"})
    d = s1[s1.stage == "N2"].merge(inv[["session_id", "participant", "cohort", "visit"]], on="session_id", how="left")
    d["trt"] = (d.arm == "Treatment").astype(int); d["win"] = pd.Categorical(d.win, ["pre", "stim", "post"])
    R = pd.concat([fit(d, "primary (replication)"), fit(d, "random slopes (window|participant)", re_formula="~C(win)"),
                   fit(d[d.cohort == "Cohort A"], "Cohort A only"), fit(d[d.cohort == "Cohort B"], "Cohort B only")])
    R.to_csv(T / "REV_primary_sensitivity.csv", index=False)
    ps = R[R.analysis.isin(["primary (replication)", "random slopes (window|participant)", "Cohort A only", "Cohort B only"])].copy()
    ps["analysis"] = ps.analysis.replace({"primary (replication)": "primary model", "random slopes (window|participant)": "random slope for window",
                                          "Cohort A only": "stream A only", "Cohort B only": "stream B only"})
    ps.drop(columns=["conv"]).to_csv(T / "REV_T35a_model_specifications.csv", index=False)
    si = pd.read_csv(T / "REV_interaction_sensitivity.csv")
    si = si[si.analysis.isin(["window-specific thresholds (primary)", "recording-wide thresholds", "chance-corrected (observed - surrogate)", "opposite phase convention"])].copy()
    si["analysis"] = si.analysis.replace({"window-specific thresholds (primary)": "window-specific thresholds (same cells)",
                                          "chance-corrected (observed - surrogate)": "chance-corrected prevalence", "opposite phase convention": "opposite half-cycle"})
    si.drop(columns=[c for c in ("note",) if c in si]).to_csv(T / "REV_T35b_detection_and_criterion.csv", index=False)
    print(ps.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
