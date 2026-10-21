# -*- coding: utf-8 -*-
"""Spatial extent under window-specific and recording-wide thresholds: statistics (Results: spatial extent; Supplementary Table 3).

Input   REV_extent_thresholds_cells.csv, REV_extent_thresholds_session.csv  (from rev_extent_thresholds.py)
Output  REV_extent_thresholds.csv        coupling probability by extent level (Spearman), condition x window interaction
                                         for mean extent and fraction widespread, within-stimulation-recording change (Wilcoxon)
        REV_extent_thresholds_curve.csv  coupling probability per extent level and scope
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402

warnings.filterwarnings("ignore")
T = LP.PROJECT_ROOT / "Final analysis/tables"


def main():
    c = pd.read_csv(T / "REV_extent_thresholds_cells.csv"); s = pd.read_csv(T / "REV_extent_thresholds_session.csv")
    rows = []
    for sc in ("window-specific", "recording-wide"):
        x = c[c.scope == sc]; g = x.groupby("extent")[["n_SO", "n_coupled"]].sum(); g = g[g.n_SO >= 200]; g["prob"] = g.n_coupled / g.n_SO
        rho, p = stats.spearmanr(g.index, g.prob)
        ss = s[s.scope == sc].copy(); ss["trt"] = (ss.arm == "Treatment").astype(int); ss["win"] = pd.Categorical(ss.win, ["pre", "stim", "post"])
        for y in ("mean_extent", "frac_widespread"):
            md = smf.mixedlm("%s ~ trt*C(win)" % y, ss, groups=ss.participant).fit()
            for k, name in (("trt:C(win)[T.stim]", "stim vs pre"), ("trt:C(win)[T.post]", "post vs pre")):
                ci = md.conf_int().loc[k]
                rows.append(dict(scope=sc, outcome=y, contrast=name, beta=md.params[k], ci_lo=ci[0], ci_hi=ci[1], p=md.pvalues[k], n_part=ss.participant.nunique()))
            w = ss[ss.arm == "Treatment"].pivot_table(index="participant", columns="win", values=y)
            for wn in ("stim", "post"):
                d = (w[wn] - w["pre"]).dropna()
                rows.append(dict(scope=sc, outcome=y, contrast="within stimulation recordings: %s - pre" % wn, beta=d.mean(), ci_lo=np.nan, ci_hi=np.nan,
                                 p=stats.wilcoxon(d).pvalue, n_part=len(d)))
        rows.append(dict(scope=sc, outcome="coupling_prob_vs_extent", contrast="Spearman over extent levels", beta=rho, ci_lo=np.nan, ci_hi=np.nan, p=p, n_part=len(g)))
        rows.append(dict(scope=sc, outcome="coupling_prob_extent1", contrast="level", beta=g.prob.iloc[0], ci_lo=np.nan, ci_hi=np.nan, p=np.nan, n_part=int(g.n_SO.iloc[0])))
        rows.append(dict(scope=sc, outcome="coupling_prob_extent_max", contrast="level at extent %d" % g.index[-1], beta=g.prob.iloc[-1], ci_lo=np.nan, ci_hi=np.nan, p=np.nan, n_part=int(g.n_SO.iloc[-1])))
    pd.DataFrame(rows).to_csv(T / "REV_extent_thresholds.csv", index=False)
    gc = c.groupby(["scope", "extent"])[["n_SO", "n_coupled"]].sum().reset_index(); gc["prob"] = gc.n_coupled / gc.n_SO
    gc.to_csv(T / "REV_extent_thresholds_curve.csv", index=False); print("saved")


if __name__ == "__main__":
    main()
