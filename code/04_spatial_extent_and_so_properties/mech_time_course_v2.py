# -*- coding: utf-8 -*-
"""Place stimulation → coupling → spatial extent on the same time axis.

For each recording, stimulation onset (the first slow wave of the stimulation block) is set to 0 and
events are grouped into 5-min bins; per bin, coupling prevalence, slow-wave spatial extent and the
fraction of widespread slow waves are computed in both conditions.
It then checks whether the coupling and spatial-extent time series move together within a recording
(correlation at lag 0 and ±1 bin).

Detector Staresina, N2+N3 (re-detected event table). Outputs go to Final analysis/tables/.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
import mech_so_properties as MP                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402

R = LP.PROJECT_ROOT
TBL = R / "Final analysis/tables"
SF = 125.0
BIN = 5.0                      # min
EDGES = np.arange(-15, 45.1, BIN)          # full 15/30/15 schedule: -15 … 45 min
MIN_SO = 10
ROIS = ["Frontal", "Temporal", "Central", "Parietal"]


def run():
    E, _ = MP.load()
    E = E[E.ROI.isin(ROIS)]
    G = MP.globality(E)
    G["t_min"] = G.trough / SF / 60.0
    on = (G[G.block == "stimulation 1"].groupby("file").t_min.min()
          .rename("t_on"))
    G = G.merge(on, on="file", how="inner")
    G["t_rel"] = G.t_min - G.t_on
    G["bin"] = pd.cut(G.t_rel, EDGES, labels=EDGES[:-1] + BIN / 2)
    G = G.dropna(subset=["bin"])
    G["bin"] = G.bin.astype(float)

    # recording × bin (regions pooled — to secure enough events)
    B = (G.groupby(["file", "session", "arm", "bin"])
         .agg(n_so=("coupled", "size"), prev=("coupled", "mean"),
              extent=("involve", "mean"), frac_wide=("global_so", "mean"))
         .reset_index())
    B = B[B.n_so >= MIN_SO]
    B.to_csv(TBL / "M7_time_course_bins_v2.csv", index=False,
             encoding="utf-8-sig")

    # per-condition bin mean ± sem, between-condition comparison per bin
    rows = []
    for b, g in B.groupby("bin"):
        for y in ("prev", "extent", "frac_wide"):
            a = g[g.arm == "Treatment"][y].values
            s = g[g.arm == "Sham"][y].values
            rows.append(dict(bin=b, measure=y,
                             m_stim=a.mean(), se_stim=a.std(ddof=1) / np.sqrt(a.size),
                             n_stim=a.size, m_sham=s.mean() if s.size else np.nan,
                             se_sham=s.std(ddof=1) / np.sqrt(s.size) if s.size > 1 else np.nan,
                             n_sham=s.size,
                             p=stats.mannwhitneyu(a, s).pvalue if (a.size > 4 and s.size > 4) else np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(TBL / "M7_time_course_summary_v2.csv", index=False,
             encoding="utf-8-sig")

    # within-recording correlation of the coupling and spatial-extent time series (lag -1, 0, +1 bin)
    lag_rows = []
    for (fl, arm), g in B.groupby(["file", "arm"]):
        g = g.sort_values("bin")
        if len(g) < 5:
            continue
        p = g.prev.values; x = g.extent.values
        for lag in (-1, 0, 1):
            if lag == 0:
                a, b_ = p, x
            elif lag > 0:          # does spatial extent lead? (extent[t] vs prev[t+1])
                a, b_ = p[lag:], x[:-lag]
            else:
                a, b_ = p[:lag], x[-lag:]
            if len(a) >= 4 and np.std(a) > 0 and np.std(b_) > 0:
                lag_rows.append(dict(file=fl, arm=arm, lag=lag,
                                     r=stats.spearmanr(a, b_).statistic))
    L = pd.DataFrame(lag_rows)
    L.to_csv(TBL / "M7_lag_correlation_v2.csv", index=False, encoding="utf-8-sig")

    print("recordings with onset:", G.file.nunique(), " bins:", len(B))
    print("\n--- 구간별 조건 평균 (prev %, extent)")
    for y in ("prev", "extent", "frac_wide"):
        t = S[S.measure == y]
        print(y)
        print(t[["bin", "m_stim", "n_stim", "m_sham", "n_sham", "p"]]
              .round(4).to_string(index=False))
    print("\n--- 녹화 내 결합-공간범위 상관 (Spearman), 조건·지연별")
    for (arm, lag), g in L.groupby(["arm", "lag"]):
        w = stats.wilcoxon(g.r.dropna()).pvalue if len(g) > 5 else np.nan
        print("  %-9s lag %+d  n=%3d  median r=%+.3f  mean r=%+.3f  P=%.4f"
              % (arm, lag, len(g), g.r.median(), g.r.mean(), w))


if __name__ == "__main__":
    run()
