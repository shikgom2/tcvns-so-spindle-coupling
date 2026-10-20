# -*- coding: utf-8 -*-
"""Alternative-explanation tests — those possible from the tables alone (statistics for candidate Figure 6 panels).

 A  denominator problem: condition × window for coupled / uncoupled / total SO density (S8)
 B  starting-point (baseline) adjustment: during ~ arm + pre (participant intercept), per region
    + baseline-tertile stratification + baseline–change correlation within sham (regression-to-the-mean index)
 C  visit-order covariate
 D  recruitment-stream (cohort) stratification
 E  detector × sleep-stage grid (direction, size, q)
 F  does the N3 share differ by window and condition (S1 duration)
Output: Final analysis/tables/ALT_*.csv and console summary
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

R = Path(r"D:/SO-Spindle논문")
OLD = R / "이전/outputs/paper/S_Table_session_metrics.csv"
S1 = R / "Final analysis/manuscript/supplementary materials/tables/S1_session_metrics.csv"
S8 = R / "Final analysis/manuscript/supplementary materials/tables/S8_event_densities.csv"
INV = R / "Final analysis/tables/FIF_index.csv"
OUT = R / "Final analysis/tables"
ROIS = ["Frontal", "Temporal", "Central", "Parietal"]


def fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.empty(n)
    r[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(r, 0, 1)


def load():
    g = pd.read_csv(OLD)
    inv = pd.read_csv(INV)[["Session ID", "Visit order", "Cohort"]]
    inv.columns = ["session_id", "visit", "cohort"]
    g = g.merge(inv, on="session_id", how="left")
    return g


def wide(g, det):
    t = g[g.detector == det]
    w = (t.pivot_table(index=["participant", "session_id", "arm", "ROI",
                              "cohort", "visit"],
                       columns="win", values="prevalence").reset_index())
    return w


# ────────────────────────────────────────────── A  denominator
def part_a():
    s8 = pd.read_csv(S8)
    s8 = s8[(s8.stage == "N2")]
    t = s8[s8.contrast == "stim vs pre"].pivot_table(
        index="ROI", columns="outcome", values=["beta", "p"])
    print("\n[A] condition × window (stim vs pre), N2 — β per outcome")
    print(t["beta"].reindex(ROIS).round(4).to_string())
    print(t["p"].reindex(ROIS).round(4).to_string())
    s8.to_csv(OUT / "ALT_A_densities_N2.csv", index=False)


# ────────────────────────────────────────────── B  baseline
def part_b(g, det="staresina_N2"):
    w = wide(g, det).dropna(subset=["pre", "stim"])
    w["d"] = w.stim - w.pre
    rows = []
    for roi in ROIS:
        t = w[w.ROI == roi].copy()
        t["arm01"] = (t.arm == "Treatment").astype(int)
        # ANCOVA-type mixed model: during ~ arm + pre, participant intercept
        m = smf.mixedlm("stim ~ arm01 + pre", t, groups=t.participant).fit(reml=False)
        # unadjusted (change score) for comparison
        m0 = smf.mixedlm("d ~ arm01", t, groups=t.participant).fit(reml=False)
        # sham: baseline–change correlation (regression to the mean signature)
        sh = t[t.arm == "Sham"]; tr = t[t.arm == "Treatment"]
        r_sh = stats.spearmanr(sh.pre, sh.d); r_tr = stats.spearmanr(tr.pre, tr.d)
        rows.append(dict(ROI=roi, n_rec=len(t), n_stim=len(tr), n_sham=len(sh),
                         beta_change=m0.params["arm01"], p_change=m0.pvalues["arm01"],
                         beta_adj=m.params["arm01"], se_adj=m.bse["arm01"],
                         p_adj=m.pvalues["arm01"], beta_pre=m.params["pre"],
                         rho_pre_change_sham=r_sh[0], p_sham=r_sh[1],
                         rho_pre_change_stim=r_tr[0], p_stim=r_tr[1]))
    B = pd.DataFrame(rows); B["q_adj"] = fdr(B.p_adj)
    print("\n[B] baseline-adjusted condition effect on DURING prevalence (N2)")
    print(B.round(4).to_string(index=False))
    B.to_csv(OUT / "ALT_B_baseline_adjusted.csv", index=False)

    # tertile stratification (pooled across regions, per region)
    rows = []
    for roi in ROIS:
        t = w[w.ROI == roi].copy()
        t["tert"] = pd.qcut(t.pre.rank(method="first"), 3, labels=["low", "mid", "high"])
        for k, s in t.groupby("tert", observed=True):
            a = s[s.arm == "Treatment"].d; b = s[s.arm == "Sham"].d
            p = stats.mannwhitneyu(a, b).pvalue if len(a) > 3 and len(b) > 3 else np.nan
            rows.append(dict(ROI=roi, tertile=k, n_stim=len(a), n_sham=len(b),
                             change_stim=a.mean() * 100, change_sham=b.mean() * 100,
                             diff_pp=(a.mean() - b.mean()) * 100, p=p))
    T = pd.DataFrame(rows)
    print("\n[B2] during−pre change (pp) by baseline tertile")
    print(T.round(3).to_string(index=False))
    T.to_csv(OUT / "ALT_B2_baseline_tertiles.csv", index=False)
    return w


# ────────────────────────────────────────────── C  visit order
def part_c(g, det="staresina_N2"):
    t = g[(g.detector == det) & g.win.isin(["pre", "stim"])].copy()
    t["arm01"] = (t.arm == "Treatment").astype(int)
    t["w01"] = (t.win == "stim").astype(int)
    rows = []
    for roi in ROIS:
        s = t[t.ROI == roi].dropna(subset=["prevalence", "visit"])
        m0 = smf.mixedlm("prevalence ~ arm01 * w01", s, groups=s.participant).fit(reml=False)
        m1 = smf.mixedlm("prevalence ~ arm01 * w01 + visit", s, groups=s.participant).fit(reml=False)
        rows.append(dict(ROI=roi, beta_int=m0.params["arm01:w01"], p_int=m0.pvalues["arm01:w01"],
                         beta_int_visitadj=m1.params["arm01:w01"], p_int_visitadj=m1.pvalues["arm01:w01"],
                         beta_visit=m1.params["visit"], p_visit=m1.pvalues["visit"]))
    C = pd.DataFrame(rows)
    print("\n[C] interaction with and without visit order as covariate")
    print(C.round(4).to_string(index=False))
    C.to_csv(OUT / "ALT_C_visit_order.csv", index=False)


# ────────────────────────────────────────────── D  cohort
def part_d(g, det="staresina_N2"):
    t = g[(g.detector == det) & g.win.isin(["pre", "stim"])].copy()
    t["arm01"] = (t.arm == "Treatment").astype(int)
    t["w01"] = (t.win == "stim").astype(int)
    rows = []
    for roi in ROIS:
        s = t[t.ROI == roi].dropna(subset=["prevalence"])
        m1 = smf.mixedlm("prevalence ~ arm01 * w01 + C(cohort)", s, groups=s.participant).fit(reml=False)
        sA = s[s.cohort == "Cohort A"]
        mA = smf.mixedlm("prevalence ~ arm01 * w01", sA, groups=sA.participant).fit(reml=False)
        rows.append(dict(ROI=roi, beta_int_cohortadj=m1.params["arm01:w01"], p_cohortadj=m1.pvalues["arm01:w01"],
                         beta_int_cohortA_only=mA.params["arm01:w01"], p_cohortA=mA.pvalues["arm01:w01"],
                         n_A=sA.participant.nunique()))
    D = pd.DataFrame(rows)
    print("\n[D] interaction adjusted for recruitment stream, and within stream A only")
    print(D.round(4).to_string(index=False))
    D.to_csv(OUT / "ALT_D_cohort.csv", index=False)


# ────────────────────────────────────────────── E  detector grid
def part_e(g):
    t = g[g.win.isin(["pre", "stim"])].copy()
    t["arm01"] = (t.arm == "Treatment").astype(int)
    t["w01"] = (t.win == "stim").astype(int)
    rows = []
    for det in sorted(t.detector.unique()):
        ps = []
        for roi in ROIS:
            s = t[(t.detector == det) & (t.ROI == roi)].dropna(subset=["prevalence"])
            m = smf.mixedlm("prevalence ~ arm01 * w01", s, groups=s.participant).fit(reml=False)
            rows.append(dict(detector=det, ROI=roi, beta=m.params["arm01:w01"], p=m.pvalues["arm01:w01"]))
    E = pd.DataFrame(rows)
    E["q"] = E.groupby("detector").p.transform(fdr)
    print("\n[E] detector × stage grid, stim-vs-pre interaction")
    print(E.pivot_table(index="ROI", columns="detector", values="beta").reindex(ROIS).round(4).to_string())
    print(E.pivot_table(index="ROI", columns="detector", values="q").reindex(ROIS).round(3).to_string())
    E.to_csv(OUT / "ALT_E_detector_grid.csv", index=False)


# ────────────────────────────────────────────── F  N3 share
def part_f():
    s1 = pd.read_csv(S1)
    d = s1[s1.ROI == "Frontal"].pivot_table(index=["session_id", "arm", "win"],
                                            columns="stage", values="duration_sec").reset_index()
    d["n3_share"] = 1 - d["N2"] / d["N2+N3"]
    d = d.dropna(subset=["n3_share"])
    summ = d.groupby(["arm", "win"]).n3_share.agg(["mean", "median", "count"])
    print("\n[F] share of N3 within qualifying sleep, by condition × window")
    print(summ.round(3).to_string())
    w = d.pivot_table(index=["session_id", "arm"], columns="win", values="n3_share").reset_index()
    w["d"] = w.stim - w.pre; w = w.dropna(subset=["d"])
    a = w[w.arm == "Treatment"].d; b = w[w.arm == "Sham"].d
    print("  Δ N3 share (during−before): stim %+.3f (n=%d, P=%.3f) | sham %+.3f (n=%d, P=%.3f) | between P=%.3f"
          % (a.mean(), len(a), stats.wilcoxon(a).pvalue if len(a) else np.nan,
             b.mean(), len(b), stats.wilcoxon(b).pvalue if len(b) else np.nan,
             stats.mannwhitneyu(a, b).pvalue))
    d.to_csv(OUT / "ALT_F_n3_share.csv", index=False)


if __name__ == "__main__":
    g = load()
    part_a()
    part_b(g)
    part_c(g)
    part_d(g)
    part_e(g)
    part_f()
