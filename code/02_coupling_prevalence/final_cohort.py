# -*- coding: utf-8 -*-
"""Full re-aggregation under the final cohort definition + assignment of T/S session IDs.

Treatment = TB stimulation visits + IIT treatment-arm sessions
Sham      = TB sham-condition night + IIT baseline sessions + IIT sham-arm sessions

Session ID : Treatment = T001…  /  Sham = S001…
The word 'washout' is not used anywhere in the outputs.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
import statsmodels.api as sm                                     # noqa: E402
import statsmodels.formula.api as smf                            # noqa: E402

P = Path(r"D:/SO-Spindle논문")
OUT = P / "outputs/paper"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261003)

ROIS = {"Frontal": ["Fp1", "Fp2", "F3", "F4"],
        "Temporal": ["F7", "F8", "T3", "T4"],
        "Central": ["C3", "Cz", "C4"], "Parietal": ["P3", "P4"]}
CH_ROI = {c: r for r, cs in ROIS.items() for c in cs}
DET = {"jy_N2": P / "redetect_20261002/REDETECT_summary.parquet",
       "jy_N2N3": P / "redetect_multi/jy_N2N3.parquet",
       "staresina_N2": P / "redetect_multi/staresina_N2.parquet",
       "staresina_N2N3": P / "redetect_multi/staresina_N2N3.parquet"}
BLK = {"baseline": "pre", "stimulation 1": "stim", "recovery 1": "post"}


def assign_arm(row):
    """Final arm assignment."""
    if row.cohort == "TB":
        return "Treatment" if row.Order <= 3 else "Sham"
    if row.Order == 1:
        return "Sham"                       # IIT baseline sessions
    return "Sham" if row.DB_Group == "SHAM" else "Treatment"


def load(det):
    d = pd.read_parquet(DET[det])
    d["ID"] = d["원ID"].astype(str)
    d["Order"] = pd.to_numeric(d["Order"], errors="coerce")
    d = d[d.Order.notna()].copy()
    d["Order"] = d.Order.astype(int)
    d["ROI"] = d.ch.map(CH_ROI)
    d = d[d.ROI.notna() & d.block.isin(BLK)]
    d["arm"] = d.apply(assign_arm, axis=1)
    d["win"] = d.block.map(BLK)
    d["detector"] = det
    return d.drop(columns=[c for c in ("session_id", "participant",
                                       "cohort_label") if c in d.columns])


def perm(d, n=20000):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if d.size < 4:
        return np.nan
    o = abs(d.mean())
    return float((np.abs((RNG.choice([-1., 1.], (n, d.size)) * d)
                         .mean(1)) >= o).mean())


def hedges_g(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 3 or ny < 3:
        return np.nan
    s = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1))
                / (nx + ny - 2))
    return (x.mean() - y.mean()) / s * (1 - 3 / (4 * (nx + ny) - 9)) if s else np.nan


def boot_g(x, y, n=4000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan
    b = np.array([hedges_g(RNG.choice(x, len(x), True),
                           RNG.choice(y, len(y), True)) for _ in range(n)])
    b = b[np.isfinite(b)]
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))) \
        if b.size > 100 else (np.nan, np.nan)


def fdr(p):
    p = np.asarray(p, float)
    o = np.full_like(p, np.nan)
    k = np.isfinite(p); m = int(k.sum())
    if m == 0:
        return o
    i = np.argsort(p[k])
    q = np.minimum.accumulate((p[k][i] * m / (np.arange(m) + 1))[::-1])[::-1].clip(0, 1)
    t = np.empty(m); t[i] = q
    o[k] = t
    return o


def main():
    A = pd.concat([load(k) for k in DET], ignore_index=True)

    # ── Assign session IDs (T/S)
    ses = (A.drop_duplicates(["cohort", "ID", "Order"])
           [["cohort", "ID", "Order", "arm"]]
           .sort_values(["arm", "cohort", "ID", "Order"]).reset_index(drop=True))
    pid = {p: f"P{n:02d}" for n, p in enumerate(
        sorted(ses.ID.unique(), key=lambda s: (s[0], s)), 1)}
    ses["participant"] = ses.ID.map(pid)
    ses["session_id"] = ""
    for arm, pre in (("Treatment", "T"), ("Sham", "S")):
        m = ses.arm == arm
        ses.loc[m, "session_id"] = [f"{pre}{i:03d}"
                                    for i in range(1, int(m.sum()) + 1)]
    ses["cohort_label"] = ses.cohort.map({"TB": "Cohort A", "IIT": "Cohort B"})
    ses[["session_id", "participant", "arm", "cohort_label"]].to_csv(
        OUT / "S_Table_session_inventory.csv", index=False, encoding="utf-8-sig")
    # Internal lookup table that keeps the original IDs (for compiling the submission file list; not supplementary material)
    ses[["session_id", "participant", "arm", "cohort_label",
         "cohort", "ID", "Order"]].to_csv(
        OUT / "INTERNAL_session_idmap.csv", index=False, encoding="utf-8-sig")

    A = A.merge(ses[["cohort", "ID", "Order", "participant", "session_id",
                     "cohort_label"]], on=["cohort", "ID", "Order"])

    # ── ROI aggregation
    g = A.groupby(["detector", "participant", "session_id", "cohort_label",
                   "arm", "win", "ROI"], as_index=False).agg(
        n_SO=("n_SO", "sum"), n_coupled=("n_coupled", "sum"),
        duration_sec=("duration_sec", "first"))
    g["prevalence"] = g.n_coupled / g.n_SO.replace(0, np.nan)
    g.to_csv(OUT / "S_Table_session_metrics.csv", index=False,
             encoding="utf-8-sig")

    # ── Channel level (for topography)
    gc = A.groupby(["detector", "participant", "session_id", "arm", "win",
                    "ch"], as_index=False).agg(
        n_SO=("n_SO", "sum"), n_coupled=("n_coupled", "sum"))
    gc["prevalence"] = gc.n_coupled / gc.n_SO.replace(0, np.nan)
    gc.to_csv(OUT / "S_Table_channel_metrics.csv", index=False,
              encoding="utf-8-sig")

    # ── Cohort table
    print("=" * 92)
    print("최종 코호트")
    print("=" * 92)
    print(ses.groupby(["arm", "cohort_label"]).agg(
        participants=("participant", "nunique"),
        sessions=("session_id", "size")).to_string())
    print(f"\n  Treatment {ses[ses.arm=='Treatment'].participant.nunique()}명 / "
          f"{(ses.arm=='Treatment').sum()}세션")
    print(f"  Sham      {ses[ses.arm=='Sham'].participant.nunique()}명 / "
          f"{(ses.arm=='Sham').sum()}세션")

    print("\n■ 창별 투입")
    w = g[g.detector == "jy_N2"].drop_duplicates(["session_id", "win"])
    print(w.groupby(["win", "arm"]).agg(
        participants=("participant", "nunique"),
        sessions=("session_id", "size")).to_string())

    # ── Between-condition tests (participant level + event level)
    rows = []
    for det in DET:
        for win in ("stim", "post", "pre"):
            for roi in ROIS:
                t = g[(g.detector == det) & (g.win == win) & (g.ROI == roi)]
                a = t[t.arm == "Treatment"].groupby("participant").prevalence.mean()
                b = t[t.arm == "Sham"].groupby("participant").prevalence.mean()
                a, b = a.dropna(), b.dropna()
                if len(a) < 6 or len(b) < 6:
                    continue
                gg = hedges_g(a.values, b.values)
                lo, hi = boot_g(a.values, b.values)
                pm = float(stats.mannwhitneyu(a.values, b.values,
                                              alternative="two-sided").pvalue)
                pw = float(stats.ttest_ind(a.values, b.values,
                                           equal_var=False).pvalue)
                # Event-level binomial GEE
                u = t[t.n_SO > 0].copy()
                orv = lo2 = hi2 = pg = np.nan
                if u.participant.nunique() >= 8:
                    u["arm_bin"] = (u.arm == "Treatment").astype(int)
                    u["fail"] = u.n_SO - u.n_coupled
                    try:
                        m = smf.gee("n_coupled + fail ~ arm_bin",
                                    groups="participant", data=u,
                                    family=sm.families.Binomial()).fit()
                        bb = float(m.params["arm_bin"])
                        se = float(m.bse["arm_bin"])
                        if abs(bb) < 5 and se > 1e-8:
                            orv = float(np.exp(bb))
                            lo2, hi2 = float(np.exp(bb - 1.96 * se)), float(np.exp(bb + 1.96 * se))
                            pg = float(m.pvalues["arm_bin"])
                    except Exception:
                        pass
                rows.append(dict(detector=det, window=win, ROI=roi,
                                 n_trt=len(a), n_sham=len(b),
                                 mean_trt=a.mean(), mean_sham=b.mean(),
                                 g=gg, g_lo=lo, g_hi=hi, p_mwu=pm, p_welch=pw,
                                 OR=orv, OR_lo=lo2, OR_hi=hi2, p_gee=pg,
                                 n_SO=int(u.n_SO.sum())))
    r = pd.DataFrame(rows)
    r["q_mwu"] = r.groupby(["detector", "window"]).p_mwu.transform(fdr)
    r["q_gee"] = r.groupby(["detector", "window"]).p_gee.transform(fdr)
    r.to_csv(OUT / "S_Table_between_condition.csv", index=False,
             encoding="utf-8-sig")

    print("\n" + "=" * 92)
    print("군간 — Treatment vs Sham, 결합비율")
    print("=" * 92)
    for win in ("stim", "post", "pre"):
        t = r[r.window == win]
        print(f"\n■ {win}  (MWU p<.05 {int((t.p_mwu<.05).sum())}/{len(t)} · "
              f"GEE p<.05 {int((t.p_gee<.05).sum())}/{len(t)})")
        s = t[(t.p_mwu < .05) | (t.p_gee < .05)].sort_values("p_mwu")
        if len(s):
            print(s[["detector", "ROI", "n_trt", "n_sham", "mean_trt",
                     "mean_sham", "g", "g_lo", "g_hi", "p_mwu", "q_mwu",
                     "OR", "OR_lo", "OR_hi", "p_gee", "q_gee", "n_SO"]]
                  .round(4).to_string(index=False))
        else:
            print("   유의 칸 없음")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
