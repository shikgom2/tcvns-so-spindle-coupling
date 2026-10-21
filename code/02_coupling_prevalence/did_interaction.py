# -*- coding: utf-8 -*-
"""DiD + Treatment x Phase interaction + visit-order sensitivity."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
import statsmodels.formula.api as smf                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402

P = LP.PROJECT_ROOT
OUT = P / "outputs/paper"
RNG = np.random.default_rng(20261003)
ROIS = ["Frontal", "Temporal", "Central", "Parietal"]
DETS = ["jy_N2", "jy_N2N3", "staresina_N2", "staresina_N2N3"]


def perm1(d, n=20000):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 4:
        return np.nan
    o = abs(d.mean())
    flips = RNG.choice([-1.0, 1.0], (n, d.size))
    return float((np.abs((flips * d).mean(1)) >= o).mean())


def boot(d, n=4000):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 4:
        return np.nan, np.nan
    b = RNG.choice(d, (n, d.size), True).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fdr(p):
    p = np.asarray(p, float)
    o = np.full_like(p, np.nan)
    k = np.isfinite(p)
    m = int(k.sum())
    if m == 0:
        return o
    i = np.argsort(p[k])
    q = np.minimum.accumulate((p[k][i] * m / (np.arange(m) + 1))[::-1])[::-1]
    t = np.empty(m)
    t[i] = q.clip(0, 1)
    o[k] = t
    return o


def did(g, late):
    rows = []
    for det in DETS:
        for roi in ROIS:
            t = g[(g.detector == det) & (g.ROI == roi)]
            w = (t.groupby(["participant", "arm", "win"]).prevalence.mean()
                 .unstack())
            if not set(["pre", late]).issubset(set(w.columns)):
                continue
            w = w.reset_index()
            w["delta"] = w[late] - w["pre"]
            d = w.pivot_table(index="participant", columns="arm",
                              values="delta").dropna()
            if "Treatment" not in d or "Sham" not in d or len(d) < 6:
                continue
            dd = (d["Treatment"] - d["Sham"]).values
            lo, hi = boot(dd)
            sd = dd.std(ddof=1)
            rows.append(dict(
                window=late, detector=det, ROI=roi, n=len(dd),
                mean_active=float(d["Treatment"].mean()),
                mean_sham=float(d["Sham"].mean()),
                DiD=float(dd.mean()), ci_lo=lo, ci_hi=hi,
                dz=float(dd.mean() / sd) if sd else np.nan,
                n_pos=int((dd > 0).sum()),
                p_wilcoxon=float(stats.wilcoxon(dd).pvalue),
                p_perm=perm1(dd)))
    return pd.DataFrame(rows)


def interaction(g):
    rows = []
    for det in DETS:
        for roi in ROIS:
            t = g[(g.detector == det) & (g.ROI == roi)
                  & g.win.isin(["pre", "stim", "post"])].dropna(
                subset=["prevalence"]).copy()
            t["arm_b"] = (t.arm == "Treatment").astype(int)
            t["win"] = pd.Categorical(t.win, ["pre", "stim", "post"])
            if t.participant.nunique() < 15 or t.arm_b.nunique() < 2:
                continue
            try:
                m = smf.mixedlm("prevalence ~ arm_b * C(win)", t,
                                groups=t.participant).fit()
            except Exception:
                continue
            pairs = [("arm_b:C(win)[T.stim]", "stim vs pre"),
                     ("arm_b:C(win)[T.post]", "post vs pre")]
            for term, lab in pairs:
                if term not in m.params.index:
                    continue
                b = float(m.params[term])
                se = float(m.bse[term])
                rows.append(dict(detector=det, ROI=roi, contrast=lab,
                                 n_participants=t.participant.nunique(),
                                 n_rows=len(t), beta=b,
                                 ci_lo=b - 1.96 * se, ci_hi=b + 1.96 * se,
                                 p=float(m.pvalues[term])))
    return pd.DataFrame(rows)


def order_sens(g):
    idm = pd.read_csv(OUT / "INTERNAL_session_idmap.csv")
    g2 = g.merge(idm[["session_id", "Order", "cohort"]], on="session_id",
                 how="left")
    rows = []
    for det in DETS:
        for roi in ROIS:
            t = g2[(g2.detector == det) & (g2.ROI == roi) & (g2.win == "pre")
                   & (g2.arm == "Treatment")].dropna(
                subset=["prevalence", "Order"])
            if t.participant.nunique() < 10:
                continue
            r, p = stats.spearmanr(t.Order, t.prevalence)
            rows.append(dict(detector=det, ROI=roi, n_rows=len(t),
                             rho=float(r), p=float(p)))
    return pd.DataFrame(rows)


def main():
    g = pd.read_csv(OUT / "S_Table_session_metrics.csv")

    D = pd.concat([did(g, "stim"), did(g, "post")], ignore_index=True)
    D["q"] = D.groupby(["window", "detector"]).p_perm.transform(
        lambda p: fdr(p.values))
    D.to_csv(OUT / "S_Table_DiD.csv", index=False, encoding="utf-8-sig")
    cols = ["detector", "ROI", "n", "mean_active", "mean_sham", "DiD",
            "ci_lo", "ci_hi", "dz", "n_pos", "p_wilcoxon", "p_perm", "q"]
    for wk in ("stim", "post"):
        t = D[D.window == wk]
        print("=" * 104)
        print("(1) Difference-in-differences  [%s - pre]" % wk)
        print("=" * 104)
        print(t.sort_values("p_perm")[cols].round(4).to_string(index=False))

    I = interaction(g)
    I["q"] = I.groupby(["detector", "contrast"]).p.transform(
        lambda p: fdr(p.values))
    I.to_csv(OUT / "S_Table_arm_x_phase.csv", index=False,
             encoding="utf-8-sig")
    print("\n" + "=" * 104)
    print("(2) Treatment x Phase interaction")
    print("=" * 104)
    print(I.sort_values("p")[
        ["detector", "ROI", "contrast", "n_participants", "n_rows",
         "beta", "ci_lo", "ci_hi", "p", "q"]].round(4).to_string(index=False))

    V = order_sens(g)
    V.to_csv(OUT / "S_Table_order_sensitivity.csv", index=False,
             encoding="utf-8-sig")
    print("\n" + "=" * 104)
    print("(3) Visit-order sensitivity (pre-stim window, treatment only)")
    print("=" * 104)
    print("   significant cells : %d/%d" % (int((V.p < .05).sum()), len(V)))
    print(V.sort_values("p").round(4).to_string(index=False))
    print("\n[saved] %s" % OUT)


if __name__ == "__main__":
    main()
