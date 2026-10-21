# -*- coding: utf-8 -*-
"""On the final cohort definition: (1) within-participant crossover contrast, (2) within-night window contrast.

Reads only S_Table_session_metrics.csv produced by final_cohort.py.
"""
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
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if d.size < 4:
        return np.nan
    o = abs(d.mean())
    return float((np.abs((RNG.choice([-1., 1.], (n, d.size)) * d)
                         .mean(1)) >= o).mean())


def boot_mean(d, n=4000):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if d.size < 4:
        return np.nan, np.nan
    b = RNG.choice(d, (n, d.size), True).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fdr(p):
    p = np.asarray(p, float); o = np.full_like(p, np.nan)
    k = np.isfinite(p); m = int(k.sum())
    if m == 0:
        return o
    i = np.argsort(p[k])
    q = np.minimum.accumulate((p[k][i] * m / (np.arange(m) + 1))[::-1])[::-1].clip(0, 1)
    t = np.empty(m); t[i] = q; o[k] = t
    return o


def paired(a, b, label, **kw):
    """a, b = two series indexed by the same participants."""
    j = a.index.intersection(b.index)
    x, y = a.loc[j].astype(float), b.loc[j].astype(float)
    k = np.isfinite(x) & np.isfinite(y)
    d = (x[k] - y[k]).values
    if d.size < 4:
        return None
    lo, hi = boot_mean(d)
    return dict(contrast=label, n=int(d.size), mean_diff=float(d.mean()),
                ci_lo=lo, ci_hi=hi,
                dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) else np.nan,
                n_pos=int((d > 0).sum()),
                p_wilcoxon=float(stats.wilcoxon(d).pvalue) if d.size >= 6 else np.nan,
                p_perm=perm1(d), **kw)


def main():
    g = pd.read_csv(OUT / "S_Table_session_metrics.csv")

    # ─────────────────────────── (1) within-participant crossover contrast
    rows = []
    for det in DETS:
        for roi in ROIS:
            for win in ("stim", "post", "pre"):
                t = g[(g.detector == det) & (g.ROI == roi) & (g.win == win)]
                a = t[t.arm == "Treatment"].groupby("participant").prevalence.mean()
                b = t[t.arm == "Sham"].groupby("participant").prevalence.mean()
                r = paired(a, b, "Treatment − Sham", detector=det, ROI=roi,
                           window=win)
                if r:
                    rows.append(r)
    X = pd.DataFrame(rows)
    X["q"] = X.groupby(["detector", "window"]).p_perm.transform(fdr)
    X.to_csv(OUT / "S_Table_crossover.csv", index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("(1) 참가자 내 교차 대비 — 같은 사람의 Treatment 세션 vs Sham 세션")
    print("=" * 100)
    for win in ("stim", "post", "pre"):
        t = X[X.window == win]
        print(f"\n■ {win}   perm P<.05 : {int((t.p_perm < .05).sum())}/{len(t)}")
        print(t.sort_values("p_perm")[["detector", "ROI", "n", "mean_diff",
                                      "ci_lo", "ci_hi", "dz", "n_pos",
                                      "p_wilcoxon", "p_perm", "q"]]
              .round(4).to_string(index=False))

    # mixed model (session level, participant random intercept)
    print("\n" + "=" * 100)
    print("(1b) 선형혼합모형  prevalence ~ arm + (1|participant)")
    print("=" * 100)
    mm = []
    for det in DETS:
        for roi in ROIS:
            t = g[(g.detector == det) & (g.ROI == roi) & (g.win == "stim")].dropna(
                subset=["prevalence"]).copy()
            t["trt"] = (t.arm == "Treatment").astype(int)
            both = t.groupby("participant").trt.nunique()
            t = t[t.participant.isin(both[both == 2].index)]
            if t.participant.nunique() < 10:
                continue
            try:
                m = smf.mixedlm("prevalence ~ trt", t, groups=t.participant).fit()
                b, se = float(m.params["trt"]), float(m.bse["trt"])
                mm.append(dict(detector=det, ROI=roi,
                               n_participants=t.participant.nunique(),
                               n_sessions=len(t), beta=b,
                               ci_lo=b - 1.96 * se, ci_hi=b + 1.96 * se,
                               p=float(m.pvalues["trt"])))
            except Exception:
                pass
    M = pd.DataFrame(mm)
    if len(M):
        M["q"] = M.groupby("detector").p.transform(fdr)
        M.to_csv(OUT / "S_Table_crossover_lmm.csv", index=False,
                 encoding="utf-8-sig")
        print(M.sort_values("p").round(4).to_string(index=False))

    # ─────────────────────────── (2) within-night window contrast (Treatment sessions only)
    rows = []
    for det in DETS:
        for roi in ROIS:
            t = g[(g.detector == det) & (g.ROI == roi) & (g.arm == "Treatment")]
            w = {k: t[t.win == k].groupby("participant").prevalence.mean()
                 for k in ("pre", "stim", "post")}
            for lab, (x, y) in {"Stim − Pre": ("stim", "pre"),
                                "Post − Pre": ("post", "pre"),
                                "Post − Stim": ("post", "stim")}.items():
                r = paired(w[x], w[y], lab, detector=det, ROI=roi)
                if r:
                    rows.append(r)
    W = pd.DataFrame(rows)
    W["q"] = W.groupby(["detector", "contrast"]).p_perm.transform(fdr)
    W.to_csv(OUT / "S_Table_withinnight.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("(2) 야간 내 창 대비 — Treatment 세션만")
    print("=" * 100)
    for c in ("Stim − Pre", "Post − Pre", "Post − Stim"):
        t = W[W.contrast == c]
        print(f"\n■ {c}   perm P<.05 : {int((t.p_perm < .05).sum())}/{len(t)}")
        print(t.sort_values("p_perm")[["detector", "ROI", "n", "mean_diff",
                                      "ci_lo", "ci_hi", "dz", "n_pos",
                                      "p_wilcoxon", "p_perm", "q"]]
              .round(4).to_string(index=False))
    # ─────────────────────────── (3) channel-level within-night contrast (for topography)
    gc = pd.read_csv(OUT / "S_Table_channel_metrics.csv")
    rows = []
    for det in DETS:
        for ch in gc.ch.unique():
            t = gc[(gc.detector == det) & (gc.ch == ch)
                   & (gc.arm == "Treatment")]
            w = {k: t[t.win == k].groupby("participant").prevalence.mean()
                 for k in ("pre", "stim", "post")}
            for lab, (x, y) in {"Stim − Pre": ("stim", "pre"),
                                "Post − Pre": ("post", "pre"),
                                "Post − Stim": ("post", "stim")}.items():
                r = paired(w[x], w[y], lab, detector=det, ch=ch)
                if r:
                    rows.append(r)
    C = pd.DataFrame(rows)
    C["q"] = C.groupby(["detector", "contrast"]).p_perm.transform(fdr)
    C.to_csv(OUT / "S_Table_channel_withinnight.csv", index=False,
             encoding="utf-8-sig")
    print("\n" + "=" * 100)
    print("(3) 채널 수준 야간 내 대비 — P<.05 칸 수")
    print(C.groupby("contrast").apply(
        lambda d: f"{int((d.p_perm < .05).sum())}/{len(d)}").to_string())

    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
