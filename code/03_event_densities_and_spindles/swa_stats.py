# -*- coding: utf-8 -*-
"""SWA statistics — same estimators and same correction as the coupling analysis.

  Primary: log SWA ~ condition x window + (1|participant), per region, BH correction across the four regions
  Auxiliary: between-condition comparison in the stimulation window (participant level, Mann-Whitney + Hedges g)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

R = Path(r"D:/SO-Spindle논문/Final analysis")
T = pd.read_csv(R / "tables/SWA_windows.csv")
OUT = R / "tables/SWA_stats.csv"
REGS = ["Frontal", "Temporal", "Central", "Parietal"]
KEY = {"Frontal": "fron", "Temporal": "temp", "Central": "cent", "Parietal": "pari"}
BANDS = ["swa", "so_band", "delta", "sigma"]


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    m = len(p)
    prev = 1.0
    for rank, i in enumerate(o[::-1], 1):
        prev = min(prev, p[i] * m / (m - rank + 1))
        q[i] = prev
    return q


def hedges(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp
    return d * (1 - 3 / (4 * (na + nb) - 9))


def main():
    t = T.copy()
    t["win"] = pd.Categorical(t.win, ["pre", "stim", "post"])
    t["arm_b"] = (t.arm == "Treatment").astype(int)
    rows = []
    for band in BANDS:
        for reg in REGS:
            col = "%s_%s" % (KEY[reg], band)
            if col not in t:
                continue
            d = t.dropna(subset=[col]).copy()
            d["y"] = np.log(d[col].clip(lower=1e-6))
            md = smf.mixedlm("y ~ arm_b * C(win)", d, groups=d.participant).fit()
            for term, lab in (("arm_b:C(win)[T.stim]", "stim vs pre"),
                              ("arm_b:C(win)[T.post]", "post vs pre")):
                if term in md.params:
                    rows.append(dict(band=band, ROI=reg, contrast=lab,
                                     beta_log=md.params[term], p=md.pvalues[term],
                                     pct=(np.exp(md.params[term]) - 1) * 100,
                                     n_part=d.participant.nunique(), n_rows=len(d)))
    r = pd.DataFrame(rows)
    for (band, contr), sub in r.groupby(["band", "contrast"]):
        r.loc[sub.index, "q"] = bh(sub.p.values)
    r.to_csv(OUT, index=False)

    print("=== condition x window interaction on log power (BH across 4 regions) ===")
    for band in BANDS:
        print("\n--", band)
        s = r[(r.band == band)]
        for _, x in s.iterrows():
            flag = " *" if x.q < .05 else ""
            print("   %-12s %-11s  %+6.1f %%   P = %.3f   q = %.3f%s"
                  % (x.ROI, x.contrast, x.pct, x.p, x.q, flag))

    print("\n=== stimulation window, between conditions (participant level) ===")
    s = t[t.win == "stim"]
    for band in BANDS:
        print("\n--", band)
        for reg in REGS:
            col = "%s_%s" % (KEY[reg], band)
            a = s[s.arm == "Treatment"].groupby("participant")[col].mean().dropna()
            b = s[s.arm == "Sham"].groupby("participant")[col].mean().dropna()
            u = stats.mannwhitneyu(a, b)
            print("   %-9s  %8.1f vs %8.1f uV2/Hz   g = %+.2f   P = %.3f   (n=%d/%d)"
                  % (reg, a.median(), b.median(), hedges(np.log(a), np.log(b)),
                     u.pvalue, len(a), len(b)))
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
