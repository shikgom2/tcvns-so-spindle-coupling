# -*- coding: utf-8 -*-
"""Final tables of the mechanism analysis — built in one place so that figures and text use the same numbers.

T1  properties of coupled vs uncoupled slow waves (per region)
T2  coupling probability as a function of spatial extent
T3  condition x window interaction — 6 measures x 4 regions x 2 contrasts + region average (pooled)
T4  auxiliary tests that could falsify the mechanistic claim (mediation, phase, spindle properties)

Detector is Staresina (0.30-1.25 Hz, 0.8-2.0 s, mean + 1 SD), stages N2+N3.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

R = Path(r"D:/SO-Spindle논문")
TBL = R / "Final analysis/tables"
ROIS = ["Frontal", "Temporal", "Central", "Parietal"]
RNG = np.random.default_rng(20261004)

MEAS = [("amp_all", "Slow-oscillation amplitude (uV)", "A"),
        ("slope_coupled", "Slope, coupled waves (uV/s)", "A"),
        ("slope_uncoupled", "Slope, uncoupled waves (uV/s)", "A"),
        ("amp_gap", "Amplitude, coupled - uncoupled (uV)", "A"),
        ("involve", "Derivations involved per wave", "G"),
        ("frac_global", "Fraction of widespread waves", "G")]


def _pid():
    F = pd.read_csv(TBL / "FIF_index.csv")[["Session ID", "Participant"]]
    F.columns = ["session", "participant"]
    return F


def load():
    F = _pid()
    A = pd.read_csv(TBL / "M1_amp_gating.csv")
    A = A[A.ROI.isin(ROIS)].merge(F, on="session")
    G = pd.read_csv(TBL / "M2_globality_recording.csv")
    G = G[G.ROI.isin(ROIS)].merge(F, on="session")
    return A, G


def ci(v, n=4000):
    v = np.asarray(v, float)
    bs = RNG.choice(v, (n, v.size), True).mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ----------------------------------------------------------------- T1
def t1(A):
    rows = []
    for roi in ROIS:
        g = A[A.ROI == roi]
        for col, lab in (("amp", "Peak-to-peak amplitude (uV)"),
                         ("slope", "Trough-to-peak slope (uV/s)")):
            c = g[col + "_coupled"].dropna()
            u = g[col + "_uncoupled"].dropna()
            d = (g[col + "_coupled"] - g[col + "_uncoupled"]).dropna()
            lo, hi = ci(d.values)
            rows.append(dict(
                Region=roi, Property=lab, n_cells=len(d),
                Coupled=round(c.mean(), 2), Uncoupled=round(u.mean(), 2),
                Difference=round(d.mean(), 2),
                CI="%.2f to %.2f" % (lo, hi),
                dz=round(d.mean() / d.std(ddof=1), 3),
                P=float(stats.wilcoxon(d).pvalue)))
    T = pd.DataFrame(rows)
    T.to_csv(TBL / "T_M1_coupled_vs_uncoupled.csv", index=False,
             encoding="utf-8-sig")
    return T


# ----------------------------------------------------------------- T2
def t2():
    I = pd.read_csv(TBL / "M2_involve_curve.csv")
    I = I[I.ROI.isin(ROIS)]
    w = (I.assign(num=I["mean"] * I["size"])
         .groupby("involve")[["num", "size"]].sum())
    w = w[w["size"] >= 200]
    p = w.num / w["size"]
    se = np.sqrt(p * (1 - p) / w["size"])
    T = pd.DataFrame(dict(
        Derivations=w.index.astype(int),
        Slow_oscillations=w["size"].astype(int).values,
        Coupling_pct=(p * 100).round(2).values,
        CI=["%.2f to %.2f" % (a, b) for a, b in
            zip((p - 1.96 * se) * 100, (p + 1.96 * se) * 100)]))
    rho, pv = stats.spearmanr(T.Derivations, T.Coupling_pct)
    T.attrs["rho"], T.attrs["p"] = rho, pv
    T.to_csv(TBL / "T_M2_coupling_by_extent.csv", index=False,
             encoding="utf-8-sig")
    return T


# ----------------------------------------------------------------- T3
def _fit(g, y):
    g = g.copy()
    g["win"] = pd.Categorical(g.win, ["pre", "stim", "post"])
    g["arm"] = pd.Categorical(g.arm, ["Sham", "Treatment"])
    m = smf.mixedlm("%s ~ C(arm)*C(win)" % y, g, groups=g.participant).fit()
    out = {}
    for k, c in (("stim", "C(arm)[T.Treatment]:C(win)[T.stim]"),
                 ("post", "C(arm)[T.Treatment]:C(win)[T.post]")):
        if c in m.params.index:
            out[k] = (float(m.params[c]), float(m.bse[c]),
                      float(m.pvalues[c]))
    return out, g.participant.nunique()


def t3(A, G):
    rows = []
    for y, lab, fam in MEAS:
        d = A if y in A.columns else G
        for roi in ROIS:
            g = d[d.ROI == roi].dropna(subset=[y])
            if g.participant.nunique() < 20:
                continue
            o, n = _fit(g, y)
            for k, (b, se, p) in o.items():
                rows.append(dict(Measure=lab, family=fam, Region=roi,
                                 Contrast=k, beta=b, se=se, P=p, n=n))
        pooled = (d.groupby(["session", "participant", "arm", "win"],
                            as_index=False)[y].mean().dropna(subset=[y]))
        o, n = _fit(pooled, y)
        for k, (b, se, p) in o.items():
            rows.append(dict(Measure=lab, family=fam, Region="All four",
                             Contrast=k, beta=b, se=se, P=p, n=n))
    T = pd.DataFrame(rows)
    T["ci_lo"] = T.beta - 1.96 * T.se
    T["ci_hi"] = T.beta + 1.96 * T.se
    T.to_csv(TBL / "T_M3_stimulation_effects.csv", index=False,
             encoding="utf-8-sig")
    return T


# ----------------------------------------------------------------- T4
def t4(G):
    rows = []
    # (a) mediation — does the change in spatial extent predict the change in coupling?
    for roi in ROIS + ["All four"]:
        d = G if roi == "All four" else G[G.ROI == roi]
        w = d.pivot_table(index=["session", "arm"], columns="win",
                          values=["involve", "prev"])
        if ("involve", "post") not in w:
            continue
        h = pd.DataFrame({
            "di": w[("involve", "post")] - w[("involve", "pre")],
            "dp": (w[("prev", "post")] - w[("prev", "pre")]) * 100
        }).dropna().reset_index()
        h = h[h.arm == "Treatment"]
        if len(h) < 20:
            continue
        rho, p = stats.spearmanr(h.di, h.dp)
        rows.append(dict(Test="Change in spatial extent predicts change in "
                              "coupling (after - before, stimulation only)",
                         Region=roi, n=len(h), Statistic="rho = %+.3f" % rho,
                         P=float(p)))
    # (b) phase precision
    P6 = pd.read_csv(TBL / "M6_coupling_phase.csv")
    S = P6[P6.win == "stim"]
    for roi in ROIS + ["All four"]:
        d = S if roi == "All four" else S[S.ROI == roi]
        g = d.groupby(["participant", "arm"]).ppc.mean().reset_index()
        a = g[g.arm == "Treatment"].ppc
        b = g[g.arm == "Sham"].ppc
        if len(a) < 15 or len(b) < 8:
            continue
        sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var())
                     / (len(a) + len(b) - 2))
        rows.append(dict(Test="Phase concentration of coupled spindles "
                              "(stimulation window, between conditions)",
                         Region=roi, n=len(a) + len(b),
                         Statistic="g = %+.2f" % ((a.mean() - b.mean()) / sp),
                         P=float(stats.mannwhitneyu(a, b).pvalue)))
    # (c) spindle properties
    SP = pd.read_csv(TBL / "M3_spindle_props.csv")
    SP = SP[SP.ROI.isin(ROIS)].merge(_pid(), on="session")
    for col, lab in (("dur", "Spindle duration (s)"),
                     ("rms", "Spindle envelope amplitude"),
                     ("ncyc", "Spindle cycle count")):
        g = (SP.groupby(["session", "participant", "arm", "win"],
                        as_index=False)[col].mean())
        o, n = _fit(g, col)
        b, se, p = o["post"]
        rows.append(dict(Test=lab + " (condition x window, after vs before)",
                         Region="All four", n=n,
                         Statistic="beta = %+.4f" % b, P=p))
    T = pd.DataFrame(rows)
    T.to_csv(TBL / "T_M4_falsification_tests.csv", index=False,
             encoding="utf-8-sig")
    return T


def run():
    A, G = load()
    T1, T2 = t1(A), t2()
    T3, T4 = t3(A, G), t4(G)
    print("--- T1 coupled vs uncoupled"); print(T1.to_string(index=False))
    print("\n--- T2 coupling by spatial extent  (rho=%+.2f, P=%.4f)"
          % (T2.attrs["rho"], T2.attrs["p"]))
    print(T2.to_string(index=False))
    print("\n--- T3 stimulation effects (pooled rows only)")
    print(T3[T3.Region == "All four"].round(4).to_string(index=False))
    print("\n  per-region sign counts")
    for m, g in T3[T3.Region != "All four"].groupby("Measure"):
        print("   %-38s %d/%d positive, min P=%.3f"
              % (m, int((g.beta > 0).sum()), len(g), g.P.min()))
    print("\n--- T4 falsification"); print(T4.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
