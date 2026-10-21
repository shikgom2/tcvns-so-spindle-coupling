# -*- coding: utf-8 -*-
"""Stimulation-specificity tests — are the time-aligned changes in coupling and spatial extent tied to stimulation?

T1  condition x phase interaction (aligned to stimulation onset, per recording, mixed model) — coupling vs spatial extent
T2  time to effect: paired change per 5-min bin relative to the pre-stimulation mean in stimulation recordings (when does it rise)
T3  crossover participants (n = 42): difference of (during − before) differences between the same person's stimulation vs no-stimulation night
T4  whether the coupling increase arises in local or widespread slow oscillations (within-recording during − before, by condition)

Input: M7_time_course_bins.csv and the re-detection event table. Output: M8_*.csv
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
import mech_so_properties as MP                                    # noqa: E402

R = LP.PROJECT_ROOT
TBL = R / "Final analysis/tables"
SF = 125.0
BIN = 5.0
EDGES = np.arange(-15, 35.1, BIN)
STIM_END = 20.0


def pid_map():
    F = pd.read_csv(TBL / "FIF_index.csv")
    return dict(zip(F["Session ID"], F["Participant"]))


def phase_of(b):
    return np.where(b < 0, "pre", np.where(b < STIM_END, "during", "post"))


def t1_interaction(B):
    B = B.copy(); B["phase"] = phase_of(B.bin.values)
    pm = pid_map(); B["participant"] = B.session.map(pm)
    W = (B.groupby(["file", "participant", "arm", "phase"])[["prev", "extent", "frac_wide"]]
         .mean().reset_index())
    W["phase"] = pd.Categorical(W.phase, ["pre", "during", "post"])
    W["arm"] = pd.Categorical(W.arm, ["Sham", "Treatment"])
    rows = []
    for y, lab in (("prev", "Coupling prevalence"),
                   ("extent", "Derivations per slow oscillation"),
                   ("frac_wide", "Fraction widespread")):
        m = smf.mixedlm("%s ~ C(arm)*C(phase)" % y, W, groups=W.participant).fit()
        for k, c in (("during vs pre", "C(arm)[T.Treatment]:C(phase)[T.during]"),
                     ("post vs pre", "C(arm)[T.Treatment]:C(phase)[T.post]")):
            rows.append(dict(measure=lab, contrast=k, beta=m.params[c],
                             se=m.bse[c], P=m.pvalues[c],
                             n_participants=W.participant.nunique(),
                             n_recordings=W.file.nunique()))
    return pd.DataFrame(rows)


def t2_time_to_effect(B):
    s = B[B.arm == "Treatment"].copy()
    pre = s[s.bin < 0].groupby("file")[["prev", "extent"]].mean()
    pre.columns = ["prev_pre", "extent_pre"]
    s = s.merge(pre, on="file")
    rows = []
    for b, g in s[s.bin >= 0].groupby("bin"):
        for y in ("prev", "extent"):
            d = (g[y] - g[y + "_pre"]).values
            if len(d) < 10:
                continue
            rows.append(dict(bin_centre_min=b, measure=y, n=len(d),
                             mean_change=d.mean() * (100 if y == "prev" else 1),
                             P_wilcoxon=stats.wilcoxon(d).pvalue))
    return pd.DataFrame(rows)


def t3_crossover(B):
    B = B.copy(); B["phase"] = phase_of(B.bin.values)
    pm = pid_map(); B["participant"] = B.session.map(pm)
    W = (B.groupby(["participant", "arm", "phase"])[["prev", "extent"]]
         .mean().unstack("phase"))
    rows = []
    for y in ("prev", "extent"):
        d = (W[(y, "during")] - W[(y, "pre")]).unstack("arm").dropna()
        dd = d["Treatment"] - d["Sham"]
        sc = 100 if y == "prev" else 1
        rows.append(dict(measure=y, n_crossover=len(dd),
                         stim_night_change=d["Treatment"].mean() * sc,
                         sham_night_change=d["Sham"].mean() * sc,
                         difference=dd.mean() * sc,
                         P_wilcoxon=stats.wilcoxon(dd).pvalue if len(dd) > 5 else np.nan,
                         n_positive=int((dd > 0).sum())))
    return pd.DataFrame(rows)


def t4_by_extent(E):
    G = MP.globality(E)
    G["t_min"] = G.trough / SF / 60.0
    on = G[G.block == "stimulation 1"].groupby("file").t_min.min().rename("t_on")
    G = G.merge(on, on="file"); G["t_rel"] = G.t_min - G.t_on
    G = G[(G.t_rel >= -15) & (G.t_rel < 35)]
    G["phase"] = phase_of(G.t_rel.values)
    G["kind"] = np.where(G.involve >= 3, "widespread (≥3 derivations)",
                         "local (1–2 derivations)")
    W = (G.groupby(["file", "arm", "phase", "kind"]).coupled
         .agg(["mean", "size"]).reset_index())
    W = W[W["size"] >= 10]
    P = W.pivot_table(index=["file", "arm", "kind"], columns="phase",
                      values="mean").dropna(subset=["pre", "during"]).reset_index()
    rows = []
    for (arm, kind), g in P.groupby(["arm", "kind"]):
        d = (g.during - g.pre).values * 100
        rows.append(dict(arm=arm, kind=kind, n_recordings=len(d),
                         change_during_minus_pre_pp=d.mean(),
                         P_wilcoxon=stats.wilcoxon(d).pvalue if len(d) > 5 else np.nan))
    T = pd.DataFrame(rows)
    # Between-condition comparison
    for kind, g in P.groupby("kind"):
        a = (g[g.arm == "Treatment"].during - g[g.arm == "Treatment"].pre) * 100
        s = (g[g.arm == "Sham"].during - g[g.arm == "Sham"].pre) * 100
        T.loc[T.kind == kind, "P_between_conditions"] = stats.mannwhitneyu(a, s).pvalue
    return T


def run():
    B = pd.read_csv(TBL / "M7_time_course_bins.csv")
    E, _ = MP.load()
    E = E[E.ROI.isin(["Frontal", "Temporal", "Central", "Parietal"])]
    T1 = t1_interaction(B); T2 = t2_time_to_effect(B)
    T3 = t3_crossover(B); T4 = t4_by_extent(E)
    for n, T in (("M8_T1_condition_x_phase", T1), ("M8_T2_time_to_effect", T2),
                 ("M8_T3_crossover_night", T3), ("M8_T4_change_by_extent", T4)):
        T.to_csv(TBL / (n + ".csv"), index=False, encoding="utf-8-sig")
        print("\n---", n); print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
