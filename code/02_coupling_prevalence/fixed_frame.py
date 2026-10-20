# -*- coding: utf-8 -*-
"""Fixed analysis-frame check — does the single population used throughout the manuscript hold for every result."""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
_R = Path(r"D:/SO-Spindle논문")
P = _R if (_R / "outputs/paper").exists() else _R / "이전"
OUT = P / "outputs/paper"
F = _R / "Final analysis"
(F / "tables").mkdir(parents=True, exist_ok=True)

ROIS = ["Frontal", "Temporal", "Central", "Parietal"]
DETS = ["jy_N2", "jy_N2N3", "staresina_N2", "staresina_N2N3"]
g = pd.read_csv(OUT / "S_Table_session_metrics.csv")
idm = pd.read_csv(OUT / "INTERNAL_session_idmap.csv")
idm["ID"] = idm.ID.astype(str)
P2ID = idm.drop_duplicates("participant").set_index("participant").ID
P2CO = idm.drop_duplicates("participant").set_index("participant").cohort_label

L, rows = [], []


def who(d):
    return set(d.participant.unique())


# ── whether the sizes of the three sets are constant across cells
for det in DETS:
    for roi in ROIS:
        t = g[(g.detector == det) & (g.ROI == roi)].dropna(
            subset=["prevalence"])
        s = t[t.win == "stim"]
        trt, shm = who(s[s.arm == "Treatment"]), who(s[s.arm == "Sham"])
        allp = who(t[t.win.isin(["pre", "stim", "post"])])
        # within-night contrast (stimulation nights, participants with both pre & stim)
        tn = t[t.arm == "Treatment"].pivot_table(
            index=["participant", "session_id"], columns="win",
            values="prevalence")
        wn = set(tn.dropna(subset=["pre", "stim"]).reset_index().participant) \
            if {"pre", "stim"} <= set(tn.columns) else set()
        rows.append(dict(detector=det, ROI=roi, n_all=len(allp),
                         n_trt=len(trt), n_nostim=len(shm),
                         n_both=len(trt & shm),
                         sum_check=len(trt) + len(shm) - len(trt & shm),
                         n_withinnight=len(wn)))
T = pd.DataFrame(rows)
T.to_csv(F / "tables/FIXED_frame_by_cell.csv", index=False,
         encoding="utf-8-sig")
L.append("=== 칸별 집합 크기 (16칸) ===")
L.append(T.to_string(index=False))
L.append("")
for c in ("n_all", "n_trt", "n_nostim", "n_both", "n_withinnight"):
    L.append("  %-14s unique values across 16 cells: %s"
             % (c, sorted(T[c].unique())))

# ── fix the membership lists from the reference cell (staresina_N2 / Parietal)
t = g[(g.detector == "staresina_N2") & (g.ROI == "Parietal")].dropna(
    subset=["prevalence"])
s = t[t.win == "stim"]
TRT, SHM = who(s[s.arm == "Treatment"]), who(s[s.arm == "Sham"])
ALL = who(t[t.win.isin(["pre", "stim", "post"])])
BOTH = TRT & SHM
L.append("")
L.append("=== 고정 분석틀 (staresina_N2 / Parietal 기준) ===")
L.append("  analysis population (any window) : %d" % len(ALL))
L.append("  stimulation nights               : %d" % len(TRT))
L.append("  no-stimulation nights            : %d" % len(SHM))
L.append("  both conditions                  : %d" % len(BOTH))
L.append("  %d + %d - %d = %d   (== %d ?  %s)"
         % (len(TRT), len(SHM), len(BOTH), len(TRT) + len(SHM) - len(BOTH),
            len(ALL), len(TRT) + len(SHM) - len(BOTH) == len(ALL)))
L.append("  stimulation only                 : %d" % len(TRT - SHM))
L.append("  no-stimulation only              : %d" % len(SHM - TRT))
L.append("  in population but no stim-window data : %d  (%s)"
         % (len(ALL - TRT - SHM),
            ", ".join(sorted(P2ID.get(p, p) for p in ALL - TRT - SHM))))

for nm, st in (("stimulation nights", TRT), ("no-stimulation nights", SHM),
               ("both conditions", BOTH), ("stimulation only", TRT - SHM),
               ("no-stimulation only", SHM - TRT)):
    ids = sorted(str(P2ID.get(p, p)) for p in st)
    a = sum(1 for p in st if P2CO.get(p) == "Cohort A")
    L.append("")
    L.append("[%s]  n=%d  (A %d / B %d)" % (nm, len(st), a, len(st) - a))
    L.append("   " + ", ".join(ids))

# ── whether this frame applies unchanged to each result
L.append("")
L.append("=== 결과별 적합 여부 ===")
chk = [("R1 interaction", "all 75", sorted(T.n_all.unique())),
       ("R2 between-condition", "70 vs 47", sorted(
           set(zip(T.n_trt, T.n_nostim)))),
       ("R2 within-participant", "42", sorted(T.n_both.unique())),
       ("R3 within-night", "treatment nights", sorted(
           T.n_withinnight.unique()))]
for a, b, c in chk:
    L.append("  %-24s expected %-18s observed %s" % (a, b, c))

open(r"D:/_tmp_claude/frame.txt", "w", encoding="utf-8").write("\n".join(L))
