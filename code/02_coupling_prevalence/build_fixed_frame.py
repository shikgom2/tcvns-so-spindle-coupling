# -*- coding: utf-8 -*-
"""Finalise the fixed analysis frame + refit R3 (within-night contrasts) on the fixed 57 participants.

Outputs
  tables/FIXED_frame.csv          Table 1 — the five numbers
  tables/FIXED_frame_members.csv  complete participant list per set
  tables/R3_withinnight_fixed57.csv  window contrasts rerun on the fixed 57 participants
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
_R = Path(r"D:/SO-Spindle논문")
P = _R if (_R / "outputs/paper").exists() else _R / "이전"
OUT = P / "outputs/paper"
T = _R / "Final analysis/tables"
T.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261003)

ROIS = ["Frontal", "Temporal", "Central", "Parietal"]
DETS = ["jy_N2", "jy_N2N3", "staresina_N2", "staresina_N2N3"]
g = pd.read_csv(OUT / "S_Table_session_metrics.csv")
idm = pd.read_csv(OUT / "INTERNAL_session_idmap.csv")
idm["ID"] = idm.ID.astype(str)
P2ID = idm.drop_duplicates("participant").set_index("participant").ID
P2CO = idm.drop_duplicates("participant").set_index("participant").cohort_label


def ids(s):
    return sorted(str(P2ID.get(p, p)) for p in s)


def split(s):
    a = sum(1 for p in s if P2CO.get(p) == "Cohort A")
    return a, len(s) - a


# ── set definitions (already verified identical across all cells)
base = g[(g.detector == "staresina_N2") & (g.ROI == "Parietal")].dropna(
    subset=["prevalence"])
stimw = base[base.win == "stim"]
POP = set(base[base.win.isin(["pre", "stim", "post"])].participant)
TRT = set(stimw[stimw.arm == "Treatment"].participant)
NOS = set(stimw[stimw.arm == "Sham"].participant)
BOTH = TRT & NOS

# Within-night: intersection of participants with all three windows in all 16 cells.
# Requiring all three windows fixes a single n for the three contrasts
# (During-Before, After-Before, After-During). Requiring only two splits it into 57/54.
sets_ = []
for det in DETS:
    for roi in ROIS:
        t = g[(g.detector == det) & (g.ROI == roi)
              & (g.arm == "Treatment")].dropna(subset=["prevalence"])
        pv = t.pivot_table(index=["participant", "session_id"], columns="win",
                           values="prevalence")
        need = [w for w in ("pre", "stim", "post") if w in pv.columns]
        sets_.append(set(pv.dropna(subset=need).reset_index().participant)
                     if len(need) == 3 else set())
WN = set.intersection(*sets_)

SETS = [
    ("Analysis population", POP,
     "every participant with analysable EEG in any window"),
    ("Stimulation nights", TRT,
     "contributed at least one night with stimulation"),
    ("No-stimulation nights", NOS,
     "contributed a night without stimulation"),
    ("Crossover subset", BOTH,
     "contributed both conditions — the paired comparison"),
    ("Within-night subset", WN,
     "stimulation nights with all three windows present in all 16 cells"),
]
rows, mem = [], []
for name, s, desc in SETS:
    a, b = split(s)
    rows.append(dict(set_name=name, n=len(s), cohort_A=a, cohort_B=b,
                     definition=desc))
    mem.append(dict(set_name=name, n=len(s), participant_ids=", ".join(ids(s))))
F = pd.DataFrame(rows)
F.to_csv(T / "FIXED_frame.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(mem).to_csv(T / "FIXED_frame_members.csv", index=False,
                         encoding="utf-8-sig")

L = ["=== Table 1 · fixed analysis frame ===", F.to_string(index=False), "",
     "closure check : %d + %d - %d = %d  (population %d)  %s"
     % (len(TRT), len(NOS), len(BOTH), len(TRT) + len(NOS) - len(BOTH),
        len(POP), len(TRT) + len(NOS) - len(BOTH) == len(POP)),
     "within-night subset is %d of the %d stimulation-night participants; "
     "dropped: %s" % (len(WN), len(TRT),
                      ", ".join(ids(TRT - WN)) or "none")]


# ── R3 refit (fixed 57 participants)
def perm1(d, n=20000):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 4:
        return np.nan
    o = abs(d.mean())
    fl = RNG.choice([-1.0, 1.0], (n, d.size))
    return float((np.abs((fl * d).mean(1)) >= o).mean())


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


CONTR = [("stim", "pre", "During - Before"),
         ("post", "pre", "After - Before"),
         ("post", "stim", "After - During")]
out = []
for det in DETS:
    for roi in ROIS:
        t = g[(g.detector == det) & (g.ROI == roi)
              & (g.arm == "Treatment") & g.participant.isin(WN)].dropna(
            subset=["prevalence"])
        pv = t.pivot_table(index="participant", columns="win",
                           values="prevalence")
        for a, b, lab in CONTR:
            if not {a, b} <= set(pv.columns):
                continue
            d = (pv[a] - pv[b]).dropna()
            if len(d) < 6:
                continue
            v = d.values
            lo, hi = boot(v)
            sd = v.std(ddof=1)
            out.append(dict(
                detector=det, ROI=roi, contrast=lab, n=len(v),
                mean_before=float(pv[b].reindex(d.index).mean()),
                mean_after=float(pv[a].reindex(d.index).mean()),
                diff_pp=float(v.mean() * 100), ci_lo_pp=lo * 100,
                ci_hi_pp=hi * 100,
                dz=float(v.mean() / sd) if sd else np.nan,
                n_pos=int((v > 0).sum()),
                p_wilcoxon=float(stats.wilcoxon(v).pvalue),
                p_perm=perm1(v)))
W = pd.DataFrame(out)
W["q"] = W.groupby(["contrast", "detector"]).p_perm.transform(
    lambda p: fdr(p.values))
W.to_csv(T / "R3_withinnight_fixed.csv", index=False, encoding="utf-8-sig")

# before/after fixing — record in full what fixing the frame changes
O = pd.read_csv(OUT / "S_Table_withinnight.csv")
O["contrast"] = O.contrast.map({"Stim − Pre": "During - Before",
                                "Post − Pre": "After - Before",
                                "Post − Stim": "After - During"})
cmp_ = O.merge(W, on=["detector", "ROI", "contrast"],
               suffixes=("_varying", "_fixed"))
cmp_ = cmp_[["detector", "ROI", "contrast", "n_varying", "n_fixed",
             "dz_varying", "dz_fixed", "p_perm_varying", "p_perm_fixed",
             "q_varying", "q_fixed"]]
cmp_["changed_verdict"] = ((cmp_.q_varying < .05) != (cmp_.q_fixed < .05))
cmp_.to_csv(T / "R3_fixed_vs_varying.csv", index=False,
            encoding="utf-8-sig")

L.append("")
L.append("=== R3 재적합 (고정 n=%d) ===" % len(WN))
for c in ("During - Before", "After - Before", "After - During"):
    t = W[W.contrast == c]
    L.append("  %-16s  positive %2d/16   P<.05 %2d/16   q<.05 %2d/16"
             % (c, int((t.diff_pp > 0).sum()), int((t.p_perm < .05).sum()),
                int((t.q < .05).sum())))
L.append("")
L.append(W[W.detector == "staresina_N2"][
    ["contrast", "ROI", "n", "diff_pp", "ci_lo_pp", "ci_hi_pp", "dz",
     "n_pos", "p_perm", "q"]].round(4).to_string(index=False))
L.append("")
L.append("n 이 모든 칸에서 동일한가: %s  (관측된 n: %s)"
         % (W.n.nunique() == 1, sorted(W.n.unique())))
L.append("")
L.append("=== 고정이 바꾼 것 ===")
L.append("  q<.05 칸수  varying %d  ->  fixed %d"
         % (int((cmp_.q_varying < .05).sum()),
            int((cmp_.q_fixed < .05).sum())))
ch = cmp_[cmp_.changed_verdict]
L.append("  판정이 바뀐 칸 %d개:" % len(ch))
L.append(ch.round(4).to_string(index=False))
open(r"D:/_tmp_claude/frame2.txt", "w", encoding="utf-8").write("\n".join(L))
