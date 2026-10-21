# -*- coding: utf-8 -*-
"""Mechanism analysis — examine in the EEG itself why the coupled ratio increased.

Three questions are asked at the event level.

 1  Amplitude gate   Which slow oscillations recruit spindles? Does stimulation lower that threshold?
 2  Propagation/globality  On how many electrodes does an SO appear simultaneously (global vs local)?
                Do global SOs recruit more spindles? Does stimulation increase globality?
 3  Spindle properties   Do coupled spindles differ from uncoupled spindles?

Input is the re-detected event tables (read-only). Outputs go only to Final analysis/tables/.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402
R = LP.PROJECT_ROOT
EV = R / "이전/outputs/nn2023"
TBL = R / "Final analysis/tables"
SF = 125.0                      # common to all sessions
WIN = {"baseline": "pre", "stimulation 1": "stim", "recovery 1": "post"}
ROIS = ["Frontal", "Temporal", "Central", "Parietal"]
TOL = 0.40                      # globality decision window (s)


def load():
    E = pd.concat([pd.read_parquet(EV / k / "EV_SO.parquet")
                   for k in ("TB", "IIT")], ignore_index=True)
    S = pd.concat([pd.read_parquet(EV / k / "EV_SP.parquet")
                   for k in ("TB", "IIT")], ignore_index=True)
    F = pd.read_csv(TBL / "FIF_index.csv")
    stem = F["Processed file (.fif)"].str.extract(
        r"(\d{4}-\d{2}-\d{2}-\d{4})")[0]
    M = (pd.DataFrame(dict(file=stem, arm=F.Condition,
                           participant=F.Participant,
                           session=F["Session ID"]))
         .dropna().drop_duplicates("file"))
    for T in (E, S):
        T["win"] = T.block.map(WIN)
    E = E.merge(M, on="file", how="inner")
    S = S.merge(M, on="file", how="inner")
    E["coupled"] = (E.coupled_window > 0).astype(int)
    E["slope"] = E.amp_ptp / ((E.peak - E.trough).clip(lower=1) / SF)
    return E.dropna(subset=["win"]), S.dropna(subset=["win"])


# ----------------------------------------------------------------- 1
def gating(E):
    """Amplitude difference between coupled and uncoupled SOs, and coupling probability by amplitude."""
    rows, curve = [], []
    for (sess, arm, win, roi), g in E.groupby(["session", "arm", "win",
                                               "ROI"]):
        if g.coupled.sum() < 5 or (1 - g.coupled).sum() < 5:
            continue
        rows.append(dict(session=sess, arm=arm, win=win, ROI=roi,
                         n=len(g),
                         amp_coupled=g.loc[g.coupled == 1, "amp_ptp"].mean(),
                         amp_uncoupled=g.loc[g.coupled == 0,
                                             "amp_ptp"].mean(),
                         slope_coupled=g.loc[g.coupled == 1, "slope"].mean(),
                         slope_uncoupled=g.loc[g.coupled == 0,
                                               "slope"].mean(),
                         amp_all=g.amp_ptp.mean(), prev=g.coupled.mean()))
        # amplitude-coupling curve from within-recording quintiles
        q = pd.qcut(g.amp_ptp, 5, labels=False, duplicates="drop")
        for k, h in g.groupby(q):
            curve.append(dict(session=sess, arm=arm, win=win, ROI=roi,
                              quintile=int(k) + 1, prev=h.coupled.mean(),
                              amp=h.amp_ptp.mean(), n=len(h)))
    A = pd.DataFrame(rows)
    A["amp_gap"] = A.amp_coupled - A.amp_uncoupled
    A["slope_gap"] = A.slope_coupled - A.slope_uncoupled
    return A, pd.DataFrame(curve)


# ----------------------------------------------------------------- 2
def globality(E):
    """On how many electrodes a single SO appears simultaneously + whether that SO is coupled."""
    out = []
    for fl, g in E.groupby("file"):
        chs = {c: np.sort(h.trough.values / SF)
               for c, h in g.groupby("ch")}
        nch = len(chs)
        if nch < 8:
            continue
        for c, h in g.groupby("ch"):
            t = h.trough.values / SF
            inv = np.ones(t.size)
            for c2, t2 in chs.items():
                if c2 == c or t2.size == 0:
                    continue
                i = np.searchsorted(t2, t)
                d = np.full(t.size, np.inf)
                for j in (i - 1, i):
                    k = np.clip(j, 0, t2.size - 1)
                    d = np.minimum(d, np.abs(t2[k] - t))
                inv += (d <= TOL)
            out.append(h.assign(involve=inv, n_ch_file=nch))
    G = pd.concat(out, ignore_index=True)
    G["global_so"] = (G.involve >= 0.5 * G.n_ch_file).astype(int)
    return G


def globality_summary(G):
    rec = (G.groupby(["session", "arm", "win", "ROI"])
           .agg(involve=("involve", "mean"),
                frac_global=("global_so", "mean"),
                prev=("coupled", "mean"), n=("coupled", "size"))
           .reset_index())
    rec = rec[rec.n >= 20]
    # Are global SOs more often coupled? — paired comparison within recordings
    pair = (G.groupby(["session", "arm", "win", "ROI", "global_so"])
            .agg(prev=("coupled", "mean"), n=("coupled", "size"))
            .reset_index())
    pair = pair[pair.n >= 10].pivot_table(
        index=["session", "arm", "win", "ROI"], columns="global_so",
        values="prev").dropna()
    pair.columns = ["local", "global"]
    return rec, pair.reset_index()


# ----------------------------------------------------------------- 3
def spindles(S):
    S = S.assign(coupled=(S.coupled_window > 0).astype(int))
    return (S.groupby(["session", "arm", "win", "ROI", "coupled"])
            .agg(dur=("dur_sec", "mean"), rms=("rms_peak", "mean"),
                 ncyc=("n_cycles", "mean"), n=("dur_sec", "size"))
            .reset_index().query("n >= 10"))


def run():
    E, S = load()
    print("SO events %d · spindles %d · sessions %d · participants %d"
          % (len(E), len(S), E.session.nunique(), E.participant.nunique()))
    A, C = gating(E)
    A.to_csv(TBL / "M1_amp_gating.csv", index=False, encoding="utf-8-sig")
    C.to_csv(TBL / "M1_amp_curve.csv", index=False, encoding="utf-8-sig")
    G = globality(E)
    rec, pair = globality_summary(G)
    rec.to_csv(TBL / "M2_globality_recording.csv", index=False,
               encoding="utf-8-sig")
    pair.to_csv(TBL / "M2_globality_pair.csv", index=False,
                encoding="utf-8-sig")
    G.groupby(["session", "arm", "win", "ROI", "involve"]).coupled.agg(
        ["mean", "size"]).reset_index().to_csv(
        TBL / "M2_involve_curve.csv", index=False, encoding="utf-8-sig")
    spindles(S).to_csv(TBL / "M3_spindle_props.csv", index=False,
                       encoding="utf-8-sig")
    print("[saved] M1_amp_gating %d · M1_amp_curve %d · "
          "M2_globality_recording %d · M2_globality_pair %d · "
          "M3_spindle_props rows" % (len(A), len(C), len(rec), len(pair)))


if __name__ == "__main__":
    run()
