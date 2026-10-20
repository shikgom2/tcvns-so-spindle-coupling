# -*- coding: utf-8 -*-
"""Hypnogram metrics per recording from the sleep-stage channel of each .fif (Methods: sleep staging).

Input   Final analysis/tables/FIF_index.csv  (session code, participant, condition, stream, fif path)
Output  REV_hypnogram_metrics.csv   per recording: latency, TST, efficiency, minutes and % per stage, WASO
        REV_hypnogram_summary.csv   stimulation vs sham (participant means), Mann-Whitney, by stream and pooled
Stage codes in the 'sleep' channel: 0 W, 1 REM, 2 N1, 3 N2, 4 N3.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
T = Path(r"D:/SO-Spindle논문/Final analysis/tables")
CODE = {0: "W", 1: "R", 2: "N1", 3: "N2", 4: "N3"}
COLS = ["recording_min", "latency_sleep_min", "latency_N2_min", "TST_min", "sleep_efficiency_pct",
        "W_min", "N1_min", "N2_min", "N3_min", "REM_min", "WASO_min", "N1_pct", "N2_pct", "N3_pct", "REM_pct", "W_pct"]


def epochs(fif):
    raw = mne.io.read_raw_fif(fif, preload=False, verbose=False)
    sl = raw.get_data(picks="sleep")[0]; sf = raw.info["sfreq"]; n = int(np.floor(sl.size / (sf * 30)))
    return np.array([CODE.get(int(round(np.median(sl[int(i * sf * 30):int((i + 1) * sf * 30)]))), "?") for i in range(n)])


def main():
    d = pd.read_csv(T / "FIF_index.csv"); rows = []
    for _, r in d.iterrows():
        ep = epochs(r.fif_path)
        mins = {s: 0.5 * (ep == s).sum() for s in ("W", "N1", "N2", "N3", "R")}
        sleep = np.isin(ep, ["N1", "N2", "N3", "R"]); onset = int(np.argmax(sleep)) if sleep.any() else None
        n2 = ep == "N2"; onset_n2 = int(np.argmax(n2)) if n2.any() else None
        rec_min = len(ep) * 0.5; tst = sum(mins[s] for s in ("N1", "N2", "N3", "R"))
        rows.append(dict(session_id=r["Session ID"], participant=r.Participant, condition=r.Condition,
                         stream=str(r.Cohort).replace("Cohort ", ""), recording_min=rec_min,
                         latency_sleep_min=np.nan if onset is None else 0.5 * onset,
                         latency_N2_min=np.nan if onset_n2 is None else 0.5 * onset_n2,
                         TST_min=tst, sleep_efficiency_pct=100 * tst / rec_min, W_min=mins["W"], N1_min=mins["N1"],
                         N2_min=mins["N2"], N3_min=mins["N3"], REM_min=mins["R"],
                         WASO_min=0.5 * (ep[onset:] == "W").sum() if onset is not None else np.nan))
    df = pd.DataFrame(rows)
    for s in ("N1", "N2", "N3", "REM", "W"):
        df[s + "_pct"] = 100 * df[s + "_min"] / df.recording_min
    df.to_csv(T / "REV_hypnogram_metrics.csv", index=False)
    summ = []
    for st in ("A", "B", "all"):
        sub = df if st == "all" else df[df.stream == st]
        part = sub.groupby(["participant", "condition"])[COLS].mean().reset_index()
        for c in COLS:
            a = part[part.condition == "Treatment"][c].dropna(); b = part[part.condition == "Sham"][c].dropna()
            summ.append(dict(stream=st, metric=c, stim_median=a.median(), stim_q1=a.quantile(.25), stim_q3=a.quantile(.75), n_stim=len(a),
                             nostim_median=b.median(), nostim_q1=b.quantile(.25), nostim_q3=b.quantile(.75), n_nostim=len(b),
                             mannwhitney_p=stats.mannwhitneyu(a, b).pvalue if len(a) > 2 and len(b) > 2 else np.nan))
    pd.DataFrame(summ).to_csv(T / "REV_hypnogram_summary.csv", index=False)
    print("recordings", len(df))


if __name__ == "__main__":
    main()
