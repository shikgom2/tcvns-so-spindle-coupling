# -*- coding: utf-8 -*-
"""Epoch-wise agreement between the two staging classifiers on the stream-B recordings (Methods: sleep staging).

Compares the stage channel of the analysed .fif (second classifier) with the YASA-staged copy of the same recording.
Output  REV_staging_agreement_stages.csv      one-vs-rest kappa, minutes, sensitivity and PPV per stage
        REV_staging_agreement_recordings.csv  agreement and kappa per recording
        REV_staging_agreement_confusion.csv   5 x 5 confusion matrix
"""
from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

warnings.filterwarnings("ignore")
T = Path(r"D:/SO-Spindle논문/Final analysis/tables")
YASA_DIR = r"D:/HJ/vns/rawfiles_fromJY/DATA/Data/iit_delivery/Data/IITY"        # YASA-staged copies of the stream-B recordings
CODE = {0: "W", 1: "R", 2: "N1", 3: "N2", 4: "N3"}; L = ["W", "N1", "N2", "N3", "R"]


def epochs(f):
    raw = mne.io.read_raw_fif(f, preload=False, verbose=False); sl = raw.get_data(picks="sleep")[0]; sf = raw.info["sfreq"]
    n = int(np.floor(sl.size / (sf * 30)))
    return np.array([CODE.get(int(round(np.median(sl[int(i * sf * 30):int((i + 1) * sf * 30)]))), "?") for i in range(n)])


def main():
    d = pd.read_csv(T / "FIF_index.csv"); d = d[d.Cohort == "Cohort B"]
    A, B, per = [], [], []
    for _, r in d.iterrows():
        fy = os.path.join(YASA_DIR, r["Source subject ID"], os.path.basename(r.fif_path))
        if not os.path.exists(fy):
            cand = glob.glob(os.path.join(YASA_DIR, r["Source subject ID"], "*" + os.path.basename(r.fif_path).split("_raw")[0] + "*raw_fin.fif"))
            fy = cand[0] if cand else None
        if fy is None:
            print("missing YASA file for", r["Session ID"]); continue
        a, b = epochs(r.fif_path), epochs(fy); n = min(len(a), len(b)); a, b = a[:n], b[:n]
        A.append(a); B.append(b)
        per.append(dict(session_id=r["Session ID"], n_epochs=n, agreement=(a == b).mean(),
                        kappa=cohen_kappa_score(a, b) if len(set(a) | set(b)) > 1 else np.nan,
                        yasa_N2_min=0.5 * (b == "N2").sum(), neuronet_N2_min=0.5 * (a == "N2").sum(),
                        yasa_W_min=0.5 * (b == "W").sum(), neuronet_W_min=0.5 * (a == "W").sum()))
    a = np.concatenate(A); b = np.concatenate(B); P = pd.DataFrame(per)
    print("recordings", len(P), "epochs", len(a), "agreement %.3f kappa %.3f" % ((a == b).mean(), cohen_kappa_score(a, b)))
    cm = confusion_matrix(a, b, labels=L)
    rows = []
    for s in L:
        ta, tb = (a == s), (b == s)
        if ta.sum() + tb.sum() == 0:
            continue
        tp = (ta & tb).sum()
        rows.append(dict(stage=s, kappa_one_vs_rest=cohen_kappa_score(ta, tb), neuronet_min=0.5 * ta.sum(), yasa_min=0.5 * tb.sum(),
                         sensitivity_vs_yasa=tp / max(tb.sum(), 1), ppv_vs_yasa=tp / max(ta.sum(), 1)))
    pd.DataFrame(rows).to_csv(T / "REV_staging_agreement_stages.csv", index=False)
    P.to_csv(T / "REV_staging_agreement_recordings.csv", index=False)
    pd.DataFrame(cm, index=["NeuroNet_" + l for l in L], columns=["YASA_" + l for l in L]).to_csv(T / "REV_staging_agreement_confusion.csv")


if __name__ == "__main__":
    main()
