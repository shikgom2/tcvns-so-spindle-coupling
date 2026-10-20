# -*- coding: utf-8 -*-
"""Alternative hypothesis 4 (detection-threshold effect) — tested directly on the event-level table.

If the ratio rose because amplitude dropped and 'small SOs fell below the threshold',
the interaction should disappear when only SOs above an absolute amplitude threshold common
to all recordings (pooled 50th/75th percentile) are counted. If it persists, it is not a threshold effect.

Data: NN/{TB,IIT}/EV_SO.parquet (Staresina re-detection, N2+N3; NN is defined below)
Output: Final analysis/tables/ALT_I_absolute_threshold.csv
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
R = Path(r"D:/SO-Spindle논문")
NN = R / "이전/outputs/nn2023"
INV = pd.read_csv(R / "Final analysis/tables/FIF_index.csv")
OUT = R / "Final analysis/tables/ALT_I_absolute_threshold.csv"
ROIS = ["Frontal", "Temporal", "Central", "Parietal"]
BLK = {"baseline": "pre", "stimulation 1": "stim", "recovery 1": "post"}


def token(path):
    b = Path(path).name.replace("_raw_fin.fif", "").replace("_raw.fif", "")
    return re.sub(r"^[A-Z]\d{3}_", "", b)

def load():
    ev = pd.concat([pd.read_parquet(NN / d / "EV_SO.parquet") for d in ("TB", "IIT")])
    ev = ev[ev.ROI.isin(ROIS) & ev.block.isin(BLK)]
    inv = INV.assign(tok=INV.fif_path.map(token))[["tok", "Session ID", "Participant", "Condition"]]
    inv.columns = ["file", "session_id", "participant", "arm"]
    ev = ev.merge(inv, on="file", how="inner")
    ev["win"] = ev.block.map(BLK)
    return ev


def prevalence(ev, thr):
    s = ev[ev.amp_ptp >= thr]
    g = s.groupby(["participant", "session_id", "arm", "ROI", "win"])
    p = g.coupled_window.agg(["mean", "size"]).reset_index()
    p.columns = ["participant", "session_id", "arm", "ROI", "win", "prev", "n"]
    return p[p.n >= 5]


def interaction(p, roi):
    t = p[p.win.isin(["pre", "stim"])].copy()
    if roi != "All":
        t = t[t.ROI == roi]
    t["arm01"] = (t.arm == "Treatment").astype(int)
    t["w01"] = (t.win == "stim").astype(int)
    m = smf.mixedlm("prev ~ arm01 * w01", t, groups=t.participant).fit(reml=False)
    return m.params["arm01:w01"], m.pvalues["arm01:w01"], t.participant.nunique(), len(t)


def main():
    ev = load()
    print("events", len(ev), "files", ev.file.nunique(), "sessions", ev.session_id.nunique(),
          "arms", ev.arm.value_counts().to_dict())
    rows = []
    for lab, q in (("all", 0), ("≥ pooled median", 50), ("≥ pooled 75th", 75)):
        thr = np.percentile(ev.amp_ptp, q) if q else 0
        p = prevalence(ev, thr)
        for roi in ROIS + ["All"]:
            b, pv, npart, nrow = interaction(p, roi)
            rows.append(dict(threshold=lab, thr_uV=round(thr, 1), ROI=roi, beta=b, p=pv,
                             n_participants=npart, n_rows=nrow))
    D = pd.DataFrame(rows)
    print(D.pivot_table(index="ROI", columns="threshold", values=["beta", "p"]).round(4).to_string())
    D.to_csv(OUT, index=False)
    # For reference: amplitude distribution of SOs lost in the stimulation window — during−before amplitude change per condition
    a = ev.groupby(["session_id", "arm", "win"]).amp_ptp.median().unstack("win")
    a["d"] = a.stim - a.pre
    print("\nmedian SO amplitude during−before (µV):",
          a.groupby("arm").d.describe()[["count", "mean", "50%"]].round(2).to_string())


if __name__ == "__main__":
    main()
