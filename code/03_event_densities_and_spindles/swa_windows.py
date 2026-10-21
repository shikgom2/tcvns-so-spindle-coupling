# -*- coding: utf-8 -*-
"""Absolute slow-wave activity (SWA) — by window x region, same framework as the coupling analysis.

Unlike coupling, SWA is a continuous measure. A Welch PSD is computed over artefact-free N2 segments and
  SWA     0.5-4 Hz   (conventional definition)
  SO band 0.5-1 Hz   (band used by the SO detector)
  delta   1-4 Hz
are integrated in uV^2/Hz. The distribution is log-normal, so values are reported on a log scale.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

warnings.filterwarnings("ignore")
import mne                                                        # noqa: E402
mne.set_log_level("ERROR")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402

R = LP.PROJECT_ROOT
INV = pd.read_csv(R / "Final analysis/tables/FIF_index.csv")
OUT = R / "Final analysis/tables/SWA_windows.csv"
WINS = {"baseline": "pre", "stimulation 1": "stim", "recovery 1": "post"}
REG = {"Frontal": ["Fp1", "Fp2", "F3", "F4"],
       "Temporal": ["F7", "F8", "T3", "T4"],
       "Central": ["C3", "Cz", "C4"],
       "Parietal": ["P3", "P4"]}
BANDS = {"swa": (0.5, 4.0), "so_band": (0.5, 1.0), "delta": (1.0, 4.0),
         "sigma": (12.0, 16.0)}


def one(row):
    raw = mne.io.read_raw_fif(row.fif_path, preload=True)
    sf = raw.info["sfreq"]
    ann = raw.annotations
    win = {d: (o, o + du) for d, o, du in
           zip(ann.description, ann.onset, ann.duration) if d in WINS}
    bad = [(o, o + du) for d, o, du in
           zip(ann.description, ann.onset, ann.duration) if d.startswith("BAD")]
    names = raw.ch_names
    if "sleep" not in names:
        return []
    sleep = raw.get_data(picks=[names.index("sleep")])[0]
    t = raw.times
    out = []
    for name, (a, b) in win.items():
        m = (t >= a) & (t < b)
        if m.sum() < sf * 60:
            continue
        s = sleep[m]
        badmask = np.zeros(s.size, bool)
        tt = t[m]
        for o, e in bad:
            badmask |= (tt >= o) & (tt < e)
        good = (s == 3) & ~badmask            # artefact-free N2
        if good.sum() < sf * 60:
            continue
        rec = dict(session_id=row["Session ID"], participant=row["Participant"],
                   arm=row.Condition, win=WINS[name],
                   n2_min=float(good.sum() / sf / 60))
        for reg, chans in REG.items():
            ch = [c for c in chans if c in names]
            if not ch:
                continue
            x = raw.get_data(picks=[names.index(c) for c in ch])[:, m][:, good] * 1e6
            f, p = signal.welch(x, sf, nperseg=int(4 * sf), axis=1)
            df_ = f[1] - f[0]
            for bn, (lo, hi) in BANDS.items():
                sel = (f >= lo) & (f < hi)
                # Average uV^2/Hz across the band — corrects for differences in bandwidth
                rec["%s_%s" % (reg[:4].lower(), bn)] = float(np.mean(p[:, sel].mean(1)))
        out.append(rec)
    return out


def main():
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else len(INV)
    rows = []
    for i, row in INV.head(lim).iterrows():
        try:
            rows += one(row)
        except Exception as exc:                                   # noqa: BLE001
            print("FAIL", row["Session ID"], exc, file=sys.stderr, flush=True)
        if i % 40 == 0:
            print(i, "/", lim, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    print("saved", OUT, d.shape)


if __name__ == "__main__":
    main()
