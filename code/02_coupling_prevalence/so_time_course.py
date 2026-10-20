# -*- coding: utf-8 -*-
"""Coupling prevalence over elapsed time — Result 4, "does not change with time itself".

Each recording is cut into 30-min bins; slow oscillations are detected only within N2/N3, and the
bin's coupling ratio (= coupled SOs / all SOs) is computed. All 13 channels are pooled
(not split by region — this panel only needs to show the time trend).

Output: tables/R4_time_course.csv   (recording x bin)
        tables/R4_stim_span.csv     (how far into the recording the stimulation window occurred)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import mne                                                        # noqa: E402

mne.set_log_level("ERROR")
from so_locked_windows import (SF, STAGE, NS, SO_LO, SO_HI, SP_LO,  # noqa: E402
                               SP_HI, bp, detect_so, spindle_peaks)
from scipy.signal import hilbert                                   # noqa: E402

_R = Path(r"D:/SO-Spindle논문")
TBL = _R / "Final analysis/tables"
CHS = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4",
       "C3", "Cz", "C4", "P3", "P4"]
# Recordings are about 40 min long, not a whole night (baseline / stimulation 1 /
# recovery 1). The time axis is therefore "minutes since recording start" and bins are 5 min.
BIN_MIN = 5.0                       # bin length (min)
MIN_NREM = 1.5                      # minimum N2/N3 per bin (min)
MIN_SO = 5                          # minimum number of SOs per bin


def session(fif):
    raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    if "sleep" not in raw.ch_names:
        return None, None
    n = raw.n_times
    nrem = np.isin(raw.get_data(picks="sleep")[0], [STAGE["N2"], STAGE["N3"]])

    # When the stimulation window falls relative to recording start (for shading)
    span = None
    ann = raw.annotations
    m = [i for i, d in enumerate(ann.description) if d == "stimulation 1"]
    if m:
        i = m[0]
        span = (ann.onset[i] / 60.0,
                (ann.onset[i] + ann.duration[i]) / 60.0)

    chs = [c for c in CHS if c in raw.ch_names]
    if not chs:
        return None, span
    sig = {}
    for ch in chs:
        x = -raw.get_data(picks=ch)[0] * 1e6          # polarity inversion
        xso = bp(x, SO_LO, SO_HI)
        sig[ch] = (xso, np.angle(hilbert(xso)),
                   np.abs(hilbert(bp(x, SP_LO, SP_HI))))

    step = int(BIN_MIN * 60 * SF)
    rows = []
    for b0 in range(0, n, step):
        b1 = min(b0 + step, n)
        ok0 = np.zeros(n, bool)
        ok0[b0:b1] = True
        ok0 &= nrem
        if ok0.sum() < MIN_NREM * 60 * SF:
            continue
        nco = nun = 0
        for ch in chs:
            xso, ph, env = sig[ch]
            sos = [t for t in detect_so(xso)
                   if ok0[t[0]] and NS < t[0] < n - NS - 1]
            if not sos:
                continue
            pk = spindle_peaks(env)
            pk = pk[ok0[pk]] if len(pk) else pk
            for t0, s0, e0 in sos:
                inside = pk[(pk >= s0) & (pk < e0)] if len(pk) else []
                hit = any(abs(np.angle(np.exp(1j * ph[p]))) <= np.pi / 2
                          for p in inside)
                nco += int(hit)
                nun += int(not hit)
        if nco + nun < MIN_SO:
            continue
        # Use the nominal bin centre so the grid is kept even if the last bin is truncated
        rows.append(dict(t_min=b0 / SF / 60.0 + BIN_MIN / 2,
                         nrem_min=ok0.sum() / SF / 60.0,
                         n_so=nco + nun, n_coupled=nco,
                         prevalence=nco / (nco + nun)))
    return (pd.DataFrame(rows) if rows else None), span


def main(limit=None):
    inv = pd.read_csv(TBL / "FIF_index.csv").dropna(subset=["fif_path"])
    if limit:
        inv = inv.head(int(limit))
    out, spans, done, bad = [], [], 0, 0
    for row in inv.itertuples(index=False):
        try:
            d, span = session(Path(row.fif_path))
        except Exception:
            bad += 1
            continue
        arm = "Treatment" if str(row.Condition).lower().startswith("t") \
            else "Sham"
        if span is not None:
            spans.append(dict(session=Path(row.fif_path).name, arm=arm,
                              stim_start_min=span[0], stim_end_min=span[1]))
        if d is None:
            bad += 1
            continue
        d["session"] = Path(row.fif_path).name
        d["arm"] = arm
        d["cohort"] = row.Cohort
        out.append(d)
        done += 1
        if done % 40 == 0:
            print("  %d recordings" % done, flush=True)
    D = pd.concat(out, ignore_index=True)
    D.to_csv(TBL / "R4_time_course.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(spans).to_csv(TBL / "R4_stim_span.csv", index=False,
                               encoding="utf-8-sig")
    print("recordings used %d · skipped %d · bins %d" % (done, bad, len(D)))
    print(D.groupby("arm").prevalence.describe().round(3).to_string())
    print("[saved] %s" % (TBL / "R4_time_course.csv"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
