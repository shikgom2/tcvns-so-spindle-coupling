# -*- coding: utf-8 -*-
"""Low-frequency phase-consistency spectrum seen from spindle peaks (PPC).

For every recording and every region:
  · find the times of spindle (12-16 Hz) envelope peaks within N2/N3.
  · sweep 0.5-3.0 Hz and read the instantaneous phase of each frequency at those times.
  · compute the pairwise phase consistency — it is not biased by the number of events.
        PPC = (|Σ e^{iθ}|² − n) / (n(n−1))

Output: tables/R_ppc_spectrum.csv  (recording × region × frequency)
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
from scipy.signal import butter, filtfilt, hilbert                 # noqa: E402

from so_locked_windows import SF, STAGE, SP_LO, SP_HI, spindle_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402

_R = LP.PROJECT_ROOT
TBL = _R / "Final analysis/tables"
REGIONS = {"Frontal": ["Fp1", "Fp2", "F3", "F4"],
           "Parietal": ["P3", "P4"],
           "Central": ["C3", "Cz", "C4"],
           "Temporal": ["F7", "F8", "T3", "T4"]}
FREQS = np.round(np.arange(0.5, 3.01, 0.1), 2)
BW = 0.35                      # half-width (Hz)
MIN_PEAKS = 20                 # PPC has no event-count bias, so a low minimum is acceptable


def bp(x, lo, hi, order=3):
    b, a = butter(order, [lo / (SF / 2), hi / (SF / 2)], btype="band")
    return filtfilt(b, a, x)


def ppc(theta):
    n = theta.size
    if n < 2:
        return np.nan
    r = np.abs(np.exp(1j * theta).sum()) ** 2
    return float((r - n) / (n * (n - 1)))


def session(fif):
    raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    if "sleep" not in raw.ch_names:
        return None
    nrem = np.isin(raw.get_data(picks="sleep")[0],
                   [STAGE["N2"], STAGE["N3"]])
    if nrem.sum() < SF * 120:
        return None
    rows = []
    for reg, want in REGIONS.items():
        chs = [c for c in want if c in raw.ch_names]
        if not chs:
            continue
        # Region-averaged signal (polarity inversion follows the slow-oscillation phase convention)
        x = -raw.get_data(picks=chs).mean(0) * 1e6
        env = np.abs(hilbert(bp(x, SP_LO, SP_HI)))
        pk = spindle_peaks(env)
        pk = pk[nrem[pk]] if len(pk) else pk
        if len(pk) < MIN_PEAKS:
            continue
        for f0 in FREQS:
            lo, hi = max(f0 - BW, .20), f0 + BW
            ph = np.angle(hilbert(bp(x, lo, hi)))
            rows.append(dict(region=reg, freq=float(f0), n_peaks=len(pk),
                             ppc=ppc(ph[pk])))
    return rows or None


def main(limit=None):
    inv = pd.read_csv(TBL / "FIF_index.csv").dropna(subset=["fif_path"])
    if limit:
        inv = inv.head(int(limit))
    out, done, bad = [], 0, 0
    for row in inv.itertuples(index=False):
        try:
            got = session(Path(row.fif_path))
        except Exception:
            bad += 1
            continue
        if got is None:
            bad += 1
            continue
        arm = "Treatment" if str(row.Condition).lower().startswith("t") \
            else "Sham"
        for r in got:
            r.update(session=Path(row.fif_path).name, arm=arm,
                     cohort=row.Cohort, participant=row.Participant)
            out.append(r)
        done += 1
        if done % 30 == 0:
            print("  %d recordings" % done, flush=True)
    D = pd.DataFrame(out)
    D.to_csv(TBL / "R_ppc_spectrum.csv", index=False, encoding="utf-8-sig")
    print("recordings used %d · skipped %d · rows %d" % (done, bad, len(D)))
    pk = (D.loc[D.groupby(["session", "region"]).ppc.idxmax()]
          .groupby(["region", "arm"]).freq.median())
    print("\npeak-PPC frequency, median (Hz):")
    print(pk.round(2).to_string())
    print("[saved] %s" % (TBL / "R_ppc_spectrum.csv"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
