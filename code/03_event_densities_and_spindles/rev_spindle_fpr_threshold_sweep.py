# -*- coding: utf-8 -*-
"""Spindle-detector false-positive share as a function of the RMS threshold multiple (Sensitivity analyses: spindle threshold).

For 40 random recordings, the stimulation window of every channel is compared with a phase-randomised surrogate that keeps the
amplitude spectrum but flattens the 11-17 Hz sigma peak; the same detector is run on both at k = 1, 1.5, 2, 2.5, 3 SD.
Output  REV_spindle_fpr_threshold_sweep_cells.csv (per recording x channel x k), REV_spindle_fpr_threshold_sweep.csv (summary)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:/SO-Spindle논문/code/paper")
import rev_redetect as RD                                         # noqa: E402
import staresina_detector as SD                                   # noqa: E402
from tvns.jy_source import bad_annot_mask                         # noqa: E402

T = Path(r"D:/SO-Spindle논문/Final analysis/tables"); RNG = np.random.default_rng(20261018)
KS = (1.0, 1.5, 2.0, 2.5, 3.0)


def rms_of(x, sf):
    xf = SD._bp(x, sf, SD.SP_BAND); w = max(1, int(round(SD.RMS_WIN * sf))); k = np.ones(w) / w
    r = np.sqrt(np.convolve(xf ** 2, k, mode="same")); return np.convolve(r, k, mode="same")


def surrogate(x, sf, flatten=True):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(x.size, 1 / sf); A = np.abs(X).copy()
    if flatten:
        lo = (f >= 9.5) & (f <= 10.5); hi = (f >= 17.5) & (f <= 18.5); m = (f > 11) & (f < 17)
        la, ha = np.log(A[lo].mean() + 1e-12), np.log(A[hi].mean() + 1e-12); A[m] = np.exp(la + (ha - la) * (f[m] - 10) / 8)
    ph = RNG.uniform(0, 2 * np.pi, X.size); ph[0] = 0
    if x.size % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(A * np.exp(1j * ph), n=x.size)


def main():
    inv = RD.INV[RD.INV.Cohort.isin(["Cohort A", "Cohort B"])].reset_index(drop=True); pick = RNG.choice(len(inv), 40, replace=False); rows = []
    for i in pick:
        row = inv.iloc[i]; raw = RD.mne.io.read_raw_fif(row.fif_path, preload=True)
        if "sleep" not in raw.ch_names:
            continue
        sf = raw.info["sfreq"]; br = RD.block_concat(raw, "stimulation 1")
        if br is None:
            continue
        bad = bad_annot_mask(br); gm = (~bad).sum() / sf / 60
        for ch in [c for c in RD.CH if c in br.ch_names]:
            x = br.get_data(picks=ch)[0] * 1e6; xs = surrogate(x, sf)
            r = rms_of(x, sf)[~bad]; m, s = r.mean(), r.std(); rs = rms_of(xs, sf)[~bad]; ms, ss = rs.mean(), rs.std()
            for k in KS:
                th = dict(sp_th=m + k * s, sp_hi=m + 9 * s); ths = dict(sp_th=ms + k * ss, sp_hi=ms + 9 * ss)
                rows.append(dict(session_id=row["Session ID"], ch=ch, k=k, good_min=gm,
                                 n_real=len(SD.detect_spindles(x, sf, bad, th=th)), n_sur=len(SD.detect_spindles(xs, sf, bad, th=ths))))
    d = pd.DataFrame(rows); d.to_csv(T / "REV_spindle_fpr_threshold_sweep_cells.csv", index=False)
    S = d.groupby("k").apply(lambda g: pd.Series(dict(real_per_min=g.n_real.sum() / g.good_min.sum(), surrogate_per_min=g.n_sur.sum() / g.good_min.sum(),
                                                      fp_pct=100 * g.n_sur.sum() / g.n_real.sum()))).reset_index()
    S.to_csv(T / "REV_spindle_fpr_threshold_sweep.csv", index=False); print(S.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
