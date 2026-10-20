# -*- coding: utf-8 -*-
"""YASA detector benchmark (reviewer: compare with an established pipeline, report absolute densities).

On artefact-free N2 of each recording × window (pre/stim/post) × channel, run
  yasa.sw_detect   (amplitude criteria from the original thesis pipeline: neg 10–200, pos 5–150, ptp 20–350 µV — defaults give 0 events on 13-channel average reference)
  yasa.spindles_detect (freq_sp 12–16 Hz, otherwise defaults)
and store SO count, spindle count, coupled-SO count (two definitions: spindle peak inside the SO / additionally within ±90° of the positive peak) and N2 duration.
Output  tables/REV_yasa_cells.csv
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yasa
from scipy.signal import butter, filtfilt, hilbert

warnings.filterwarnings("ignore")
import logging; logging.getLogger("yasa").setLevel(logging.ERROR)
sys.path.insert(0, r"D:/SO-Spindle논문/code/paper")
import rev_redetect as RD                                         # noqa: E402
from tvns.jy_source import bad_annot_mask                         # noqa: E402

R = Path(r"D:/SO-Spindle논문/Final analysis/tables")
SO_AMP = dict(amp_neg=(10, 200), amp_pos=(5, 150), amp_ptp=(20, 350))


def so_phase(x, sf):
    b, a = butter(3, [0.3 / (sf / 2), 1.25 / (sf / 2)], "band")
    return np.angle(hilbert(filtfilt(b, a, x)))


def one(x, sf, bad):
    """x: µV, concatenated N2 signal. bad: bool mask. YASA cannot exclude BAD via the hypnogram, so BAD segments are marked 'W'(0) to drop out under include=2."""
    hyp = np.where(bad, 0, 2).astype(int)
    out = dict(n_SO=0, n_spindle=0, n_cp_any=0, n_cp_90=0)
    sw = yasa.sw_detect(x, sf, hypno=hyp, include=2, freq_sw=(0.3, 1.5), dur_neg=(0.3, 1.5), dur_pos=(0.1, 1.0), **SO_AMP, remove_outliers=False, verbose=False)
    sp = yasa.spindles_detect(x, sf, hypno=hyp, include=2, freq_sp=(12, 16), remove_outliers=False, verbose=False)
    if sw is None:
        return out
    S = sw.summary(); out["n_SO"] = len(S)
    if sp is None:
        return out
    P = sp.summary(); out["n_spindle"] = len(P)
    pk = np.sort(P.Peak.values); ph = so_phase(x, sf)
    pk_i = np.clip((pk * sf).astype(int), 0, len(x) - 1); pk_ph = ph[pk_i]
    starts, ends = S.Start.values, S.End.values
    lo = np.searchsorted(pk, starts); hi = np.searchsorted(pk, ends, side="right")
    any_ = hi > lo; out["n_cp_any"] = int(any_.sum())
    n90 = 0
    for i in np.flatnonzero(any_):
        if np.any(np.abs(pk_ph[lo[i]:hi[i]]) <= np.pi / 2):
            n90 += 1
    out["n_cp_90"] = n90
    return out


def session(row):
    raw = RD.mne.io.read_raw_fif(row.fif_path, preload=True)
    if "sleep" not in raw.ch_names:
        return []
    sf = raw.info["sfreq"]; rows = []
    for b, br in {b: RD.block_concat(raw, b) for b in RD.BLOCKS}.items():
        if br is None:
            continue
        bad = bad_annot_mask(br); dur = float(br.times[-1] - br.times[0])
        for ch in RD.CH:
            if ch not in br.ch_names:
                continue
            o = one(br.get_data(picks=ch)[0] * 1e6, sf, bad)
            rows.append(dict(session_id=row["Session ID"], participant=row.Participant, arm=row.Condition, cohort=row.Cohort,
                             win=RD.BLOCKS[b], ch=ch, duration_sec=dur, good_sec=float((~bad).sum() / sf), **o))
    return rows


def main():
    rows, t0 = [], time.time()
    for i, (_, row) in enumerate(RD.INV.iterrows(), 1):
        try:
            rows += session(row)
        except Exception as exc:                                   # noqa: BLE001
            print("FAIL", row["Session ID"], exc, flush=True)
        if i % 10 == 0 or i == len(RD.INV):
            el = time.time() - t0
            print("%d/%d  %.1f min  remaining ~%.1f min" % (i, len(RD.INV), el / 60, el / i * (len(RD.INV) - i) / 60), flush=True)
    d = pd.DataFrame(rows); d.to_csv(R / "REV_yasa_cells.csv", index=False); print("saved", d.shape)


if __name__ == "__main__":
    main()
