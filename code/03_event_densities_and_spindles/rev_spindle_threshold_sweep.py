# -*- coding: utf-8 -*-
"""Re-detection as a function of the spindle RMS threshold multiple (k = 1, 1.5, 2, 2.5 SD): SOs unchanged, only spindle and coupled counts per k.
Per-window thresholds, N2, 246 recordings, 13 channels. Output tables/REV_spindle_threshold_sweep_cells.csv"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:/SO-Spindle논문/code/paper")
import rev_redetect as RD                                         # noqa: E402
import staresina_detector as SD                                   # noqa: E402
from tvns.jy_source import bad_annot_mask                         # noqa: E402

T = Path(r"D:/SO-Spindle논문/Final analysis/tables")
KS = (1.0, 1.5, 2.0, 2.5)


def rms_of(x, sf):
    xf = SD._bp(x, sf, SD.SP_BAND); w = max(1, int(round(SD.RMS_WIN * sf))); k = np.ones(w) / w
    r = np.sqrt(np.convolve(xf ** 2, k, mode="same")); return np.convolve(r, k, mode="same")


def one(x, sf, bad):
    sos, xf = SD.detect_so(x, sf, bad, th=None); phase = np.angle(signal.hilbert(xf))
    r = rms_of(x, sf)[~bad]
    if r.size < 3 * int(round(SD.RMS_WIN * sf)):
        return None
    m, s = r.mean(), r.std(); out = dict(n_SO=len(sos))
    for k in KS:
        sps = SD.detect_spindles(x, sf, bad, th=dict(sp_th=m + k * s, sp_hi=m + 9 * s))
        cen = np.array(sorted(sp["center"] for sp in sps), int)
        g = np.abs(phase[cen]) <= np.pi / 2 if cen.size else np.zeros(0, bool)
        n_cp, _ = RD.couple_vec(sos, cen, g)
        out["n_spindle_k%g" % k] = len(sps); out["n_coupled_k%g" % k] = n_cp
    return out


def main():
    rows, t0 = [], time.time()
    for i, (_, row) in enumerate(RD.INV.iterrows(), 1):
        try:
            raw = RD.mne.io.read_raw_fif(row.fif_path, preload=True)
            if "sleep" in raw.ch_names:
                sf = raw.info["sfreq"]
                for b, br in {b: RD.block_concat(raw, b) for b in RD.BLOCKS}.items():
                    if br is None:
                        continue
                    bad = bad_annot_mask(br); dur = float(br.times[-1] - br.times[0])
                    for ch in [c for c in RD.CH if c in br.ch_names]:
                        o = one(br.get_data(picks=ch)[0] * 1e6, sf, bad)
                        if o is not None:
                            rows.append(dict(session_id=row["Session ID"], participant=row.Participant, arm=row.Condition, cohort=row.Cohort, win=RD.BLOCKS[b], ch=ch, duration_sec=dur, **o))
        except Exception as exc:                                   # noqa: BLE001
            print("FAIL", row["Session ID"], exc, flush=True)
        if i % 10 == 0 or i == len(RD.INV):
            el = time.time() - t0
            print("%d/%d  %.1f min  remaining ~%.1f min" % (i, len(RD.INV), el / 60, el / i * (len(RD.INV) - i) / 60), flush=True)
    d = pd.DataFrame(rows); d.to_csv(T / "REV_spindle_threshold_sweep_cells.csv", index=False); print("saved", d.shape)


if __name__ == "__main__":
    main()
