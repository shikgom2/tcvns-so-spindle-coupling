# -*- coding: utf-8 -*-
"""Coupled-SO and spindle counts under recording-wide thresholds (N2+N3) for spindle threshold multipliers k = 1, 1.5, 2, 2.5 SD (sensitivity of the Figure 4 density analysis).
SO threshold unchanged (recording-wide mean+1SD). Output tables/REV_density_threshold_sweep_cells.csv"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402
sys.path.insert(0, str(LP.PROJECT_ROOT / "code/paper"))
import rev_redetect as RD                                         # noqa: E402
import rev_redetect_n2n3 as RN                                    # noqa: E402
import staresina_detector as SD                                   # noqa: E402
from tvns.jy_source import bad_annot_mask                         # noqa: E402

T = LP.PROJECT_ROOT / "Final analysis/tables"
KS = (1.0, 1.5, 2.0, 2.5)


def rms_of(x, sf):
    xf = SD._bp(x, sf, SD.SP_BAND); w = max(1, int(round(SD.RMS_WIN * sf))); k = np.ones(w) / w
    r = np.sqrt(np.convolve(xf ** 2, k, mode="same")); return np.convolve(r, k, mode="same")


def main():
    rows, t0 = [], time.time()
    inv = RD.INV
    for i, (_, row) in enumerate(inv.iterrows(), 1):
        try:
            raw = RD.mne.io.read_raw_fif(row.fif_path, preload=True); sf = raw.info["sfreq"]
            blocks = {b: RN.block_concat_n2n3(raw, b) for b in RD.BLOCKS}
            if blocks["stimulation 1"] is None:
                continue
            thr = {}
            for ch in RD.CH:
                xs, bs = [], []
                for b, br in blocks.items():
                    if br is None or ch not in br.ch_names:
                        continue
                    xs.append(br.get_data(picks=ch)[0] * 1e6); bs.append(bad_annot_mask(br))
                if xs:
                    xa, ba = np.concatenate(xs), np.concatenate(bs); th = SD.session_thresholds(xa, sf, ba)
                    r = rms_of(xa, sf)[~ba]; thr[ch] = (th, r.mean(), r.std())
            for b, br in blocks.items():
                if br is None:
                    continue
                bad = bad_annot_mask(br); dur = float(br.times[-1] - br.times[0])
                for ch in RD.CH:
                    if ch not in br.ch_names or ch not in thr:
                        continue
                    th, m, s = thr[ch]; x = br.get_data(picks=ch)[0] * 1e6
                    sos, xf = SD.detect_so(x, sf, bad, th=th); phase = np.angle(signal.hilbert(xf))
                    o = dict(session_id=row["Session ID"], participant=row.Participant, arm=row.Condition, cohort=row.Cohort, win=RD.BLOCKS[b], ch=ch, duration_sec=dur, n_SO=len(sos))
                    for k in KS:
                        sps = SD.detect_spindles(x, sf, bad, th=dict(sp_th=m + k * s, sp_hi=m + 9 * s))
                        cen = np.array(sorted(sp["center"] for sp in sps), int); g = np.abs(phase[cen]) <= np.pi / 2 if cen.size else np.zeros(0, bool)
                        o["n_spindle_k%g" % k] = len(sps); o["n_coupled_k%g" % k] = RD.couple_vec(sos, cen, g)[0]
                    rows.append(o)
        except Exception as exc:                                   # noqa: BLE001
            print("FAIL", row["Session ID"], exc, flush=True)
        if i % 20 == 0 or i == len(inv):
            el = time.time() - t0; print("%d/%d  %.1f min  remaining ~%.1f min" % (i, len(inv), el / 60, el / i * (len(inv) - i) / 60), flush=True)
    d = pd.DataFrame(rows); d.to_csv(T / "REV_density_threshold_sweep_cells.csv", index=False); print("saved", d.shape)


if __name__ == "__main__":
    main()
