# -*- coding: utf-8 -*-
"""Compute spatial extent with two thresholds: window-specific and recording-wide (reviewer: per-channel relative thresholds mix in the noise level).
N2+N3, 230 recordings, 13 channels. Extent per SO = number of channels, including its own, with a trough within ±400 ms.
Outputs tables/REV_extent_thresholds_cells.csv (coupling probability per session×window×extent), REV_extent_thresholds_session.csv (mean extent and widespread fraction per session×window)"""
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
import rev_redetect_n2n3 as RN                                    # noqa: E402
import staresina_detector as SD                                   # noqa: E402
from tvns.jy_source import bad_annot_mask                         # noqa: E402

T = Path(r"D:/SO-Spindle논문/Final analysis/tables")


def events(x, sf, bad, th):
    sos, xf = SD.detect_so(x, sf, bad, th=th); sps = SD.detect_spindles(x, sf, bad, th=th)
    phase = np.angle(signal.hilbert(xf)); cen = np.array(sorted(s["center"] for s in sps), int)
    good = np.abs(phase[cen]) <= np.pi / 2 if cen.size else np.zeros(0, bool)
    out = []
    for s in sos:
        lo, hi = np.searchsorted(cen, s["onset"]), np.searchsorted(cen, s["end"])
        out.append((s["trough"] if "trough" in s else s["onset"], bool(good[lo:hi].any())))
    return out


def main():
    rows, sess, t0 = [], [], time.time()
    inv = RD.INV
    for i, (_, row) in enumerate(inv.iterrows(), 1):
        try:
            raw = RD.mne.io.read_raw_fif(row.fif_path, preload=True); sf = raw.info["sfreq"]; w400 = int(0.4 * sf)
            blocks = {b: RN.block_concat_n2n3(raw, b) for b in RD.BLOCKS}
            if blocks["stimulation 1"] is None:
                continue
            th_rw = {}
            for ch in RD.CH:
                xs, bs = [], []
                for b, br in blocks.items():
                    if br is None or ch not in br.ch_names:
                        continue
                    xs.append(br.get_data(picks=ch)[0] * 1e6); bs.append(bad_annot_mask(br))
                if xs:
                    th_rw[ch] = SD.session_thresholds(np.concatenate(xs), sf, np.concatenate(bs))
            for b, br in blocks.items():
                if br is None:
                    continue
                bad = bad_annot_mask(br); chs = [c for c in RD.CH if c in br.ch_names]; nch = len(chs)
                for scope in ("window-specific", "recording-wide"):
                    ev = {ch: events(br.get_data(picks=ch)[0] * 1e6, sf, bad, None if scope == "window-specific" else th_rw.get(ch)) for ch in chs}
                    troughs = {ch: np.array(sorted(t for t, _ in ev[ch]), int) for ch in chs}
                    ext, cp = [], []
                    for ch in chs:
                        for t, c in ev[ch]:
                            n = 1 + sum(1 for o in chs if o != ch and troughs[o].size and np.any(np.abs(troughs[o] - t) <= w400))
                            ext.append(n); cp.append(c)
                    ext, cp = np.array(ext), np.array(cp)
                    if ext.size == 0:
                        continue
                    for lvl in range(1, nch + 1):
                        m = ext == lvl
                        if m.any():
                            rows.append(dict(session_id=row["Session ID"], participant=row.Participant, arm=row.Condition, cohort=row.Cohort, win=RD.BLOCKS[b], scope=scope, extent=lvl, n_SO=int(m.sum()), n_coupled=int(cp[m].sum())))
                    sess.append(dict(session_id=row["Session ID"], participant=row.Participant, arm=row.Condition, cohort=row.Cohort, win=RD.BLOCKS[b], scope=scope, n_SO=int(ext.size),
                                     mean_extent=float(ext.mean()), frac_widespread=float((ext >= nch / 2).mean()), n_channels=nch, coupled_prob=float(cp.mean())))
        except Exception as exc:                                   # noqa: BLE001
            print("FAIL", row["Session ID"], exc, flush=True)
        if i % 20 == 0 or i == len(inv):
            el = time.time() - t0; print("%d/%d  %.1f min  remaining ~%.1f min" % (i, len(inv), el / 60, el / i * (len(inv) - i) / 60), flush=True)
    pd.DataFrame(rows).to_csv(T / "REV_extent_thresholds_cells.csv", index=False); pd.DataFrame(sess).to_csv(T / "REV_extent_thresholds_session.csv", index=False)
    print("saved", len(rows), len(sess))


if __name__ == "__main__":
    main()
