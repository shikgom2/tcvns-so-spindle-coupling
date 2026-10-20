# -*- coding: utf-8 -*-
"""For the Figure 4 rewrite — recount SOs, spindles and coupled events in N2+N3 under per-window and recording-wide thresholds (surrogates and PSD omitted).
Output  Final analysis/tables/REV_redetect_cells_N2N3.csv
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:/HJ/vns/04_SHARED/code")
sys.path.insert(0, r"D:/SO-Spindle논문/이전/reanalysis_code")
sys.path.insert(0, r"D:/SO-Spindle논문/code/paper")
import mne                                                        # noqa: E402
mne.set_log_level("ERROR")
from tvns.jy_source import bad_annot_mask, group_consecutive       # noqa: E402
import staresina_detector as SD                                   # noqa: E402
import rev_redetect as RV                                         # noqa: E402

R = Path(r"D:/SO-Spindle논문/Final analysis/tables")


def block_concat_n2n3(raw, block):
    ann = next((a for a in raw.annotations if a["description"] == block), None)
    if ann is None:
        return None
    seg = raw.copy().crop(tmin=ann["onset"], tmax=min(ann["onset"] + ann["duration"], raw.times[-1]))
    slp = seg.get_data(picks="sleep")[0]
    idx = np.flatnonzero(np.isin(slp, [RV.MAP["N2"], RV.MAP["N3"]]))
    if idx.size == 0:
        return None
    parts = [seg.copy().crop(tmin=seg.times[g[0]], tmax=seg.times[g[-1]]) for g in group_consecutive(idx) if len(g)]
    return mne.concatenate_raws(parts) if parts else None


def one(x, sf, bad, th_rw):
    out = {}
    sos, xf = SD.detect_so(x, sf, bad, th=None); sps = SD.detect_spindles(x, sf, bad, th=None)
    phase = np.angle(signal.hilbert(xf)); cen = np.array(sorted(s["center"] for s in sps), int)
    g = np.abs(phase[cen]) <= np.pi / 2 if cen.size else np.zeros(0, bool)
    n_cp, _ = RV.couple_vec(sos, cen, g)
    out.update(n_SO=len(sos), n_spindle=len(sps), n_coupled=n_cp)
    if th_rw is not None:
        sos2, _ = SD.detect_so(x, sf, bad, th=th_rw); sps2 = SD.detect_spindles(x, sf, bad, th=th_rw)
        cen2 = np.array(sorted(s["center"] for s in sps2), int)
        g2 = np.abs(phase[cen2]) <= np.pi / 2 if cen2.size else np.zeros(0, bool)
        n2, _ = RV.couple_vec(sos2, cen2, g2)
        out.update(n_SO_rw=len(sos2), n_spindle_rw=len(sps2), n_coupled_rw=n2)
    return out


def session(row):
    raw = mne.io.read_raw_fif(row.fif_path, preload=True)
    sf = raw.info["sfreq"]
    blocks = {b: block_concat_n2n3(raw, b) for b in RV.BLOCKS}
    if blocks["stimulation 1"] is None:
        return []
    th_rw = {}
    for ch in RV.CH:
        xs, bs = [], []
        for b, br in blocks.items():
            if br is None or ch not in br.ch_names:
                continue
            xs.append(br.get_data(picks=ch)[0] * 1e6); bs.append(bad_annot_mask(br))
        if xs:
            th_rw[ch] = SD.session_thresholds(np.concatenate(xs), sf, np.concatenate(bs))
    rows = []
    for b, br in blocks.items():
        if br is None:
            continue
        bad = bad_annot_mask(br); dur = float(br.times[-1] - br.times[0])
        for ch in RV.CH:
            if ch not in br.ch_names:
                continue
            o = one(br.get_data(picks=ch)[0] * 1e6, sf, bad, th_rw.get(ch))
            rows.append(dict(session_id=row["Session ID"], participant=row.Participant, arm=row.Condition, cohort=row.Cohort,
                             win=RV.BLOCKS[b], ch=ch, duration_sec=dur, **o))
    return rows


def main():
    rows, t0 = [], time.time()
    for i, (_, row) in enumerate(RV.INV.iterrows(), 1):
        try:
            rows += session(row)
        except Exception as exc:                                   # noqa: BLE001
            print("FAIL", row["Session ID"], exc, flush=True)
        if i % 20 == 0 or i == len(RV.INV):
            el = time.time() - t0
            print("%d/%d  %.1f min  remaining ~%.1f min" % (i, len(RV.INV), el / 60, el / i * (len(RV.INV) - i) / 60), flush=True)
    pd.DataFrame(rows).to_csv(R / "REV_redetect_cells_N2N3.csv", index=False)
    print("saved", len(rows))


if __name__ == "__main__":
    main()
