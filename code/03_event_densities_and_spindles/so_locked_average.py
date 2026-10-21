# -*- coding: utf-8 -*-
"""Frontal SO-locked average — SO waveform + spindle envelope, by condition.

Reproduces the standard figure of the reference papers (spindle-band envelope over the mean SO waveform)
with our data. Four frontal electrodes (Fp1 Fp2 F3 F4), stimulation window.

Outputs
  tables/SO_locked_frontal.npz   mean waveform, envelope, n
  figures/Fig1c_so_locked.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import mne                                                        # noqa: E402
from scipy.signal import butter, filtfilt, hilbert                # noqa: E402

mne.set_log_level("ERROR")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402
_R = LP.PROJECT_ROOT
P = _R if (_R / "outputs/paper").exists() else _R / "이전"
OUT = P / "outputs/paper"
TBL = _R / "Final analysis/tables"
TBL.mkdir(parents=True, exist_ok=True)

SF = 125.0
REGIONS = {"frontal": ["Fp1", "Fp2", "F3", "F4"],
           "parietal": ["P3", "P4"]}
FRONTAL = REGIONS["frontal"]
STAGE = {"W": 0, "R": 1, "N1": 2, "N2": 3, "N3": 4}
HALF = 1.5                      # ±1.5 s epoch
NS = int(HALF * SF)
SO_LO, SO_HI = 0.3, 1.25        # Staresina
SP_LO, SP_HI = 12.0, 16.0


def bp(x, lo, hi, order=3):
    b, a = butter(order, [lo / (SF / 2), hi / (SF / 2)], btype="band")
    return filtfilt(b, a, x)


def detect_so(xf):
    """Staresina 2023 rule — positive-to-negative zero crossings, 0.8–2.0 s,
    both trough and trough-to-peak exceed mean+1SD. Returns the trough samples."""
    s = np.sign(xf)
    s[s == 0] = 1
    zc = np.flatnonzero(np.diff(s) < 0)          # positive -> negative
    cand = []
    for a, b in zip(zc[:-1], zc[1:]):
        dur = (b - a) / SF
        if not (0.8 <= dur <= 2.0):
            continue
        seg = xf[a:b]
        tr = int(np.argmin(seg))
        cand.append((a + tr, float(seg[tr]), float(seg.max() - seg.min())))
    if len(cand) < 5:
        return np.array([], int)
    trough = np.array([c[1] for c in cand])
    ptp = np.array([c[2] for c in cand])
    k = ((trough < trough.mean() - trough.std()) &
         (ptp > ptp.mean() + ptp.std()))
    return np.array([cand[i][0] for i in np.flatnonzero(k)], int)


def session_average(fif, arm):
    """Per-region SO-locked mean waveform and spindle envelope for one recording."""
    raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    if "sleep" not in raw.ch_names:
        return None
    slp = raw.get_data(picks="sleep")[0]
    nrem = np.isin(slp, [STAGE["N2"], STAGE["N3"]])
    if nrem.sum() < SF * 60:
        return None
    out = {}
    for reg, want in REGIONS.items():
        chs = [c for c in want if c in raw.ch_names]
        acc_w, acc_e, n = [], [], 0
        for ch in chs:
            x = -raw.get_data(picks=ch)[0] * 1e6   # µV, polarity inverted (validation V2)
            xso = bp(x, SO_LO, SO_HI)
            env = np.abs(hilbert(bp(x, SP_LO, SP_HI)))
            tro = detect_so(xso)
            tro = tro[nrem[tro]] if len(tro) else tro
            tro = tro[(tro > NS) & (tro < len(x) - NS - 1)]
            if len(tro) < 5:
                continue
            idx = tro[:, None] + np.arange(-NS, NS + 1)[None, :]
            acc_w.append(xso[idx].mean(0))
            acc_e.append(env[idx].mean(0))
            n += len(tro)
        if acc_w:
            out[reg] = (np.mean(acc_w, 0), np.mean(acc_e, 0), n)
    return (out, arm) if out else None


def main(limit=None):
    inv = pd.read_csv(TBL / "FIF_index.csv")
    rows = inv.dropna(subset=["fif_path"])[
        ["fif_path", "Condition", "Cohort"]]
    if limit:
        rows = rows.head(limit)
    res = {(reg, arm): [] for reg in REGIONS for arm in
           ("Treatment", "Sham")}
    cnt = {k: 0 for k in res}
    done = bad = 0
    for r in rows.itertuples(index=False):
        f = Path(r.fif_path)
        if not f.exists():
            bad += 1
            continue
        try:
            got = session_average(f, r.Condition)
        except Exception:
            bad += 1
            continue
        if got is None:
            bad += 1
            continue
        per, arm = got
        key = "Treatment" if str(arm).lower().startswith("t") else "Sham"
        for reg, (w, e, n) in per.items():
            res[(reg, key)].append((w, e))
            cnt[(reg, key)] += n
        done += 1
        if done % 40 == 0:
            print("  %d recordings done" % done, flush=True)

    t = np.arange(-NS, NS + 1) / SF
    pack = {"t": t}
    for (reg, arm), v in res.items():
        pack["%s_%s_wave" % (reg, arm)] = np.array([a for a, _ in v])
        pack["%s_%s_env" % (reg, arm)] = np.array([b for _, b in v])
        pack["%s_%s_nev" % (reg, arm)] = cnt[(reg, arm)]
    np.savez(TBL / "SO_locked_regions.npz", **pack)
    print("recordings used %d · skipped %d" % (done, bad))
    for (reg, arm), n in cnt.items():
        print("  %-9s %-10s recordings %3d · SO events %6d"
              % (reg, arm, len(res[(reg, arm)]), n))
    print("[saved] %s" % (TBL / "SO_locked_regions.npz"))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
