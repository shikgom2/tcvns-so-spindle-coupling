# -*- coding: utf-8 -*-
"""For testing the arousal alternative hypothesis — per recording and window (baseline / stimulation 1 / recovery 1):

  · W+N1 fraction, N2 fraction, N3 fraction (sleep channel: 0=W 1=R 2=N1 3=N2 4=N3)
  · fraction of BAD_* segments (proxy for movement / amplitude artefacts)
  · relative alpha (8–12 Hz) and beta (16–30 Hz) power in N2 (mean of 13 channels; arousal proxy)
  · heart rate from ECG R-peaks (N2 segments)
Output: Final analysis/tables/ALT_G_arousal_windows.csv (one row = recording × window)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import signal

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

R = Path(r"D:/SO-Spindle논문")
INV = pd.read_csv(R / "Final analysis/tables/FIF_index.csv")
OUT = R / "Final analysis/tables/ALT_G_arousal_windows.csv"
WINS = {"baseline": "pre", "stimulation 1": "stim", "recovery 1": "post"}
EEG = ["Fp1", "F7", "F3", "T3", "C3", "Cz", "P3", "Fp2", "F4", "F8", "C4", "T4", "P4"]
EP = 30.0


def heart_rate(ecg, sf):
    """Simple R-peak detection: 5–30 Hz band-pass, then absolute value, 0.4 s minimum spacing."""
    if ecg.size < sf * 10:
        return np.nan
    b, a = signal.butter(3, [5 / (sf / 2), 30 / (sf / 2)], "band")
    x = signal.filtfilt(b, a, ecg - np.nanmean(ecg))
    x = np.abs(x)
    thr = np.percentile(x, 98) * 0.5
    pk, _ = signal.find_peaks(x, height=thr, distance=int(0.4 * sf))
    if pk.size < 20:
        return np.nan
    rr = np.diff(pk) / sf
    rr = rr[(rr > 0.4) & (rr < 2.0)]
    return 60 / np.median(rr) if rr.size > 10 else np.nan


def one(row):
    raw = mne.io.read_raw_fif(row.fif_path, preload=True)
    sf = raw.info["sfreq"]
    ann = raw.annotations
    win = {d: (o, o + du) for d, o, du in zip(ann.description, ann.onset, ann.duration) if d in WINS}
    bad = [(o, o + du) for d, o, du in zip(ann.description, ann.onset, ann.duration) if d.startswith("BAD")]
    names = raw.ch_names
    sleep = raw.get_data(picks=[names.index("sleep")])[0]
    eeg = raw.get_data(picks=[names.index(c) for c in EEG]) * 1e6
    ecg = raw.get_data(picks=[names.index("ecg")])[0] if "ecg" in names else None
    t = raw.times
    out = []
    for name, (a, b) in win.items():
        m = (t >= a) & (t < b)
        if m.sum() < sf * 60:
            continue
        s = sleep[m]
        n = s.size
        # stage fractions (30 s epoch scoring)
        wake_n1 = np.mean((s == 0) | (s == 2)); n2 = np.mean(s == 3); n3 = np.mean(s == 4); rem = np.mean(s == 1)
        # BAD fraction
        badmask = np.zeros(n, bool); tt = t[m]
        for o, e in bad:
            badmask |= (tt >= o) & (tt < e)
        badfrac = badmask.mean()
        # spectrum of N2 & not-BAD segments
        good = (s == 3) & ~badmask
        alpha = beta = delta = np.nan; hr = np.nan
        if good.sum() >= sf * 60:
            x = eeg[:, m][:, good]
            f, p = signal.welch(x, sf, nperseg=int(4 * sf), axis=1)
            tot = p[:, (f >= 0.5) & (f <= 30)].sum(1)
            alpha = np.mean(p[:, (f >= 8) & (f <= 12)].sum(1) / tot)
            beta = np.mean(p[:, (f >= 16) & (f <= 30)].sum(1) / tot)
            delta = np.mean(p[:, (f >= 0.5) & (f <= 4)].sum(1) / tot)
            if ecg is not None:
                hr = heart_rate(ecg[m][good], sf)
        out.append(dict(session_id=row["Session ID"], arm=row.Condition, win=WINS[name],
                        win_min=(b - a) / 60, wake_n1=wake_n1, n2=n2, n3=n3, rem=rem,
                        bad_frac=badfrac, alpha_rel=alpha, beta_rel=beta, delta_rel=delta, hr_bpm=hr))
    return out


def main():
    rows = []
    for i, row in INV.iterrows():
        try:
            rows += one(row)
        except Exception as e:  # noqa: BLE001
            print("FAIL", row["Session ID"], e, file=sys.stderr)
        if i % 20 == 0:
            print(i, "/", len(INV), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("saved", OUT, df.shape)


if __name__ == "__main__":
    main()
