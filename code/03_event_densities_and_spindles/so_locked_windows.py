# -*- coding: utf-8 -*-
"""SO-locked averages by window and region — for Result 5.

For each recording, in four regions × three windows (baseline / stimulation 1 / recovery 1), computes
  · the mean SO waveform
  · the mean spindle-band envelope
  · coupled/uncoupled SO density (per minute)
An SO is classed as coupled when a spindle-envelope peak falls inside the SO window and the
SO phase at that time lies within ±90° of the up-state.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402

mne.set_log_level("ERROR")
_R = LP.PROJECT_ROOT
TBL = _R / "Final analysis/tables"
SF = 125.0
REGIONS = {"Frontal": ["Fp1", "Fp2", "F3", "F4"],
           "Temporal": ["F7", "F8", "T3", "T4"],
           "Central": ["C3", "Cz", "C4"],
           "Parietal": ["P3", "P4"]}
WINS = {"before": "baseline", "during": "stimulation 1",
        "after": "recovery 1"}
STAGE = {"W": 0, "R": 1, "N1": 2, "N2": 3, "N3": 4}
NS = int(1.5 * SF)
SO_LO, SO_HI, SP_LO, SP_HI = 0.3, 1.25, 12.0, 16.0


def bp(x, lo, hi, order=3):
    b, a = butter(order, [lo / (SF / 2), hi / (SF / 2)], btype="band")
    return filtfilt(b, a, x)


def detect_so(xf):
    """Staresina rule. List of (trough_idx, start, end)."""
    s = np.sign(xf)
    s[s == 0] = 1
    zc = np.flatnonzero(np.diff(s) < 0)
    cand = []
    for a, b in zip(zc[:-1], zc[1:]):
        if not (0.8 <= (b - a) / SF <= 2.0):
            continue
        seg = xf[a:b]
        tr = int(np.argmin(seg))
        cand.append((a + tr, a, b, float(seg[tr]),
                     float(seg.max() - seg.min())))
    if len(cand) < 5:
        return []
    tro = np.array([c[3] for c in cand])
    ptp = np.array([c[4] for c in cand])
    k = ((tro < tro.mean() - tro.std()) & (ptp > ptp.mean() + ptp.std()))
    return [cand[i][:3] for i in np.flatnonzero(k)]


def spindle_peaks(env):
    th = env.mean() + env.std()
    above = env > th
    if not above.any():
        return np.array([], int)
    d = np.diff(above.astype(int))
    st = np.flatnonzero(d == 1) + 1
    en = np.flatnonzero(d == -1) + 1
    if above[0]:
        st = np.r_[0, st]
    if above[-1]:
        en = np.r_[en, len(env)]
    return np.array([a + int(np.argmax(env[a:b])) for a, b in zip(st, en)
                     if 0.4 <= (b - a) / SF <= 3.0], int)


def session(fif):
    raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    if "sleep" not in raw.ch_names:
        return None
    ann = raw.annotations
    spans = {}
    for key, lab in WINS.items():
        m = [i for i, d in enumerate(ann.description) if d == lab]
        if not m:
            continue
        i = m[0]
        spans[key] = (int(ann.onset[i] * SF),
                      int((ann.onset[i] + ann.duration[i]) * SF))
    if len(spans) < 3:
        return None
    nrem = np.isin(raw.get_data(picks="sleep")[0],
                   [STAGE["N2"], STAGE["N3"]])
    out = {}
    for reg, want in REGIONS.items():
        chs = [c for c in want if c in raw.ch_names]
        if not chs:
            continue
        sig = {}
        for ch in chs:
            x = -raw.get_data(picks=ch)[0] * 1e6
            xso = bp(x, SO_LO, SO_HI)
            sig[ch] = (xso, np.angle(hilbert(xso)),
                       np.abs(hilbert(bp(x, SP_LO, SP_HI))))
        for wkey, (a0, a1) in spans.items():
            wav, env_, nco, nun, mins = [], [], 0, 0, 0.0
            for ch in chs:
                xso, ph, env = sig[ch]
                ok = np.zeros(len(xso), bool)
                ok[a0:a1] = True
                ok &= nrem
                if ok.sum() < SF * 20:
                    continue
                mins = max(mins, ok.sum() / SF / 60.0)
                sos = [t for t in detect_so(xso)
                       if ok[t[0]] and NS < t[0] < len(xso) - NS - 1]
                if len(sos) < 5:
                    continue
                pk = spindle_peaks(env)
                pk = pk[ok[pk]] if len(pk) else pk
                tro = np.array([t[0] for t in sos])
                idx = tro[:, None] + np.arange(-NS, NS + 1)[None, :]
                wav.append(xso[idx].mean(0))
                env_.append(env[idx].mean(0))
                for t0, s0, e0 in sos:
                    inside = pk[(pk >= s0) & (pk < e0)] if len(pk) else []
                    hit = any(abs(np.angle(np.exp(1j * ph[p]))) <= np.pi / 2
                              for p in inside)
                    nco += int(hit)
                    nun += int(not hit)
            if wav and mins > 0:
                out[(reg, wkey)] = dict(
                    wave=np.mean(wav, 0), env=np.mean(env_, 0),
                    coupled_den=nco / len(chs) / mins,
                    uncoupled_den=nun / len(chs) / mins, n_so=nco + nun)
    return out or None


def main(limit=None):
    inv = pd.read_csv(TBL / "FIF_index.csv").dropna(subset=["fif_path"])
    if limit:
        inv = inv.head(limit)
    acc, rows, done, bad = {}, [], 0, 0
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
        for (reg, wkey), d in got.items():
            acc.setdefault((reg, wkey, arm), {"w": [], "e": []})
            acc[(reg, wkey, arm)]["w"].append(d["wave"])
            acc[(reg, wkey, arm)]["e"].append(d["env"])
            rows.append(dict(session=row.fif_path.split("\\")[-1],
                             arm=arm, cohort=row.Cohort, region=reg,
                             win=wkey, coupled_den=d["coupled_den"],
                             uncoupled_den=d["uncoupled_den"],
                             n_so=d["n_so"]))
        done += 1
        if done % 40 == 0:
            print("  %d recordings" % done, flush=True)
    pack = {"t": np.arange(-NS, NS + 1) / SF}
    for (reg, wkey, arm), v in acc.items():
        pack["%s|%s|%s|wave" % (reg, wkey, arm)] = np.array(v["w"])
        pack["%s|%s|%s|env" % (reg, wkey, arm)] = np.array(v["e"])
    np.savez(TBL / "SO_locked_windows.npz", **pack)
    D = pd.DataFrame(rows)
    D.to_csv(TBL / "R5_density_by_window.csv", index=False,
             encoding="utf-8-sig")
    print("recordings used %d · skipped %d" % (done, bad))
    if len(D):
        print(D.groupby(["region", "win", "arm"])
              [["uncoupled_den", "coupled_den"]].mean().round(3).to_string())
    print("[saved] %s" % (TBL / "SO_locked_windows.npz"))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
