# -*- coding: utf-8 -*-
"""Re-detection for the reviewer response — one pass over each .fif extracts all outputs together.

  1. Reproduction            per-window (block) thresholds, Staresina detector, N2, |phase|<=90 rule  → must match S1
  2. Recording-wide threshold pool N2 of the three windows, compute thresholds once, then detect per window → reviewer point 5
  3. Chance level            circularly shift spindle centres within the window (200 times), then couple with the same rule → reviewer point 7
  4. Opposite polarity       |phase - 180|<=90 rule                                                    → reviewer point 7
  5. High-resolution PSD     per window and channel, Welch 8 s (0.125 Hz), artefact-free N2             → reviewer point 5

Output  Final analysis/tables/REV_redetect_cells.csv   (session x window x channel)
        Final analysis/tables/REV_psd.npz              (session x window x channel x frequency)
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402
sys.path.insert(0, str(LP.PROJECT_ROOT / "이전/reanalysis_code"))
import mne                                                        # noqa: E402
mne.set_log_level("ERROR")
from tvns.jy_source import bad_annot_mask, group_consecutive, EEG_CFG   # noqa: E402
import staresina_detector as SD                                   # noqa: E402

R = LP.PROJECT_ROOT / "Final analysis/tables"
INV = pd.read_csv(R / "FIF_index.csv")
BLOCKS = {"baseline": "pre", "stimulation 1": "stim", "recovery 1": "post"}
CH = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "Cz", "C4", "P3", "P4"]
MAP = EEG_CFG.sleep_staging.stage_mapping
N_SURR = 200
RNG = np.random.default_rng(20261016)
NPERSEG = 1000                      # 8 s at 125 Hz -> 0.125 Hz
FREQS = np.fft.rfftfreq(NPERSEG, 1 / 125.0)


def block_concat(raw, block):
    ann = next((a for a in raw.annotations if a["description"] == block), None)
    if ann is None:
        return None
    seg = raw.copy().crop(tmin=ann["onset"],
                          tmax=min(ann["onset"] + ann["duration"], raw.times[-1]))
    slp = seg.get_data(picks="sleep")[0]
    idx = np.flatnonzero(slp == MAP["N2"])
    if idx.size == 0:
        return None
    parts = [seg.copy().crop(tmin=seg.times[g[0]], tmax=seg.times[g[-1]])
             for g in group_consecutive(idx) if len(g)]
    return mne.concatenate_raws(parts) if parts else None


def couple_vec(sos, cen, good):
    """Number of SOs with at least one 'good' (phase-criterion-satisfying) spindle centre inside the SO interval.
    cen is a sorted array of sample indices, good is a bool array of the same length."""
    if not sos or cen.size == 0:
        return 0, 0
    on = np.array([s["onset"] for s in sos]); en = np.array([s["end"] for s in sos])
    lo = np.searchsorted(cen, on, "left"); hi = np.searchsorted(cen, en, "left")
    cs = np.r_[0, np.cumsum(good)]
    n_good = int(((cs[hi] - cs[lo]) > 0).sum())
    n_any = int(((hi - lo) > 0).sum())
    return n_good, n_any


def one_channel(x, sf, bad, th_rw):
    out = {}
    # ── 1. Per-window thresholds (exactly as in the manuscript)
    sos, xf = SD.detect_so(x, sf, bad, th=None)
    sps = SD.detect_spindles(x, sf, bad, th=None)
    phase = np.angle(signal.hilbert(xf))
    cen = np.array(sorted(s["center"] for s in sps), int)
    g_up = np.abs(phase[cen]) <= np.pi / 2 if cen.size else np.zeros(0, bool)
    g_dn = np.abs(np.angle(np.exp(1j * (phase[cen] - np.pi)))) <= np.pi / 2 if cen.size else np.zeros(0, bool)
    n_cp, n_any = couple_vec(sos, cen, g_up)
    n_cp_opp, _ = couple_vec(sos, cen, g_dn)
    out.update(n_SO=len(sos), n_spindle=len(sps), n_coupled=n_cp, n_any_spindle=n_any,
               n_coupled_opposite=n_cp_opp)
    # Mean vector of spindle-centre phases (for judging polarity)
    if cen.size:
        v = np.exp(1j * phase[cen]).mean()
        out.update(sp_phase_mrl=float(abs(v)), sp_phase_deg=float(np.degrees(np.angle(v))),
                   n_sp_phase=int(cen.size))
    # ── 3. Chance level: circular shift of spindle centres
    L = len(x)
    if sos and cen.size and L > 20 * sf:
        shifts = RNG.integers(int(10 * sf), L - int(10 * sf), N_SURR)
        ns = np.empty(N_SURR)
        for i, k in enumerate(shifts):
            c2 = np.sort((cen + k) % L)
            ns[i] = couple_vec(sos, c2, np.abs(phase[c2]) <= np.pi / 2)[0]
        out.update(n_coupled_surr_mean=float(ns.mean()), n_coupled_surr_sd=float(ns.std()),
                   n_coupled_surr_p=float((ns >= n_cp).mean()))
    # ── 2. Recording-wide thresholds
    if th_rw is not None:
        sos2, xf2 = SD.detect_so(x, sf, bad, th=th_rw)
        sps2 = SD.detect_spindles(x, sf, bad, th=th_rw)
        cen2 = np.array(sorted(s["center"] for s in sps2), int)
        g2 = np.abs(phase[cen2]) <= np.pi / 2 if cen2.size else np.zeros(0, bool)
        n2, _ = couple_vec(sos2, cen2, g2)
        out.update(n_SO_rw=len(sos2), n_spindle_rw=len(sps2), n_coupled_rw=n2)
    # ── 5. High-resolution PSD (artefacts excluded)
    good = ~bad
    if good.sum() >= NPERSEG * 3:
        f, p = signal.welch(x[good], sf, nperseg=NPERSEG)
        out["psd"] = p.astype(np.float32)
    return out


def session(row):
    raw = mne.io.read_raw_fif(row.fif_path, preload=True)
    if "sleep" not in raw.ch_names:
        return [], {}
    sf = raw.info["sfreq"]
    blocks = {b: block_concat(raw, b) for b in BLOCKS}
    if blocks["stimulation 1"] is None:
        return [], {}
    # Recording-wide thresholds: concatenate N2 of the three windows per channel
    th_rw = {}
    for ch in CH:
        xs, bs = [], []
        for b, br in blocks.items():
            if br is None or ch not in br.ch_names:
                continue
            xs.append(br.get_data(picks=ch)[0] * 1e6); bs.append(bad_annot_mask(br))
        if xs:
            th_rw[ch] = SD.session_thresholds(np.concatenate(xs), sf, np.concatenate(bs))
    rows, psd = [], {}
    for b, br in blocks.items():
        if br is None:
            continue
        bad = bad_annot_mask(br)
        dur = float(br.times[-1] - br.times[0])
        for ch in CH:
            if ch not in br.ch_names:
                continue
            x = br.get_data(picks=ch)[0] * 1e6
            o = one_channel(x, sf, bad, th_rw.get(ch))
            p = o.pop("psd", None)
            if p is not None:
                psd[(BLOCKS[b], ch)] = p
            rows.append(dict(session_id=row["Session ID"], participant=row.Participant,
                             arm=row.Condition, cohort=row.Cohort, win=BLOCKS[b], ch=ch,
                             duration_sec=dur, bad_frac=float(bad.mean()), **o))
    return rows, psd


def main(limit=None):
    inv = INV if limit is None else INV.head(limit)
    rows, P = [], {}
    t0 = time.time()
    for i, (_, row) in enumerate(inv.iterrows(), 1):
        sid = row["Session ID"]
        try:
            r, p = session(row)
        except Exception as exc:                                   # noqa: BLE001
            print("FAIL", sid, exc, flush=True); r, p = [], {}
        rows += r
        for (w, ch), v in p.items():
            P["%s|%s|%s" % (sid, w, ch)] = v
        if i % 10 == 0 or i == len(inv):
            el = time.time() - t0
            print("%d/%d  %.1f min  remaining ~%.1f min" % (i, len(inv), el / 60, el / i * (len(inv) - i) / 60), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(R / "REV_redetect_cells.csv", index=False)
    np.savez_compressed(R / "REV_psd.npz", freqs=FREQS, **P)
    print("saved", d.shape, len(P))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
