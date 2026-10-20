# -*- coding: utf-8 -*-
"""Staresina et al. 2023 (Nat Neurosci 26:1429-1437, doi:10.1038/s41593-023-01381-w)
   SO / spindle detection + coupling summary implemented directly from the Methods.

The original detector (thesis pipeline, tvns.jy_source) is left untouched. This is a 'second detector'
placed alongside it to check directional agreement (METHODS.md section 6, sensitivity axis).

Unavoidable departures from the paper — all recorded here:
  · ripples (80-120 Hz) not possible: these data are sampled at 125 Hz (Nyquist 62.5 Hz)
  · the paper used 1 kHz bipolar-referenced intracranial EEG; these are 125 Hz average-referenced scalp EEG
  · instead of the paper's z>6 artefact rule we use the BAD_AMP/BAD_Delta already in the fif,
    and add the paper's z rule on top (both applied)
  · filter: 3rd-order zero-phase Butterworth (sosfiltfilt) instead of the FieldTrip default

Points identical to the paper:
  · SO  0.3-1.25 Hz · between P->N zero crossings · 0.8-2.0 s · trough and trough-to-peak
    both exceed mean+1SD of the candidates
  · spindle  12-16 Hz · 200 ms RMS then 200 ms re-smoothing · threshold = mean+1SD of whole-NREM RMS
    · 0.4-3.0 s · exclude above upper bound mean+9SD · at least 6 cycles in the raw signal
  · stages = N2 + N3  (the original thesis pipeline uses N2 only)
  · coupling = spindle center within ±1 s of the SO down-state (trough)
"""
from __future__ import annotations

import numpy as np
from scipy import signal

SF_MIN = 100.0                      # minimum sampling rate that safely covers 12-16 Hz
SO_BAND = (0.3, 1.25)
SP_BAND = (12.0, 16.0)
SO_DUR = (0.8, 2.0)
SP_DUR = (0.4, 3.0)
RMS_WIN = 0.2                       # seconds
SP_MIN_CYCLES = 6
SP_UPPER_SD = 9.0
COUPLE_WIN = 1.0                    # seconds, ±1 s around the SO trough


def _bp(x, sf, band, order=3):
    sos = signal.butter(order, [band[0] / (sf / 2), band[1] / (sf / 2)],
                        btype="band", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, float))


def _zscore_artifact(x, sf, z_raw=6.0, z_any=4.0):
    """Paper Methods: z-score three signals — raw / 250 Hz high-pass / first derivative.
       Artefact if any one signal has |z|>6, or raw and one other simultaneously have |z|>4.
       A 250 Hz high-pass is impossible with 125 Hz data, so 0.8 x Nyquist is used instead."""
    x = np.asarray(x, float)
    hp_cut = min(0.8 * (sf / 2), 45.0)
    sos = signal.butter(3, hp_cut / (sf / 2), btype="high", output="sos")
    hp = signal.sosfiltfilt(sos, x)
    gr = np.gradient(x)

    def z(v):
        s = v.std()
        return np.abs(v - v.mean()) / s if s > 0 else np.zeros_like(v)

    zr, zh, zg = z(x), z(hp), z(gr)
    bad = (zr > z_raw) | (zh > z_raw) | (zg > z_raw)
    bad |= (zr > z_any) & ((zh > z_any) | (zg > z_any))
    # merge gaps within 3 s + 1 s padding on both sides
    pad = int(round(1.0 * sf))
    idx = np.flatnonzero(bad)
    if idx.size:
        out = np.zeros_like(bad)
        for i in idx:
            out[max(0, i - pad):min(len(bad), i + pad + 1)] = True
        bad = out
    return bad


def _so_candidates(x, sf, bad=None):
    xf = _bp(x, sf, SO_BAND)
    s = np.sign(xf)
    # positive -> negative zero crossings
    pn = np.flatnonzero((s[:-1] > 0) & (s[1:] <= 0)) + 1
    cands = []
    for a, b in zip(pn[:-1], pn[1:]):
        dur = (b - a) / sf
        if not (SO_DUR[0] <= dur <= SO_DUR[1]):
            continue
        seg = xf[a:b]
        if seg.size < 3:
            continue
        if bad is not None and bad[a:b].any():
            continue
        ti = a + int(np.argmin(seg))          # down-state (trough)
        after = xf[ti:b]
        pi = ti + int(np.argmax(after)) if after.size else ti
        cands.append(dict(onset=a, trough=ti, peak=pi, end=b,
                          trough_amp=abs(xf[ti]), p2p=xf[pi] - xf[ti]))
    return cands, xf


def detect_so(x, sf, bad=None, th=None):
    """If th is given, use that threshold (computed over the whole NREM). Otherwise compute it from this segment."""
    cands, xf = _so_candidates(x, sf, bad)
    if not cands:
        return [], xf
    if th is not None:
        th_t, th_p = th["so_th_t"], th["so_th_p"]
    else:
        ta = np.array([c["trough_amp"] for c in cands]); pp = np.array([c["p2p"] for c in cands])
        th_t = ta.mean() + ta.std(ddof=0); th_p = pp.mean() + pp.std(ddof=0)
    keep = [c for c in cands if c["trough_amp"] > th_t and c["p2p"] > th_p]
    return keep, xf


def detect_spindles(x, sf, bad=None, th=None):
    """Returns list of dicts: onset/center/end (sample)."""
    xf = _bp(x, sf, SP_BAND)
    w = max(1, int(round(RMS_WIN * sf)))
    k = np.ones(w) / w
    rms = np.sqrt(np.convolve(xf ** 2, k, mode="same"))
    rms = np.convolve(rms, k, mode="same")          # paper: re-smooth with the same window
    if th is not None:
        thr, hi = th["sp_th"], th["sp_hi"]
    else:
        ok = ~bad if bad is not None else np.ones(len(rms), bool)
        base = rms[ok]
        if base.size < w * 3:
            return []
        thr = base.mean() + base.std(ddof=0)
        hi = base.mean() + SP_UPPER_SD * base.std(ddof=0)
    above = rms > thr
    if bad is not None:
        above &= ~bad
    out = []
    d = np.diff(above.astype(int))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends, len(above)]
    for a, b in zip(starts, ends):
        dur = (b - a) / sf
        if not (SP_DUR[0] <= dur <= SP_DUR[1]):
            continue
        if rms[a:b].max() > hi:
            continue
        raw_seg = xf[a:b]
        # at least 6 cycles in the (band-passed) raw signal — counted via zero crossings
        zc = np.flatnonzero(np.diff(np.sign(raw_seg)) != 0).size
        if zc / 2.0 < SP_MIN_CYCLES:
            continue
        ci = a + int(np.argmin(raw_seg))            # center = maximal trough
        out.append(dict(onset=a, center=ci, end=b))
    return out


def couple(sos, spindles, so_phase, sf, win=COUPLE_WIN, jy_rule=False):
    """Staresina: an SO is coupled if a spindle center lies within ±win of the SO trough.
       With jy_rule=True, count with the original thesis pipeline's rule instead (within SO onset~end + |phase|<=90 deg, 1 per SO).
       Returns (n_coupled, phases_deg)"""
    if not sos or not spindles:
        return 0, []
    cen = np.array([s["center"] for s in spindles])
    w = int(round(win * sf))
    n_cp, phases = 0, []
    for so in sos:
        if jy_rule:
            m = (cen >= so["onset"]) & (cen < so["end"])
            if not m.any():
                continue
            ph = np.rad2deg(so_phase[cen[m]])
            good = np.abs(ph) <= 90
            if not good.any():
                continue
            n_cp += 1
            phases.append(float(ph[good][0]))
        else:
            m = np.abs(cen - so["trough"]) <= w
            if not m.any():
                continue
            n_cp += 1
            j = cen[m][int(np.argmin(np.abs(cen[m] - so["trough"])))]
            phases.append(float(np.rad2deg(so_phase[j])))
    return n_cp, phases


def circ_mrl_phase(phases_deg):
    p = np.asarray(phases_deg, float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan, np.nan, 0
    v = np.exp(1j * np.deg2rad(p)).mean()
    return float(np.abs(v)), float(np.degrees(np.angle(v))), int(p.size)


def session_thresholds(x_all, sf, bad_all=None):
    """As in the paper: thresholds are computed once over the 'whole NREM'.
       Recomputing per block makes between-block comparisons impossible (a problem that actually occurred in the pilot)."""
    ok = ~np.asarray(bad_all, bool) if bad_all is not None else np.ones(len(x_all), bool)
    # spindle RMS threshold
    xf = _bp(x_all, sf, SP_BAND)
    w = max(1, int(round(RMS_WIN * sf)))
    k = np.ones(w) / w
    rms = np.sqrt(np.convolve(xf ** 2, k, mode="same"))
    rms = np.convolve(rms, k, mode="same")
    b = rms[ok]
    sp_th = b.mean() + b.std(ddof=0) if b.size > w * 3 else np.inf
    sp_hi = b.mean() + SP_UPPER_SD * b.std(ddof=0) if b.size > w * 3 else np.inf
    # SO amplitude thresholds — from candidates over the whole NREM
    cands, _ = _so_candidates(x_all, sf, np.asarray(bad_all, bool) if bad_all is not None else None)
    if cands:
        ta = np.array([c["trough_amp"] for c in cands]); pp = np.array([c["p2p"] for c in cands])
        so_th_t = ta.mean() + ta.std(ddof=0); so_th_p = pp.mean() + pp.std(ddof=0)
    else:
        so_th_t = so_th_p = np.inf
    return dict(sp_th=sp_th, sp_hi=sp_hi, so_th_t=so_th_t, so_th_p=so_th_p)


def analyse_channel(x, sf, use_paper_artifact=False, bad_external=None, th=None):
    """One channel, one segment. Returns dict of metrics (both coupling rules)."""
    bad = np.zeros(len(x), bool)
    if bad_external is not None:
        bad |= np.asarray(bad_external, bool)
    if use_paper_artifact:
        bad |= _zscore_artifact(x, sf)
    sos, xf_so = detect_so(x, sf, bad, th=th)
    sps = detect_spindles(x, sf, bad, th=th)
    phase = np.angle(signal.hilbert(xf_so))      # 0 = up-state peak
    n_cp_s, ph_s = couple(sos, sps, phase, sf, jy_rule=False)
    n_cp_j, ph_j = couple(sos, sps, phase, sf, jy_rule=True)
    mrl_s, mph_s, _ = circ_mrl_phase(ph_s)
    mrl_j, mph_j, _ = circ_mrl_phase(ph_j)
    n_so, n_sp = len(sos), len(sps)
    return dict(
        n_SO=n_so, n_spindle=n_sp,
        n_coupled=n_cp_s, coupled_ratio=(n_cp_s / n_so if n_so else np.nan),
        MRL=mrl_s, mean_phase_deg=mph_s,
        n_coupled_jyrule=n_cp_j,
        coupled_ratio_jyrule=(n_cp_j / n_so if n_so else np.nan),
        MRL_jyrule=mrl_j, mean_phase_deg_jyrule=mph_j,
        bad_frac=float(bad.mean()),
    )
