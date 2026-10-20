"""tvns.coupling — faithful reconstruction of the thesis pipeline's SO/spindle detection and coupling rules.

Reproduces thesis section 2.7 directly from the signal. The observed coupled ratio then
matches the values in ``TB_SOSpindle_180_DB.parquet`` (validation: median n_SO ratio 1.00,
coupled_ratio correlation 0.98), and surrogates can be computed under **exactly the same
rules**, making the s05 time-shift control rigorous enough for the manuscript.

Thesis method (section 2.7)
---------------------------
SO detection:
  * 0.16–1.25 Hz Butterworth (3rd-order) zero-phase band-pass
  * intervals of 0.8–3.0 s between consecutive negative-going zero crossings as candidate cycles
  * peak-to-peak amplitude (negative trough -> following positive peak) above the 75th percentile per channel x segment
  * SO time = negative trough

Spindle detection:
  * 12–16 Hz (fast spindle), same filter design
  * RMS envelope (200 ms sliding window)
  * RMS above the 75th percentile per channel x segment for 0.5–3.0 s -> spindle
  * spindle time = amplitude peak

Coupling (at most one per SO):
  * the spindle peak lies within the SO trough-to-trough cycle, and
  * the SO phase (Hilbert) at that moment is within ±90° of the up-state peak (phase ∈ [−π/2, +π/2])
  * if at least one such spindle exists, the SO is coupled
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

SO_BAND = (0.16, 1.25)
SP_BAND = (12.0, 16.0)
UPSTATE_HALF = np.pi / 2  # ±90°


def so_filter_phase(x: np.ndarray, sf: float):
    """Return the SO band-passed signal and its instantaneous phase (radians, positive peak = 0)."""
    sos = signal.butter(3, [SO_BAND[0] / (sf / 2), SO_BAND[1] / (sf / 2)],
                        btype="band", output="sos")
    xf = signal.sosfiltfilt(sos, np.asarray(x, dtype=float))
    phase = np.angle(signal.hilbert(xf))  # 0 at the positive peak, ±π at the trough
    return xf, phase


def detect_so(x: np.ndarray, sf: float, pct: float = 75) -> pd.DataFrame:
    """Thesis-style SO detection. Returns DataFrame(trough, peak, end, p2p) [sample indices]."""
    xf, _ = so_filter_phase(x, sf)
    n = xf.size
    # negative-going zero crossing: xf>0 -> xf<=0
    dzc = np.flatnonzero((xf[:-1] > 0) & (xf[1:] <= 0)) + 1
    troughs, peaks, ends, p2p = [], [], [], []
    for a, b in zip(dzc[:-1], dzc[1:]):
        if not (0.8 <= (b - a) / sf <= 3.0):
            continue
        ti = a + int(np.argmin(xf[a:b]))
        if ti + 1 >= b:
            continue
        pi = ti + int(np.argmax(xf[ti:b]))
        troughs.append(ti); peaks.append(pi); ends.append(b)
        p2p.append(float(xf[pi] - xf[ti]))
    d = pd.DataFrame({"trough": troughs, "peak": peaks, "end": ends, "p2p": p2p})
    if d.empty:
        return d
    return d[d["p2p"] > np.percentile(d["p2p"], pct)].reset_index(drop=True)


def detect_spindles(x: np.ndarray, sf: float, pct: float = 75, band: tuple = SP_BAND):
    """Thesis-style spindle detection (RMS 200 ms + 75th pct). Returns (peak_idx, peak_rms).

    band : spindle band (Hz). Default 12–16; an individualised band can be passed.
    """
    sos = signal.butter(3, [band[0] / (sf / 2), band[1] / (sf / 2)],
                        btype="band", output="sos")
    xf = signal.sosfiltfilt(sos, np.asarray(x, dtype=float))
    w = max(1, int(round(0.2 * sf)))
    rms = np.sqrt(np.convolve(xf ** 2, np.ones(w) / w, mode="same"))
    above = rms > np.percentile(rms, pct)
    if not above.any():
        return np.array([], int), np.array([], float)
    d = np.diff(above.astype(np.int8))
    st = np.flatnonzero(d == 1) + 1
    en = np.flatnonzero(d == -1) + 1
    if above[0]:
        st = np.r_[0, st]
    if above[-1]:
        en = np.r_[en, rms.size]
    peaks, amps = [], []
    for a, b in zip(st, en):
        if not (0.5 <= (b - a) / sf <= 3.0):
            continue
        pi = a + int(np.argmax(rms[a:b]))
        peaks.append(pi); amps.append(float(rms[pi]))
    return np.array(peaks, int), np.array(amps, float)


def coupled_count(troughs: np.ndarray, ends: np.ndarray, phase: np.ndarray,
                  sp_idx: np.ndarray, n_samples: int, shift: int = 0):
    """Count coupled SOs (vectorised). At most one per SO.

    Parameters
    ----------
    shift : number of samples by which to circularly shift the spindle indices. 0 = observed,
        +0.7 s in samples = time-shift control, random = circular null distribution.
    """
    n_so = int(troughs.size)
    if n_so == 0 or sp_idx.size == 0:
        return n_so, 0
    sp = (sp_idx + shift) % n_samples
    # phase-eligible spindles only
    elig = sp[(phase[sp] >= -UPSTATE_HALF) & (phase[sp] <= UPSTATE_HALF)]
    if elig.size == 0:
        return n_so, 0
    # SO cycle containing each spindle: trough<=t<end
    order = np.argsort(troughs)
    tr = troughs[order]; en = ends[order]
    j = np.searchsorted(tr, elig, side="right") - 1  # candidate cycle
    ok = (j >= 0) & (elig < en[np.clip(j, 0, n_so - 1)])
    coupled_cycles = np.unique(j[ok])
    return n_so, int(coupled_cycles.size)


def individualized_bands(x: np.ndarray, sf: float,
                         so_search=(0.5, 2.0), sp_search=(11.0, 16.0),
                         sp_fallback=14.0, sp_halfwidth=2.0, so_halfwidth=0.5):
    """Separate the aperiodic (1/f) background with IRASA, then derive individual SO/spindle peak frequencies and bands.

    Hahn (2020) approach + Denis (2025) fallback rule: use the individual peak when one is found, otherwise the
    default (spindle 14 Hz). This avoids the under-detection of spindles by a fixed band in samples with large
    inter-individual variability such as ages 15–83 (Hahn).

    Returns dict(so_peak, sp_peak, so_band, sp_band, sp_from_fallback).
    """
    import yasa

    x = np.asarray(x, float)
    out = {"so_peak": np.nan, "sp_peak": np.nan,
           "so_band": (0.5, 1.25), "sp_band": (12.0, 16.0),
           "sp_from_fallback": True, "so_from_fallback": True}
    if x.size < int(4 * sf):
        return out
    try:
        # IRASA: fractal/oscillatory decomposition over a broad band
        freqs, _psd_ap, psd_osc, _fit = yasa.irasa(
            x[None, :] * 1e6, sf=sf, band=(0.3, 25.0),
            win_sec=min(4.0, x.size / sf / 2), return_fit=True)
        osc = np.asarray(psd_osc).ravel()
        freqs = np.asarray(freqs).ravel()

        def peak_in(lo, hi):
            m = (freqs >= lo) & (freqs <= hi)
            if not m.any():
                return np.nan
            seg = osc[m]
            # accept a peak only in a genuine oscillatory component (residual > 0)
            if np.nanmax(seg) <= 0:
                return np.nan
            return float(freqs[m][int(np.nanargmax(seg))])

        sp_pk = peak_in(*sp_search)
        so_pk = peak_in(*so_search)
        if np.isfinite(sp_pk):
            out["sp_peak"] = sp_pk
            out["sp_band"] = (max(0.5, sp_pk - sp_halfwidth), sp_pk + sp_halfwidth)
            out["sp_from_fallback"] = False
        else:
            out["sp_peak"] = sp_fallback
            out["sp_band"] = (sp_fallback - sp_halfwidth, sp_fallback + sp_halfwidth)
        if np.isfinite(so_pk):
            out["so_peak"] = so_pk
            out["so_band"] = (max(0.16, so_pk - so_halfwidth), so_pk + so_halfwidth)
            out["so_from_fallback"] = False
    except Exception:
        pass
    return out


def analyze_cell(x: np.ndarray, sf: float, so_pct: float = 75, sp_pct: float = 75,
                 sp_band: tuple | None = None):
    """Prepare SO/spindle/phase for one signal segment and return them ready for surrogate computation.

    Returns dict(troughs, ends, phase, sp_idx, sp_amp, n_samples, n_so, n_spindle).

    sp_band : individualised spindle band (Hz). If None, the fixed 12–16 Hz band.
    """
    so = detect_so(x, sf, so_pct)
    _, phase = so_filter_phase(x, sf)
    band = tuple(sp_band) if sp_band is not None else SP_BAND
    sp_idx, sp_amp = detect_spindles(x, sf, sp_pct, band=band)
    return {
        "troughs": so["trough"].to_numpy() if not so.empty else np.array([], int),
        "ends": so["end"].to_numpy() if not so.empty else np.array([], int),
        "phase": phase,
        "sp_idx": sp_idx,
        "sp_amp": sp_amp,
        "n_samples": int(x.size),
        "n_so": 0 if so.empty else len(so),
        "n_spindle": int(sp_idx.size),
    }
