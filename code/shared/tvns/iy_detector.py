"""tvns.iy_detector — default #2 "IRASA individual band -> YASA spindle detection" (track='iy').

Final definition, EXECUTION_PLAN_20260908.md §2.2:
  1. staging = YASA (jy_source.extract_N2_raw selects only N2 in the same way for every track --
     staging itself is already fixed at session loading (`cohorts.py` preferentially selects
     YASA-staged fif files such as IITY/VIPALLY). This module handles only "what comes next".
  2. individual sigma band = estimated with IRASA from the whole-session N2 (all blocks) -- reuses
     `tvns.coupling.individualized_bands` as is (already validated: used in EC preprocessing and TB Step5).
     Used only for frequency estimation, not for event detection (present/absent) (Hahn/Denis principle,
     identical to Phase 2 of the original thesis pipeline documentation).
  3. spindle detection = `yasa.spindles_detect(freq_sp=individual band)` -- the only point that differs
     from the jy track (only one element is changed on purpose, for "the cleanest sensitivity comparison").
  4. SO detection and coupling rules = identical to the jy track, `tvns.jy_source` (original thesis code) as is.
     The absolute µV thresholds of YASA sw_detect are unsuitable for these average-referenced 15-channel data
     (the original thesis pipeline also had to relax them to 10/5/20 µV to get a non-zero coupled_ratio),
     so the SO side is left unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .jy_source import detect_SOs, detect_coupling, compute_coupling_phase, summarize_coupling
from . import cohorts as C
from . import coupling as our_coupling  # individualized_bands() -- our existing, validated module


def session_n2_signal(session: C.Session, ch: str, blocks=None):
    """1D µV array of the N2 signal concatenated over all session blocks (for IRASA individual-band estimation).

    Block boundaries are ignored and the data simply concatenated -- IRASA only needs the overall spectral
    shape, so small discontinuities at the boundaries have negligible effect on band estimation (4 s windows).
    """
    blocks = blocks or C.CANON_BLOCKS
    chunks = []
    sf = None
    for block in blocks:
        n2raw, _bad = C.n2_concat_for_block(session, block)
        if n2raw is None or ch not in n2raw.ch_names:
            continue
        sf = n2raw.info["sfreq"]
        chunks.append(n2raw.get_data(picks=[ch])[0] * 1e6)  # V -> uV
    if not chunks:
        return None, None
    return np.concatenate(chunks), sf


def personal_band(session: C.Session, ch: str, blocks=None) -> dict:
    """Apply `tvns.coupling.individualized_bands` to one session x channel."""
    x, sf = session_n2_signal(session, ch, blocks)
    if x is None or x.size < 4 * (blocks and 1 or 1):
        return dict(sp_band=(12.0, 16.0), sp_peak=np.nan, so_band=(0.16, 1.25),
                    so_peak=np.nan, sp_from_fallback=True)
    return our_coupling.individualized_bands(x, sf)


def detect_spindles_yasa(x_uV: np.ndarray, sf: float, band: tuple[float, float],
                          bad_mask: np.ndarray | None = None) -> list[dict]:
    """YASA spindle detection -> converted to the list of dicts expected by jy_source.detect_coupling.

    Samples in bad_mask are set to 0 so the detector does not falsely detect noise as spindles
    (sample alignment is preserved -- segments are not cut out, because the output must remain usable
    for time-matching with SO cycles. The only difference from the original thesis code is silencing
    before detection rather than post-hoc exclusion).
    """
    import yasa

    x = x_uV.copy()
    if bad_mask is not None and bad_mask.size == x.size:
        x[bad_mask] = 0.0
    if x.size < int(4 * sf):
        return []
    try:
        res = yasa.spindles_detect(
            x, sf=sf, hypno=None, include=(2,), freq_sp=band,
            freq_broad=(1.0, 30.0), verbose=False,
        )
    except Exception:
        return []
    if res is None:
        return []
    df = res.summary()
    if df is None or df.empty:
        return []
    out = []
    for r in df.itertuples(index=False):
        out.append(dict(
            start_sample=int(round(r.Start * sf)),
            end_sample=int(round(r.End * sf)),
            peak_sample=int(round(r.Peak * sf)),
            peak_time=float(r.Peak),
            peak_rms=float(getattr(r, "RMS", getattr(r, "Amplitude", 0.0))),
        ))
    return out


def run_channel_block_iy(n2raw, bad_mask, ch_name: str, sp_band: tuple[float, float],
                          up_state_window_deg: float = 180.0) -> dict:
    """track='iy': SO = original thesis detector (jy_source), spindle = YASA (individual band), coupling = original thesis rules (jy_source)."""
    SOs = detect_SOs(n2raw, bad_mask, ch_name)
    x_uV = n2raw.get_data(picks=[ch_name])[0] * 1e6
    spindles = detect_spindles_yasa(x_uV, n2raw.info["sfreq"], sp_band, bad_mask)
    coupling_events = detect_coupling(
        SOs, spindles, raw=n2raw, ch_name=ch_name, up_state_window_deg=up_state_window_deg,
    )
    phases = compute_coupling_phase(n2raw, coupling_events, bad_mask, ch_name)
    summary = summarize_coupling(SOs, spindles, coupling_events, phases, print_summary=False)
    return dict(SOs=SOs, spindles=spindles, coupling_events=coupling_events,
                phases=phases, summary=summary, sp_band=sp_band)


def build_events_for_session_iy(session: C.Session, cohort: str,
                                 blocks: list[str] | None = None,
                                 channels: list[str] | None = None,
                                 up_state_window_deg: float = 180.0):
    """SO/spindle events + summary DataFrame for one session under default #2 (track='iy').

    Output schema matches jy_detector.build_events_for_session, but sp_peak_hz/so_peak_hz/
    sp_from_fallback columns are added to record the source of the individual band (1.2 aperiodic/individualization null test).
    """
    from .jy_detector import (
        so_events_df, spindle_events_df, phase_events_df, summary_row, _meta_row,
    )

    blocks = blocks or C.CANON_BLOCKS
    so_frames, sp_frames, ph_frames, summ_rows = [], [], [], []
    for block in blocks:
        n2raw, bad_mask = C.n2_concat_for_block(session, block)
        if n2raw is None:
            continue
        eeg_chs = set(n2raw.copy().pick("eeg").ch_names)
        chs = channels or [c for c in n2raw.ch_names if c in eeg_chs]
        duration_sec = float(n2raw.times[-1] - n2raw.times[0]) if len(n2raw.times) else 0.0
        for ch in chs:
            band = personal_band(session, ch, blocks)
            result = run_channel_block_iy(n2raw, bad_mask, ch, band["sp_band"],
                                           up_state_window_deg)
            meta = _meta_row(session, cohort, block, ch)
            so_frames.append(so_events_df(result, meta))
            sp_frames.append(spindle_events_df(result, meta, n2raw, ch, up_state_window_deg))
            ph_frames.append(phase_events_df(result, meta))
            row = summary_row(result, meta, duration_sec)
            row.update(sp_peak_hz=band["sp_peak"], so_peak_hz=band["so_peak"],
                       sp_from_fallback=band["sp_from_fallback"],
                       sp_band_lo=band["sp_band"][0], sp_band_hi=band["sp_band"][1])
            summ_rows.append(row)
    so_df = pd.concat(so_frames, ignore_index=True) if so_frames else pd.DataFrame()
    sp_df_all = pd.concat(sp_frames, ignore_index=True) if sp_frames else pd.DataFrame()
    ph_df = pd.concat(ph_frames, ignore_index=True) if ph_frames else pd.DataFrame()
    summ_df = pd.DataFrame(summ_rows)
    return so_df, sp_df_all, ph_df, summ_df
