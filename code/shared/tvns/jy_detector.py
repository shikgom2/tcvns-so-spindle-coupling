"""tvns.jy_detector — runs the original detector (thesis pipeline, track='jy') per session x block x
channel and builds event-level DataFrames (SO/spindle) and session summaries (same schema as the original DB).

This module does not reimplement any logic — it calls the original functions
(`detect_SOs`, `detect_spindles`, `detect_coupling`, `compute_coupling_phase`,
`summarize_coupling`) unchanged via `tvns.jy_source`. All it does is flatten their output into
tidy DataFrames and iterate over the sessions/blocks/channels, which differ by cohort.

Two "coupled" definitions (matching the SO-side / spindle-side reallocation metrics in Section 3.3 of the handover document):
  - SO-side (coupled_ratio): the original `detect_coupling` as is — at most one spindle per SO
    (the one with the largest RMS within the same cycle) is kept; the SO is coupled if it lies within the up-state phase.
    Every downstream analysis validated against the original — prevalence/phase/MRL (1.3),
    Analysis A/B (1.4/1.5), surrogate (2.1), etc. — uses this definition.
  - spindle-side (coupled_sp): for each spindle individually, tests "does this spindle's peak fall
    inside some SO cycle, with the phase at that instant within up-state ±(window/2)" (no per-SO exclusivity).
    Used for the spindle-side metrics of the 1.7 reallocation (coupled/uncoupled spindle count).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .jy_source import (
    detect_SOs, detect_spindles, detect_coupling, compute_coupling_phase,
    summarize_coupling, bandpass_filter,
)
from . import cohorts as C


def run_channel_block(n2raw, bad_mask, ch_name: str, up_state_window_deg: float = 180.0) -> dict:
    """Original pipeline unchanged: SO -> spindle -> coupling -> phase -> summary."""
    SOs = detect_SOs(n2raw, bad_mask, ch_name)
    spindles = detect_spindles(n2raw, bad_mask, ch_name)
    coupling_events = detect_coupling(
        SOs, spindles, raw=n2raw, ch_name=ch_name, up_state_window_deg=up_state_window_deg,
    )
    phases = compute_coupling_phase(n2raw, coupling_events, bad_mask, ch_name)
    summary = summarize_coupling(SOs, spindles, coupling_events, phases, print_summary=False)
    return dict(SOs=SOs, spindles=spindles, coupling_events=coupling_events,
                phases=phases, summary=summary)


def _spindle_side_coupling(n2raw, SOs: list[dict], spindles: list[dict], ch_name: str,
                            up_state_window_deg: float = 180.0):
    """Spindle-side coupled flag (no per-SO exclusivity). Returns bool array (len spindles),
    and the index of the SO each spindle belongs to (-1 if none)."""
    if not SOs or not spindles:
        return np.zeros(len(spindles), dtype=bool), np.full(len(spindles), -1)
    filtered = bandpass_filter(n2raw, ch_name, 0.16, 1.25)
    from scipy.signal import hilbert
    so_phase = np.angle(hilbert(filtered))
    half = np.deg2rad(up_state_window_deg / 2.0)

    starts = np.array([so["start_sample"] for so in SOs])
    ends = np.array([so["end_sample"] for so in SOs])
    order = np.argsort(starts)
    starts_s, ends_s = starts[order], ends[order]

    peaks = np.array([sp["peak_sample"] for sp in spindles])
    coupled = np.zeros(len(spindles), dtype=bool)
    so_idx = np.full(len(spindles), -1)
    j = np.searchsorted(starts_s, peaks, side="right") - 1
    valid = j >= 0
    within = np.zeros(len(spindles), dtype=bool)
    within[valid] = peaks[valid] <= ends_s[np.clip(j[valid], 0, len(ends_s) - 1)]
    ph_ok = np.abs(so_phase[peaks]) <= half
    ok = valid & within & ph_ok
    coupled[ok] = True
    so_idx[ok] = order[j[ok]]
    return coupled, so_idx


def _meta_row(session: C.Session, cohort: str, block: str, ch: str) -> dict:
    return dict(
        cohort=cohort, subject=session.subject, session_id=session.session_id,
        order=session.order, group=session.group,
        condition=C.condition_label(session, cohort), block=block, ch=ch,
    )


def so_events_df(result: dict, meta: dict) -> pd.DataFrame:
    SOs = result["SOs"]
    if not SOs:
        return pd.DataFrame()
    coupled_so_ids = {id(ev["so"]) for ev in result["coupling_events"]}
    rows = []
    for so in SOs:
        rows.append(dict(
            **meta,
            trough_time=so["trough_time"], peak_time=so["peak_time"],
            start_sample=so["start_sample"], end_sample=so["end_sample"],
            p2p_uV=float(so["amplitude"]) * 1e6,
            coupled=id(so) in coupled_so_ids,
        ))
    return pd.DataFrame(rows)


def spindle_events_df(result: dict, meta: dict, n2raw, ch_name: str,
                       up_state_window_deg: float = 180.0) -> pd.DataFrame:
    spindles = result["spindles"]
    if not spindles:
        return pd.DataFrame()
    coupled_sp, so_idx = _spindle_side_coupling(
        n2raw, result["SOs"], spindles, ch_name, up_state_window_deg)
    sfreq = n2raw.info["sfreq"]
    rows = []
    for i, sp in enumerate(spindles):
        rows.append(dict(
            **meta,
            peak_time=sp["peak_time"],
            start_time=sp["start_sample"] / sfreq, end_time=sp["end_sample"] / sfreq,
            dur_sec=(sp["end_sample"] - sp["start_sample"]) / sfreq,
            peak_rms=sp["peak_rms"],
            coupled_sp=bool(coupled_sp[i]), so_idx=int(so_idx[i]),
        ))
    return pd.DataFrame(rows)


def phase_events_df(result: dict, meta: dict) -> pd.DataFrame:
    """Phase (radians) of each coupled SO -- event-level table for treating the 1.3 phase as a
    co-primary outcome. `result['phases']` and `result['coupling_events']` share the same order (the
    original `compute_coupling_phase` builds it by iterating over `coupling_events` as is)."""
    events = result["coupling_events"]
    phases = result["phases"]
    if not events or len(phases) == 0:
        return pd.DataFrame()
    n = min(len(events), len(phases))
    rows = []
    for i in range(n):
        ev = events[i]
        rows.append(dict(
            **meta, phase=float(phases[i]),
            so_trough_time=ev["trough_time"], spindle_peak_time=ev["spindle_peak_time"],
            time_diff=ev["time_diff"],
        ))
    return pd.DataFrame(rows)


def summary_row(result: dict, meta: dict, duration_sec: float) -> dict:
    s = result["summary"]
    return dict(**meta, n_SO=s["n_SO"], n_spindle=s["n_spindle"], n_coupled=s["n_coupled"],
                coupled_ratio=s["coupled_ratio"], MRL=s["MRL"],
                mean_phase_deg=s["mean_phase_deg"], duration_sec=duration_sec)


def build_events_for_session(session: C.Session, cohort: str,
                              blocks: Optional[list[str]] = None,
                              channels: Optional[list[str]] = None,
                              up_state_window_deg: float = 180.0):
    """Return the SO-event / spindle-event / summary DataFrames for one session (all blocks x channels).

    Not run here -- the caller (`s21_build_events.py`) iterates over the whole cohort using this function.
    """
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
            result = run_channel_block(n2raw, bad_mask, ch, up_state_window_deg)
            meta = _meta_row(session, cohort, block, ch)
            so_frames.append(so_events_df(result, meta))
            sp_frames.append(spindle_events_df(result, meta, n2raw, ch, up_state_window_deg))
            ph_frames.append(phase_events_df(result, meta))
            summ_rows.append(summary_row(result, meta, duration_sec))
    so_df = pd.concat(so_frames, ignore_index=True) if so_frames else pd.DataFrame()
    sp_df = pd.concat(sp_frames, ignore_index=True) if sp_frames else pd.DataFrame()
    ph_df = pd.concat(ph_frames, ignore_index=True) if ph_frames else pd.DataFrame()
    summ_df = pd.DataFrame(summ_rows)
    return so_df, sp_df, ph_df, summ_df
