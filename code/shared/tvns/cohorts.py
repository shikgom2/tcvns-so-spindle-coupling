"""tvns.cohorts — cohort registry: normalises paths, session lists, block names and design type in one place.

Previously, column names (`Dir`/`ID`, `stage`/`block`) and block names
(`stim1` vs `'stimulation 1'`) differed from cohort to cohort. This module:
  1. Unifies all 4 cohorts to the **same 5 block names** (original thesis-pipeline
     convention: baseline, stimulation 1, recovery 1, stimulation 2, recovery 2).
  2. Gives EC the same interface as TB/IIT/VIP (`mne.Raw` + 'sleep' channel + block
     annotations + BAD_AMP/BAD_Delta annotations), so that `tvns.jy_source.extract_N2_raw` +
     `bad_annot_mask` behave identically in all 4 cohorts.
  3. For VIP, uses VIPALL (91 participants / 304 sessions, 2026-09 extended set) as primary,
     and distinguishes VIP1 (29 participants, manuscript period) via the `subset` column,
     remapping the physical path (Data/VIP vs VIP_all/VIPALL).

Hard rule: `rawfiles_fromJY/` is read-only. This module never writes anything inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .jy_source import EEG_CFG, extract_N2_raw, bad_annot_mask

# The raw-data layout can change when data are migrated, so roots are **located by signature files** (tvns.paths).
# 2026-09-10: E:/DATA was copied into rawfiles_fromJY/DATA/Data/ and the folder names changed as well.
from . import paths as _paths

_FALLBACK = Path("D:/HJ/vns/rawfiles_fromJY/DATA")
DATA = _paths.data_root() or (_FALLBACK / "Data")      # Recording_metadata.xlsx + TB/
ROOT = DATA.parent                                      # parent, because fif_path starts with 'Data/...'
IIT_ROOT = _paths.iit_root() or (_FALLBACK / "IIT_data")
VIP_ALL_ROOT = _paths.vip_all_root() or (_FALLBACK / "VIP_all")
RAW_EC_ROOT = _paths.ec_raw_root() or (_FALLBACK / "Raw_data" / "EC")

# Original thesis-pipeline 5 block names (same as EEG_CFG.trigger_labels) — all cohorts are unified to these names.
CANON_BLOCKS = ["baseline", "stimulation 1", "recovery 1", "stimulation 2", "recovery 2"]
STAGE_MAP = EEG_CFG.sleep_staging.stage_mapping  # {'UNK':-1,'W':0,'R':1,'N1':2,'N2':3,'N3':4}

DesignType = Literal["within_interaction", "between", "longitudinal"]

COHORTS = ["TB", "EC", "IIT", "VIP"]

DESIGN_OF: dict[str, DesignType] = {
    "TB": "within_interaction",
    "EC": "between",
    "IIT": "between",
    "VIP": "longitudinal",
    # TB (tVNS -> Treatment, Order-4 -> SHAM) + IIT pooled as one between-group
    # cohort. Not in COHORTS, so the default 4-cohort loops are unchanged;
    # request it explicitly with --cohorts COMBINED. (so_spindle_2, 2026-09-12)
    "COMBINED": "between",
}


@dataclass
class Session:
    cohort: str
    subject: str            # ID
    session_id: str         # Order (TB/IIT/VIP) or Dir+timepoint (EC), unique within cohort
    order: float | None
    group: str | None       # TB: 'tVNS'/'unstimulated_control' ; EC/IIT: 'Treatment'/'SHAM' ; VIP: 'Treatment'
    track_source: str = "fif"   # 'fif' or 'ec_csv'
    path: Path | None = None    # fif path or EC csv path
    n2_min_hint: float | None = None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TB
# ---------------------------------------------------------------------------
def _list_tb_sessions() -> list[Session]:
    meta = pd.read_excel(DATA / "Recording_metadata.xlsx")
    meta = meta[(meta.Exp_type == "TB") & (meta.epo_use_flag == 1)]
    out = []
    for _, r in meta.iterrows():
        out.append(Session(
            cohort="TB", subject=r.ID, session_id=str(r.Dir), order=None,
            group=None, track_source="fif",
            path=ROOT / str(r.fif_path).replace("_raw_clean_filt.fif", "_raw_fin.fif"),  # fif_path already carries the 'Data/' prefix
            meta=dict(r),
        ))
    # Order (1-4) + tVNS/unstimulated_control labels are attached from TB_SOSpindle_180_DB (matched on f_path).
    db = pd.read_parquet(DATA / "TB" / "TB_SOSpindle_180_DB.parquet")[
        ["ID", "Order", "f_path"]].drop_duplicates()
    db["fname"] = db.f_path.astype(str).str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
    fname_to_order = {(row.ID, row.fname): int(row.Order) for row in db.itertuples()}
    for s in out:
        fname = s.path.name
        order = fname_to_order.get((s.subject, fname))
        s.order = order
        s.group = None if order is None else ("tVNS" if order <= 3 else "unstimulated_control")
    return [s for s in out if s.order is not None and s.path.exists()]


# ---------------------------------------------------------------------------
# IIT
# ---------------------------------------------------------------------------
def _list_iit_sessions(staging: str = "yasa") -> list[Session]:
    """staging='yasa' -> IITY (re-staged, same staging as the manuscript convention) preferred; 'neuronet' -> original IIT set."""
    meta = pd.read_excel(IIT_ROOT / "IIT_Recording_metadata.xlsx")
    meta = meta[meta.epo_use_flag == 1]
    subdir = "IITY" if staging == "yasa" else "IIT"
    out = []
    for _, r in meta.iterrows():
        relpath = Path(str(r.epo_path))
        fname = relpath.name.replace("_epo.fif", "_raw_fin.fif")
        subj_dir = relpath.parent.name
        fif = IIT_ROOT / subdir / subj_dir / fname
        out.append(Session(
            cohort="IIT", subject=r.ID, session_id=f"{r.ID}_O{int(r.Order)}",
            order=float(r.Order), group=r.Group, track_source="fif", path=fif, meta=dict(r),
        ))
    return [s for s in out if s.path.exists()]


# ---------------------------------------------------------------------------
# VIP (VIPALL primary; VIP1 physical path remapped via the subset column)
# ---------------------------------------------------------------------------
def _list_vip_sessions(staging: str = "yasa", primary: str = "VIPALL") -> list[Session]:
    """staging='yasa' -> VIPALLY (re-staged, YASA) preferred. primary='VIPALL' (91 participants /
    304 sessions, default) or 'VIP1' (29 participants, sensitivity -- narrower manuscript-period set)."""
    meta = pd.read_excel(VIP_ALL_ROOT / "VIPALL_Recording_metadata.xlsx")
    meta = meta[meta.epo_use_flag == 1]
    if primary == "VIP1":
        meta = meta[meta.subset == "VIP1"]
    out = []
    for _, r in meta.iterrows():
        fname = Path(str(r.epo_path)).name.replace("_epo.fif", "_raw_fin.fif")
        if staging == "yasa":
            base = VIP_ALL_ROOT / "VIPALLY" / r.ID / fname
        else:
            # Original staging: looking in only one location by subset finds just 65 sessions.
            # Scan both Data/VIP (211) and VIP_all/VIPALL (79) to obtain 265 sessions
            # (both use the original staging, so mixing them is methodologically consistent).
            cands = [DATA / "VIP" / r.ID / fname, VIP_ALL_ROOT / "VIPALL" / r.ID / fname]
            base = next((c for c in cands if c.exists()), cands[0])
        out.append(Session(
            cohort="VIP", subject=r.ID, session_id=f"{r.ID}_O{int(r.Order)}",
            order=float(r.Order), group="Treatment", track_source="fif", path=base,
            meta=dict(r),
        ))
    return [s for s in out if s.path.exists()]


# ---------------------------------------------------------------------------
# EC -- raw csv. The pseudo-raw builder is build_ec_pseudo_raw() (below).
# ---------------------------------------------------------------------------
def _list_ec_sessions() -> list[Session]:
    from . import ec
    # RAW_EC_ROOT (D:) has no sham (B) CSVs -- delegate to find_ec_root() to pick the root (E:)
    # that holds both active and sham. Both are read-only raw sources.
    ses = ec.list_ec_sessions()
    ses = ses[ses.timepoint.isin([f"T{i}" for i in range(0, 8)])]
    out = []
    for r in ses.itertuples(index=False):
        tp_num = str(r.timepoint)[1:]
        out.append(Session(
            cohort="EC", subject=r.Dir, session_id=f"{r.Dir}_{r.timepoint}",
            order=float(tp_num) if tp_num.isdigit() else None,
            group=r.Group, track_source="ec_csv", path=r.path,
            meta=dict(Dir=r.Dir, arm=r.arm, timepoint=r.timepoint, flag=r.flag),
        ))
    return out


def _list_combined_sessions(**kw) -> list[Session]:
    """TB + IIT as one between-group cohort. Group labels are harmonised to
    Treatment/SHAM and subject/session_id get a cohort prefix so the two
    cohorts cannot collide. Only the scripts that re-read raw signal
    (1.8/1.9/1.10) call this; parquet-only scripts read the pooled event DBs
    that so_spindle_2/code/17_build_p1_combined_events.py writes."""
    label = {"tVNS": "Treatment", "unstimulated_control": "SHAM"}
    out = []
    for s in _list_tb_sessions():
        out.append(Session(cohort="COMBINED", subject=f"TB_{s.subject}",
                           session_id=f"TB_{s.session_id}", order=s.order,
                           group=label.get(s.group, s.group),
                           track_source=s.track_source, path=s.path,
                           n2_min_hint=s.n2_min_hint,
                           meta=dict(s.meta, src_cohort="TB")))
    for s in _list_iit_sessions(**kw):
        out.append(Session(cohort="COMBINED", subject=f"IIT_{s.subject}",
                           session_id=f"IIT_{s.session_id}", order=s.order,
                           group=s.group, track_source=s.track_source,
                           path=s.path, n2_min_hint=s.n2_min_hint,
                           meta=dict(s.meta, src_cohort="IIT")))
    return out


_LISTERS = {"TB": _list_tb_sessions, "EC": _list_ec_sessions,
            "IIT": _list_iit_sessions, "VIP": _list_vip_sessions,
            "COMBINED": _list_combined_sessions}


def list_sessions(cohort: str, **kw) -> list[Session]:
    if cohort not in _LISTERS:
        raise ValueError(f"알 수 없는 코호트: {cohort}")
    return _LISTERS[cohort](**kw)


# ---------------------------------------------------------------------------
# EC pseudo-raw: convert to the same interface as TB/IIT/VIP
# ---------------------------------------------------------------------------
def _build_ec_pseudo_raw_uncached(session: Session, include_ecg: bool = False):
    """EC csv -> mne.Raw with a 'sleep' channel + 5 block annotations (thesis-pipeline standard
    names) + BAD_AMP/BAD_Delta annotations. Afterwards `extract_N2_raw`/`bad_annot_mask` behave
    exactly as for TB/IIT/VIP.

    Block mapping: `tvns.ec.BLOCK_LABELS` (baseline/stim1/recovery1/stim2/recovery2, based on the
    T1-T6 five-segment markers) renamed to CANON_BLOCKS. T0/T7 (1 marker, baseline only) form a
    single baseline block.
    """
    import mne
    from . import ec, staging as staging_mod
    from .badannot import annotate_bad_like_jy

    mne.set_log_level("ERROR")
    raw, markers = ec.load_raw(session.path, include_ecg=include_ecg)
    sf = raw.info["sfreq"]
    n = raw.n_times
    n_blocks = int(markers.size)

    if n_blocks <= 1:
        block_names = ["baseline"]
        bounds = [(0, n)]
    else:
        block_names = CANON_BLOCKS[:n_blocks] if n_blocks <= 5 else \
            CANON_BLOCKS + [f"block{i}" for i in range(6, n_blocks + 1)]
        edges = list(markers) + [n]
        bounds = list(zip(edges[:-1], edges[1:]))

    onsets = [s / sf for s, _ in bounds]
    durations = [(e - s) / sf for s, e in bounds]
    ann = mne.Annotations(onset=onsets, duration=durations, description=block_names)
    raw.set_annotations(ann)

    # YASA staging (C4) -> 30 s hypnogram -> upsampled to a full-sample 'sleep' channel (thesis-pipeline EEG_CFG code convention).
    hyp, _conf = staging_mod.stage(raw, eeg_name="C4")
    codes = np.array([STAGE_MAP.get(h, -1) for h in hyp], dtype=float)
    sleep_full = np.repeat(codes, int(round(30 * sf)))
    if sleep_full.size < n:
        pad_val = codes[-1] if codes.size else -1
        sleep_full = np.pad(sleep_full, (0, n - sleep_full.size), constant_values=pad_val)
    sleep_full = sleep_full[:n]
    sleep_info = mne.create_info(["sleep"], sf, ["misc"])
    sleep_raw = mne.io.RawArray(sleep_full[None, :], sleep_info, verbose="ERROR")
    raw.add_channels([sleep_raw], force_update_info=True)

    raw = annotate_bad_like_jy(raw, picks="eeg")
    return raw


# For EC, n2_concat_for_block is called for every channel x block, rebuilding the same session's
# pseudo-raw up to 75 times (CSV re-parse + YASA staging). The result is identical, so only the most recent session is cached.
_EC_RAW_CACHE: dict = {}


def build_ec_pseudo_raw(session: Session, include_ecg: bool = False):
    """One-entry memoising wrapper around `_build_ec_pseudo_raw_uncached` (identical result; only removes recomputation)."""
    key = (str(session.path), bool(include_ecg))
    hit = _EC_RAW_CACHE.get(key)
    if hit is None:
        _EC_RAW_CACHE.clear()
        hit = _build_ec_pseudo_raw_uncached(session, include_ecg=include_ecg)
        _EC_RAW_CACHE[key] = hit
    return hit.copy()


def n2_concat_for_block(session: Session, block: str, include_ecg: bool = False):
    """Raw of only the N2 epochs within the block, concatenated, + bad_mask. (raw, bad_mask) or (None, None).
    include_ecg=True matters only for EC (TB/IIT/VIP fif files already contain an 'ecg' channel)."""
    import mne
    mne.set_log_level("ERROR")
    if session.track_source == "ec_csv":
        raw = build_ec_pseudo_raw(session, include_ecg=include_ecg)
    else:
        raw = mne.io.read_raw_fif(str(session.path), preload=True, verbose="ERROR")
    n2raw = extract_N2_raw(raw, stage=block, sleep_label="N2")
    if n2raw is None:
        return None, None
    bad_mask = bad_annot_mask(n2raw)
    return n2raw, bad_mask


def condition_label(session: Session, cohort: str) -> str:
    """Treatment label by design. TB: tVNS/unstimulated_control (session.group). EC/IIT: Group
    (Treatment/SHAM). VIP: 'Treatment' (single arm, no control)."""
    return session.group or "unknown"
