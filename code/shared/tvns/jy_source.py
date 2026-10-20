"""tvns.jy_source — **read-only import** of the original detector and preprocessing code (thesis pipeline).

Background (2026-09-08 re-assessment of the TB Step gate)
---------------------------------------------------------
`tvns/coupling.py` (our reimplementation) differed in logic from the original `rawfiles_fromJY/DATA/vip2_delivery/utils/eeg_utils/
SO_Spindle_coupling.py` (hereafter "O"):
  1. BAD segment handling — O excludes SO/spindle candidates that overlap BAD, and computes the spindle RMS percentile
     after setting BAD segments to NaN. Our reimplementation ignored BAD.
  2. N2 seams — O crops each N2 run and joins them with `mne.concatenate_raws` (which automatically inserts BAD_boundary
     annotations at the seams) → events spanning a seam are excluded automatically. Our reimplementation had no seam awareness.
  3. SO amplitude definition — O uses max-min over the whole ZC-ZC interval (the peak may precede the trough), threshold `>=`.
     Our reimplementation used only the max after the trough, threshold `>`.
  4. Coupling time window — O uses SO `start_sample` (falling ZC) to `end_sample`; ours used `trough` to `end`.
These differences were the actual cause of the misdiagnosis of "implementation error" in Step 2.5 (coupled_ratio r=0.927).

Solution: instead of fixing the reimplementation, **import and use the original functions as they are**. There is no risk of
transcription errors, and the logic is guaranteed to be 100% identical to the original. `rawfiles_fromJY/` is read-only under
the CLAUDE.md rules — this module only imports from it and never writes anything to that folder.

Usage
-----
    from tvns.jy_source import (
        detect_SOs, detect_spindles, detect_coupling, compute_coupling_phase,
        summarize_coupling, bandpass_filter, bad_annot_mask, group_consecutive,
        windowing, compute_multitaper_bp, EEG_CFG,
    )
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The original module imports matplotlib at top level (`import matplotlib.pyplot as plt` at the top of
# SO_Spindle_coupling.py). Fix the backend so that headless runs are safe.
os.environ.setdefault("MPLBACKEND", "Agg")

# Location of the original (thesis pipeline) repository — read-only source. Fail immediately if this path is missing
# (no silent workaround: if our reimplementation silently stood in for a missing original, the Step gate would again be meaningless).
# The path may change with data migration, so it is located via a signature file (paths.jy_source_root).
from . import paths as _paths
_JY_ROOT = _paths.jy_source_root() or Path("D:/HJ/vns/rawfiles_fromJY/DATA/vip2_delivery")
if not _JY_ROOT.exists():
    raise FileNotFoundError(
        f"원본(학위논문 파이프라인) 코드 폴더가 없습니다: {_JY_ROOT}\n"
        "이 모듈은 재구현으로 조용히 대체하지 않습니다 — TB Step 게이트가 다시 무의미해집니다."
    )
if str(_JY_ROOT) not in sys.path:
    sys.path.insert(0, str(_JY_ROOT))

# utils/ has no __init__.py (namespace package), so importing utils.eeg_utils submodules individually
# does not load the heavy sibling modules (eeg_analysis.py, which depends on mne_connectivity).
# eeg_utils/__init__.py itself contains only a docstring, so there are no side effects (checked 2026-09-08).
from utils.eeg_utils.SO_Spindle_coupling import (  # noqa: E402
    bandpass_filter,
    detect_SOs,
    detect_spindles,
    detect_coupling,
    extract_coupling_epochs,
    compute_coupling_phase,
    summarize_coupling,
    Compute_SO_spindle_coupling,
)
from utils.eeg_utils.eeg_preprocess import (  # noqa: E402
    bad_annot_mask,
    BadSegmentAnnotator,
)
from utils.eeg_utils.utils import (  # noqa: E402
    windowing,
    group_consecutive,
)
from utils.eeg_utils.config import EEG_CFG  # noqa: E402


def compute_multitaper_bp(raw, freqs=None, decim: int = 100, freq_band=None, picks="eeg"):
    """Call `BadSegmentAnnotator._compute_multitaper_bp` directly, without an instance.

    (The original takes `raw` as its first argument without `self`, like a staticmethod — checked 2026-09-08.)
    """
    import numpy as np

    if freqs is None:
        freqs = np.arange(0.5, 4.0, 0.5)
    return BadSegmentAnnotator._compute_multitaper_bp(
        raw, freqs=freqs, decim=decim, freq_band=freq_band, picks=picks
    )


def extract_N2_raw(raw, stage: str, sleep_label: str = "N2", sleep_mapping=None):
    """`run_measures.py::extract_N2_raw` as is (original code, used in practice in the IIT/VIP2 pipelines).

    Within one block (stage) annotation, extracts only the samples labelled `sleep_label` (default N2)
    and joins them with `mne.concatenate_raws`. Because concatenate_raws automatically inserts
    "BAD boundary" annotations at the seams, `bad_annot_mask` then recognises the seams automatically
    (rule: any description containing "BAD" is masked — this matches the BAD-exclusion logic of
    SO_Spindle_coupling.py exactly).

    Returns
    -------
    mne.io.Raw or None (no N2 in that block).
    """
    import numpy as np
    import mne

    if sleep_mapping is None:
        sleep_mapping = EEG_CFG.sleep_staging.stage_mapping
    target_val = sleep_mapping[sleep_label]
    stage_annot = next((a for a in raw.annotations if a["description"] == stage), None)
    if stage_annot is None:
        return None
    onset, duration = stage_annot["onset"], stage_annot["duration"]
    seg = raw.copy().crop(tmin=onset, tmax=min(onset + duration, raw.times[-1]))
    sleep_data = seg.get_data(picks="sleep")[0]
    slp_idx = np.where(sleep_data == target_val)[0]
    if len(slp_idx) == 0:
        return None
    groups = group_consecutive(slp_idx)
    parts = [
        seg.copy().crop(tmin=seg.times[g[0]], tmax=seg.times[g[-1]])
        for g in groups if len(g)
    ]
    return mne.concatenate_raws(parts) if parts else None


__all__ = [
    "bandpass_filter", "detect_SOs", "detect_spindles", "detect_coupling",
    "extract_coupling_epochs", "compute_coupling_phase", "summarize_coupling",
    "Compute_SO_spindle_coupling", "bad_annot_mask", "BadSegmentAnnotator",
    "windowing", "group_consecutive", "EEG_CFG", "compute_multitaper_bp",
    "extract_N2_raw",
]
