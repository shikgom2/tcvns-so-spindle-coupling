# -*- coding: utf-8 -*-
"""Spindle-detector false-positive rate: number of events the same detector finds in spectrum-preserving phase-randomised surrogates.
Sample: the same 40 random recordings as rev_spindle_specificity, stimulation-window N2, 13 channels. Output REV_spindle_fpr.csv"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(LP.PROJECT_ROOT / "code/paper"))
import rev_redetect as RD                                         # noqa: E402
import staresina_detector as SD                                   # noqa: E402
from tvns.jy_source import bad_annot_mask                         # noqa: E402

T = LP.PROJECT_ROOT / "Final analysis/tables"
RNG = np.random.default_rng(20261018)


def surrogate(x):
    X = np.fft.rfft(x); ph = RNG.uniform(0, 2 * np.pi, X.size); ph[0] = 0
    if x.size % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=x.size)


def main():
    inv = RD.INV[RD.INV.Cohort.isin(["Cohort A", "Cohort B"])].reset_index(drop=True)
    pick = RNG.choice(len(inv), 40, replace=False); rows = []
    for n, i in enumerate(pick):
        row = inv.iloc[i]; raw = RD.mne.io.read_raw_fif(row.fif_path, preload=True)
        if "sleep" not in raw.ch_names:
            continue
        sf = raw.info["sfreq"]; br = RD.block_concat(raw, "stimulation 1")
        if br is None:
            continue
        bad = bad_annot_mask(br); good_min = (~bad).sum() / sf / 60
        for ch in [c for c in RD.CH if c in br.ch_names]:
            x = br.get_data(picks=ch)[0] * 1e6
            n_real = len(SD.detect_spindles(x, sf, bad))
            n_sur = np.mean([len(SD.detect_spindles(surrogate(x), sf, bad)) for _ in range(3)])
            rows.append(dict(session_id=row["Session ID"], ch=ch, good_min=good_min, n_real=n_real, n_surrogate=n_sur))
        print("%d/40" % (n + 1), flush=True)
    d = pd.DataFrame(rows); d["real_per_min"] = d.n_real / d.good_min; d["surrogate_per_min"] = d.n_surrogate / d.good_min
    d.to_csv(T / "REV_spindle_fpr_cells.csv", index=False)
    s = dict(n_recordings=d.session_id.nunique(), n_cells=len(d), real_per_min_median=d.real_per_min.median(), surrogate_per_min_median=d.surrogate_per_min.median(),
             ratio_surrogate_to_real_pct=100 * d.n_surrogate.sum() / d.n_real.sum(), ratio_median_cell_pct=100 * (d.n_surrogate / d.n_real.clip(lower=1)).median())
    pd.DataFrame([s]).to_csv(T / "REV_spindle_fpr.csv", index=False); print(s)


if __name__ == "__main__":
    main()
