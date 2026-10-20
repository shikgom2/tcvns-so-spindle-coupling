# -*- coding: utf-8 -*-
"""Re-detect N2 events with per-window thresholds on the IITY files (stream B (IIT) rescored with YASA (P4)) -> REV_redetect_cells_yasaB.csv.
Uses session() from rev_redetect.py unchanged and only swaps the fif paths to IITY (only the sleep channel differs)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, r"D:/SO-Spindle논문/code/paper")
import rev_redetect as RD                                         # noqa: E402

R = Path(r"D:/SO-Spindle논문/Final analysis/tables")
Y = Path(r"D:/HJ/vns/rawfiles_fromJY/DATA/Data/iit_delivery/Data/IITY")


def main():
    inv = RD.INV[RD.INV.Cohort == "Cohort B"].copy()
    paths = []
    for _, r in inv.iterrows():
        fy = Y / r["Source subject ID"] / os.path.basename(r.fif_path)
        paths.append(str(fy) if fy.exists() else None)
    inv["fif_path"] = paths
    miss = inv.fif_path.isna().sum(); print("stream B recordings", len(inv), "missing YASA file", miss, flush=True)
    inv = inv.dropna(subset=["fif_path"])
    rows, t0 = [], time.time()
    for i, (_, row) in enumerate(inv.iterrows(), 1):
        try:
            r, _ = RD.session(row)
        except Exception as exc:                                   # noqa: BLE001
            print("FAIL", row["Session ID"], exc, flush=True); r = []
        rows += r
        if i % 10 == 0 or i == len(inv):
            el = time.time() - t0
            print("%d/%d  %.1f min  remaining ~%.1f min" % (i, len(inv), el / 60, el / i * (len(inv) - i) / 60), flush=True)
    d = pd.DataFrame(rows); d.to_csv(R / "REV_redetect_cells_yasaB.csv", index=False); print("saved", d.shape)


if __name__ == "__main__":
    main()
