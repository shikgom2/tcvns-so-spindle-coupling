# -*- coding: utf-8 -*-
"""Locate the .fif files of the recording inventory on disk and attach their paths."""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
_R = Path(r"D:/SO-Spindle논문")
P = _R if (_R / "outputs/paper").exists() else _R / "이전"
OUT = P / "outputs/paper"
TBL = _R / "Final analysis/tables"
ROOTS = [_R / "이전", Path(r"D:/HJ/vns/rawfiles_fromJY"), Path(r"E:/SOspindle"),
         Path(r"E:/DATA")]

idx, bystem = {}, {}
for root in ROOTS:
    if not root.exists():
        continue
    for p in root.rglob("*.fif"):
        idx.setdefault(p.name, p)
        # secondary index on the ID+date part only, e.g. '<ID>_<YYYY-MM-DD-hhmm>'
        stem = p.name.split("_raw")[0]
        bystem.setdefault(stem, []).append(p)

inv = pd.read_csv(OUT / "제출용_원본파일목록.csv")
col = "Processed file (.fif)"


def find(name):
    """Prefer the _raw_fin file, which carries the sleep channel."""
    if not isinstance(name, str):
        return None, None
    stem = name.split("_raw")[0]
    for p in bystem.get(stem, []):
        if p.name.endswith("_raw_fin.fif"):
            return str(p), "fin"
    if name in idx:
        return str(idx[name]), "other"
    for p in bystem.get(stem, []):
        return str(p), "other"
    return None, None


res = [find(n) for n in inv[col]]
inv["fif_path"] = [r[0] for r in res]
inv["fif_kind"] = [r[1] for r in res]
inv.to_csv(TBL / "FIF_index.csv", index=False, encoding="utf-8-sig")

L = ["디스크에서 찾은 fif 파일: %d개 (이름 기준)" % len(idx),
     "명단 %d건 중 매칭: %d" % (len(inv), inv.fif_path.notna().sum()),
     "  sleep 채널 있는 _raw_fin : %d" % (inv.fif_kind == "fin").sum(),
     "  _raw 만 있는 것          : %d" % (inv.fif_kind == "raw").sum(),
     "  못 찾음                  : %d" % inv.fif_path.isna().sum(), ""]
L.append(inv.groupby(["Cohort", "Condition"]).fif_kind.apply(
    lambda s: "%d/%d fin" % ((s == "fin").sum(), len(s))).to_string())
L.append("")
L.append("못 찾은 것 예시:")
L.append(inv[inv.fif_path.isna()][["Session ID", "Cohort", "Condition",
                                   col]].head(8).to_string(index=False))
open(r"D:/_tmp_claude/fifidx.txt", "w", encoding="utf-8").write("\n".join(L))
