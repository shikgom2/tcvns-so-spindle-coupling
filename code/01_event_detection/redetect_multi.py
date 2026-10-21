# -*- coding: utf-8 -*-
"""Run three detection methods × sleep-stage variants over the FT/FS folders.

  jy         original thesis pipeline (tvns.jy_source)            — reference. G3 r=1.000 verified
  iy         SO=original thesis pipeline + spindle=YASA(IRASA individual band) — existing project track='iy'
  staresina  Staresina 2023 (doi:10.1038/s41593-023-01381-w) — implementation not yet validated, exploratory

  Stages:  N2  |  N2N3

Two deliberate departures from the paper in the Staresina track (based on the pilot):
  · Threshold scope = per block (the paper uses all NREM). With an all-NREM threshold, events
    concentrated in deep-sleep blocks regardless of block length and n_SO reversed
    (stimulation 136 < recovery 267). Per block, the scope matches the 75th-percentile scope of the
    original thesis pipeline, so the comparison changes only the 'threshold rule'.
  · Coupled ratio = phase rule of the original thesis pipeline (within the SO interval + |phase|<=90 deg).
    The paper's ±1 s co-occurrence rule pushes the ratio toward 1 when spindle density is high
    (0.428 in the pilot). The ±1 s decision is kept separately in the coupled_ratio_win1s column.

  python redetect_multi.py jy_N2N3 iy_N2 staresina_N2
"""
from __future__ import annotations

import io
import contextlib
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402
sys.path.insert(0, str(LP.PROJECT_ROOT / "reanalysis_code"))

import mne
mne.set_log_level("ERROR")
from tvns import cohorts as C
from tvns.jy_source import bad_annot_mask, group_consecutive, EEG_CFG
from tvns.jy_detector import build_events_for_session
import staresina_detector as SD

BASE = LP.PROJECT_ROOT / "ppt_headline_cells"
OUT = LP.PROJECT_ROOT / "redetect_multi"
OUT.mkdir(parents=True, exist_ok=True)
AUD = pd.read_csv(LP.AUDIT_ROOT / "outputs/audit/AUDIT_sessions.csv")
AUD["stem"] = AUD["파일"].str.replace("_raw_fin.fif", "", regex=False)
AUD["date"] = AUD.stem.str.extract(r"_(\d{4}-\d{2}-\d{2}-\d{4})")
BLOCKS = ["baseline", "stimulation 1", "recovery 1"]
CH = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "Cz", "C4", "P3", "P4"]
MAP = EEG_CFG.sleep_staging.stage_mapping


def jobs():
    maps = {}
    for arm, sub, col in (("FT", "treatment", "FT_ID"), ("FS", "sham", "FS_ID")):
        m = pd.read_csv(BASE / sub / f"{arm}_id_map.csv")
        maps[arm] = dict(zip(m[col], m["원ID"]))
    out = []
    for arm, sub in (("FT", "treatment"), ("FS", "sham")):
        for d in sorted(p for p in (BASE / sub).iterdir() if p.is_dir()):
            for f in sorted(d.glob("*_raw_fin.fif")):
                date = re.search(r"_(\d{4}-\d{2}-\d{2}-\d{4})", f.name).group(1)
                orig = maps[arm][d.name]
                a = AUD[(AUD.ID == orig) & (AUD.date == date)]
                out.append(dict(arm=arm, label=d.name, orig=orig, date=date, path=f,
                                cohort=("IIT" if orig.startswith("I") else "TB"),
                                order=(int(a.Order.iloc[0]) if len(a) else None),
                                group=(a.Group.iloc[0] if len(a) else None)))
    return out


def block_concat(raw, block, labels):
    ann = next((a for a in raw.annotations if a["description"] == block), None)
    if ann is None:
        return None
    seg = raw.copy().crop(tmin=ann["onset"],
                          tmax=min(ann["onset"] + ann["duration"], raw.times[-1]))
    slp = seg.get_data(picks="sleep")[0]
    idx = np.flatnonzero(np.isin(slp, [MAP[l] for l in labels]))
    if idx.size == 0:
        return None
    parts = [seg.copy().crop(tmin=seg.times[g[0]], tmax=seg.times[g[-1]])
             for g in group_consecutive(idx) if len(g)]
    return mne.concatenate_raws(parts) if parts else None


def run_jy(j, labels):
    """If the stage is N2 only, use the original function as is. For N2+N3, build the blocks directly and pass them in."""
    s = C.Session(cohort=j["cohort"], subject=j["label"], session_id=str(j["order"]),
                  order=(float(j["order"]) if j["order"] else None),
                  group=j["group"], track_source="fif", path=j["path"])
    if labels == ["N2"]:
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, _, summ = build_events_for_session(s, j["cohort"], blocks=BLOCKS)
        return summ
    from tvns.jy_detector import run_channel_block, summary_row, _meta_row
    raw = mne.io.read_raw_fif(str(j["path"]), preload=True, verbose="ERROR")
    rows = []
    for blk in BLOCKS:
        br = block_concat(raw, blk, labels)
        if br is None:
            continue
        bad = bad_annot_mask(br)
        dur = float(br.times[-1] - br.times[0]) if len(br.times) else 0.0
        for ch in CH:
            if ch not in br.ch_names:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                res = run_channel_block(br, bad, ch, 180.0)
            rows.append(summary_row(res, _meta_row(s, j["cohort"], blk, ch), dur))
    return pd.DataFrame(rows)


def run_iy(j, labels):
    from tvns.iy_detector import run_channel_block_iy, personal_band
    from tvns.jy_detector import summary_row, _meta_row
    s = C.Session(cohort=j["cohort"], subject=j["label"], session_id=str(j["order"]),
                  order=(float(j["order"]) if j["order"] else None),
                  group=j["group"], track_source="fif", path=j["path"])
    raw = mne.io.read_raw_fif(str(j["path"]), preload=True, verbose="ERROR")
    bands = {}
    for ch in CH:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                bands[ch] = tuple(personal_band(s, ch, blocks=BLOCKS)["band"])
        except Exception:
            bands[ch] = (12.0, 16.0)
    rows = []
    for blk in BLOCKS:
        br = block_concat(raw, blk, labels)
        if br is None:
            continue
        bad = bad_annot_mask(br)
        dur = float(br.times[-1] - br.times[0]) if len(br.times) else 0.0
        for ch in CH:
            if ch not in br.ch_names:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                res = run_channel_block_iy(br, bad, ch, bands[ch], 180.0)
            rows.append(summary_row(res, _meta_row(s, j["cohort"], blk, ch), dur))
    return pd.DataFrame(rows)


def run_staresina(j, labels):
    raw = mne.io.read_raw_fif(str(j["path"]), preload=True, verbose="ERROR")
    rows = []
    for blk in BLOCKS:
        br = block_concat(raw, blk, labels)
        if br is None:
            continue
        bad = bad_annot_mask(br)
        dur = float(br.times[-1] - br.times[0]) if len(br.times) else 0.0
        for ch in CH:
            if ch not in br.ch_names:
                continue
            x = br.get_data(picks=ch)[0] * 1e6
            r = SD.analyse_channel(x, br.info["sfreq"],
                                   use_paper_artifact=False, bad_external=bad, th=None)
            rows.append(dict(cohort=j["cohort"], subject=j["label"], block=blk, ch=ch,
                             n_SO=r["n_SO"], n_spindle=r["n_spindle"],
                             n_coupled=r["n_coupled_jyrule"],
                             coupled_ratio=r["coupled_ratio_jyrule"],
                             MRL=r["MRL_jyrule"], mean_phase_deg=r["mean_phase_deg_jyrule"],
                             n_coupled_win1s=r["n_coupled"],
                             coupled_ratio_win1s=r["coupled_ratio"],
                             duration_sec=dur))
    return pd.DataFrame(rows)


RUN = {"jy": run_jy, "iy": run_iy, "staresina": run_staresina}


def main(tasks):
    J = jobs()
    print(f"세션 {len(J)}개", flush=True)
    for task in tasks:
        meth, stg = task.rsplit("_", 1)
        labels = ["N2"] if stg == "N2" else ["N2", "N3"]
        fn = RUN[meth]
        rows, fails, t0 = [], [], time.time()
        print(f"\n### {task}  ({meth}, {'+'.join(labels)})", flush=True)
        for i, j in enumerate(J, 1):
            try:
                summ = fn(j, labels)
                if summ is None or summ.empty:
                    fails.append((j["label"], j["date"], "빈 결과")); continue
                rows.append(summ.assign(arm=j["arm"], label=j["label"], 원ID=j["orig"],
                                        date=j["date"], 코호트=j["cohort"],
                                        Order=j["order"], DB_Group=j["group"]))
            except Exception as e:
                fails.append((j["label"], j["date"], f"{type(e).__name__}: {e}"))
            if i % 25 == 0 or i == len(J):
                el = time.time() - t0
                print(f"  {i}/{len(J)}  {el/60:.1f}분 경과 · 남은 추정 "
                      f"{(el/i)*(len(J)-i)/60:.1f}분 · 실패 {len(fails)}", flush=True)
        if rows:
            pd.concat(rows, ignore_index=True).to_parquet(OUT / f"{task}.parquet", index=False)
        pd.DataFrame(fails, columns=["label", "date", "이유"]).to_csv(
            OUT / f"{task}_failures.csv", index=False, encoding="utf-8-sig")
        print(f"  저장 → {task}.parquet  (실패 {len(fails)})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["jy_N2N3", "staresina_N2", "iy_N2"])
