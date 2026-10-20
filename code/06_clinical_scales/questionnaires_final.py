# -*- coding: utf-8 -*-
"""Questionnaires, final — 5 scales, EEG population of 75 participants.

Produces all three value-handling variants.
  V1 raw        no values altered
  V2 primary    only ESS -800 (missing-value code) set to missing.  <- primary analysis
  V3 mad        in addition, remove values outside median ± 3 MAD (sensitivity)
-800 is not an outlier but a code meaning 'not recorded', and it occurs only in ESS.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
_R = Path(r"D:/SO-Spindle논문")
P = _R if (_R / "outputs/paper").exists() else _R / "이전"
T = _R / "Final analysis/tables"
T.mkdir(parents=True, exist_ok=True)
XLSB = Path(r"D:/HJ/vns/rawfiles_fromJY/DATA/Data/Data/tVNS_Sleep.xlsb")
IITF = Path(r"D:/HJ/vns/rawfiles_fromJY/DATA/Data/iit_delivery/Data"
            "/IIT_Assessments.xlsx")

# Participants excluded from the questionnaire analyses (no analysable EEG).
# The ID list is kept outside version control: config/excluded_participants.txt
# (one ID per line; see config/excluded_participants.example.txt).
DROP = [ln.strip() for ln in (Path(__file__).resolve().parents[2] / "config" / "excluded_participants.txt")
        .read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
SCALES = {"PSQI": (0, 21), "ISI": (0, 28), "ESS": (0, 24),
          "BAI": (0, 63), "BDI-2": (0, 63)}
RNG = np.random.default_rng(20261003)


def mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size >= 3 \
        else np.nan


def fdr(p):
    p = np.asarray(p, float)
    o = np.full_like(p, np.nan)
    k = np.isfinite(p)
    m = int(k.sum())
    if m == 0:
        return o
    i = np.argsort(p[k])
    q = np.minimum.accumulate((p[k][i] * m / (np.arange(m) + 1))[::-1])[::-1]
    t = np.empty(m)
    t[i] = q.clip(0, 1)
    o[k] = t
    return o


def load():
    a = pd.read_excel(XLSB, sheet_name="Assessments", engine="pyxlsb")
    a = a[a.Exp_type.astype(str).str.upper() == "TB"].copy()
    a["ID"] = a.ID.astype(str)
    a = a.sort_values(["ID", "Date"])
    a["Order"] = a.groupby("ID").cumcount() + 1
    a["cohort"] = "Cohort A"
    b = pd.read_excel(IITF)
    b["ID"] = b.ID.astype(str)
    b["Order"] = pd.to_numeric(b.Order, errors="coerce")
    b = b[b.Order.notna()].copy()
    b["Order"] = b.Order.astype(int)
    b["cohort"] = "Cohort B"
    d = pd.concat([a[["ID", "cohort", "Order"] + list(SCALES)],
                   b[["ID", "cohort", "Order"] + list(SCALES)]],
                  ignore_index=True)
    for s in SCALES:
        d[s] = pd.to_numeric(d[s], errors="coerce")
    return d[~d.ID.isin(DROP)].copy()


def v_raw(d):
    return d.copy(), pd.DataFrame()


def v_primary(d):
    """Set values outside the scale range to missing. In practice only ESS -800."""
    out, rec = d.copy(), []
    for s, (lo, hi) in SCALES.items():
        v = out[s]
        bad = v.notna() & ((v < lo) | (v > hi))
        for i in out.index[bad]:
            rec.append(dict(ID=out.at[i, "ID"], cohort=out.at[i, "cohort"],
                            Order=out.at[i, "Order"], scale=s,
                            value=float(v[i]), rule="outside scale range",
                            valid="%g-%g" % (lo, hi)))
        out[s] = v.mask(bad)
    return out, pd.DataFrame(rec)


def v_mad(d):
    out, rec = v_primary(d)
    rec = rec.to_dict("records")
    for s in SCALES:
        v = out[s].copy()
        for c in out.cohort.unique():
            k = out.cohort == c
            vv = v[k].dropna()
            if len(vv) < 5:
                continue
            m, md = np.median(vv), mad(vv)
            if not np.isfinite(md) or md <= 0:
                continue
            bad = v.notna() & k & (np.abs(v - m) > 3 * md)
            for i in out.index[bad]:
                rec.append(dict(ID=out.at[i, "ID"], cohort=c,
                                Order=out.at[i, "Order"], scale=s,
                                value=float(v[i]), rule="median ± 3 MAD",
                                valid="%s %.1f ± %.1f" % (c, m, 3 * md)))
            v = v.mask(bad)
        out[s] = v
    return out, pd.DataFrame(rec)


def change_per_id(d, s):
    w = d.dropna(subset=[s]).sort_values("Order")
    n = w.groupby(["cohort", "ID"]).Order.nunique()
    keep = n[n >= 2].index
    return (w.groupby(["cohort", "ID"])[s].last()[keep]
            - w.groupby(["cohort", "ID"])[s].first()[keep]).dropna()


def pooled(d):
    rows = []
    for s in SCALES:
        w = d.dropna(subset=[s]).sort_values("Order")
        n = w.groupby(["cohort", "ID"]).Order.nunique()
        keep = n[n >= 2].index
        if len(keep) < 8:
            continue
        f = w.groupby(["cohort", "ID"])[s].first()[keep]
        l = w.groupby(["cohort", "ID"])[s].last()[keep]
        dd = (l - f).dropna()
        sd = dd.std(ddof=1)
        bt = RNG.choice(dd.values, (4000, len(dd)), True).mean(1)
        a = dd[dd.index.get_level_values(0) == "Cohort A"].values
        c = dd[dd.index.get_level_values(0) == "Cohort B"].values
        rows.append(dict(
            scale=s, n=len(dd), n_A=len(a), n_B=len(c),
            first=float(f.mean()), last=float(l.mean()),
            change=float(dd.mean()),
            ci_lo=float(np.percentile(bt, 2.5)),
            ci_hi=float(np.percentile(bt, 97.5)),
            dz=float(dd.mean() / sd) if sd else np.nan,
            n_improved=int((dd < 0).sum()),
            p=float(stats.wilcoxon(dd).pvalue),
            change_A=float(np.mean(a)), change_B=float(np.mean(c)),
            p_cohort_diff=float(stats.mannwhitneyu(a, c).pvalue)))
    r = pd.DataFrame(rows)
    r["q"] = fdr(r.p.values)
    return r


raw = load()
VAR = [("raw", v_raw, "no values altered"),
       ("primary", v_primary, "ESS −800 recoded as missing"),
       ("mad", v_mad, "plus median ± 3 MAD removal")]
L = ["참가자 %d명 (A %d / B %d) · 척도 5개 · FDR 가족 = 5"
     % (raw.ID.nunique(),
        raw[raw.cohort == "Cohort A"].ID.nunique(),
        raw[raw.cohort == "Cohort B"].ID.nunique())]

RES, REC = {}, {}
for tag, fn, _ in VAR:
    d2, rec = fn(raw)
    RES[tag] = pooled(d2)
    REC[tag] = rec
    RES[tag].to_csv(T / ("Q5_change_%s.csv" % tag), index=False,
                    encoding="utf-8-sig")

# all out-of-range values
oor = REC["primary"]
oor.to_csv(T / "Q5_out_of_range_values.csv", index=False,
           encoding="utf-8-sig")
L.append("")
L.append("=== 범위 밖 값 (주 분석에서 결측 처리한 것) ===")
L.append("  총 %d개" % len(oor))
if len(oor):
    L.append("  척도별: %s" % oor.groupby("scale").size().to_dict())
    L.append("  고유값: %s" % sorted(oor.value.unique()))
    L.append("  해당 참가자 %d명, 코호트: %s"
             % (oor.ID.nunique(), oor.groupby("cohort").ID.nunique().to_dict()))
    L.append("  영향받은 방문 회차: %s" % sorted(oor.Order.unique()))
    # do participants with -800 still have valid ESS on 2 or more visits?
    pr, _ = v_primary(raw)
    still = change_per_id(pr, "ESS")
    hit = set(oor.ID)
    L.append("  -800 을 가진 %d명 중 ESS 변화 계산이 여전히 가능한 사람: %d명"
             % (len(hit), sum(1 for (_c, i) in still.index if i in hit)))

mad_only = REC["mad"]
mad_only = mad_only[mad_only.rule == "median ± 3 MAD"]
mad_only.to_csv(T / "Q5_mad_removed_values.csv", index=False,
                encoding="utf-8-sig")
L.append("")
L.append("=== MAD 규칙이 추가로 제거하는 값 (민감도 판에서만) ===")
L.append("  총 %d개" % len(mad_only))
if len(mad_only):
    L.append(mad_only[["ID", "cohort", "Order", "scale", "value", "valid"]]
             .to_string(index=False))

cols = ["scale", "n", "n_A", "n_B", "first", "last", "change", "ci_lo",
        "ci_hi", "dz", "n_improved", "p", "q"]
for tag, _, desc in VAR:
    L.append("")
    L.append("=" * 94)
    L.append("##  %s  —  %s" % (tag, desc))
    L.append("=" * 94)
    L.append(RES[tag][cols].round(3).to_string(index=False))
    L.append("  q<.05 : %d/5   코호트 조절효과 p : %s"
             % (int((RES[tag].q < .05).sum()),
                ", ".join("%s=%.3f" % (r.scale, r.p_cohort_diff)
                          for r in RES[tag].itertuples())))

m = (RES["raw"][["scale", "n", "change", "dz", "q"]]
     .merge(RES["primary"][["scale", "n", "change", "dz", "q"]], on="scale",
            suffixes=("_raw", "_primary"))
     .merge(RES["mad"][["scale", "n", "change", "dz", "q"]].add_suffix("_mad")
            .rename(columns={"scale_mad": "scale"}), on="scale"))
m.to_csv(T / "Q5_three_variants.csv", index=False, encoding="utf-8-sig")
L.append("")
L.append("=== 세 판 나란히 ===")
L.append(m.round(4).to_string(index=False))
open(r"D:/_tmp_claude/q5.txt", "w", encoding="utf-8").write("\n".join(L))
