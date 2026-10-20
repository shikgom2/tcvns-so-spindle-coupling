# -*- coding: utf-8 -*-
"""Spindle-detector specificity check (reviewer: mean+1 SD is lenient / visual check of a random sample).

(1) 8 random recordings × 3 events = 24 spindles: plot the raw signal (0.5–50 Hz preprocessed) and the 12–16 Hz band signal over ±1.5 s → Supplementary Figure 8.
(2) Spindle events on all 13 channels in stimulation-window N2 of 40 random recordings: distributions of duration, peak frequency and amplitude, and
    the share of events whose 12–16 Hz power exceeds the neighbouring bands (8–11, 17–20 Hz) → REV_spindle_specificity.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, r"D:/SO-Spindle논문/code/paper")
import rev_redetect as RD                                         # noqa: E402
import staresina_detector as SD                                   # noqa: E402
from tvns.jy_source import bad_annot_mask                         # noqa: E402
import paperstyle as PS                                           # noqa: E402

R = Path(r"D:/SO-Spindle논문/Final analysis")
T = R / "tables"
FIG = R / "manuscript/figures/v14oct2026/Supplementary_Figure_8_spindle_specificity"
RNG = np.random.default_rng(20261018)


def bp(x, sf, lo, hi):
    b, a = signal.butter(3, [lo / (sf / 2), hi / (sf / 2)], "band"); return signal.filtfilt(b, a, x)


def event_props(x, sf, ev):
    """duration, peak frequency (raw segment), band-passed peak-to-peak amplitude, sigma > neighbours."""
    a, b = ev["onset"], ev["end"]; c = ev["center"]
    seg = x[max(0, c - int(0.5 * sf)):c + int(0.5 * sf)]
    f, p = signal.welch(seg, sf, nperseg=min(len(seg), int(sf)), nfft=1024)
    band = lambda lo, hi: p[(f >= lo) & (f <= hi)].mean()
    sig, lo_n, hi_n = band(12, 16), band(8, 11), band(17, 20)
    m = (f >= 9) & (f <= 20); pk = f[m][np.argmax(p[m])]
    xf = bp(x[max(0, a - int(sf)):b + int(sf)], sf, 12, 16)
    seg_f = xf[int(sf):int(sf) + (b - a)] if len(xf) > int(sf) + (b - a) else xf
    return dict(duration_s=(b - a) / sf, peak_freq_hz=pk, ptp_uV=float(np.ptp(seg_f)) if len(seg_f) else np.nan,
                sigma_over_both_neighbours=bool(sig > lo_n and sig > hi_n), sigma_over_low=bool(sig > lo_n), sigma_over_high=bool(sig > hi_n))


def main():
    PS.use()
    inv = RD.INV[RD.INV.Cohort.isin(["Cohort A", "Cohort B"])].reset_index(drop=True)
    pick = RNG.choice(len(inv), 40, replace=False); rows = []; examples = []
    for n, i in enumerate(pick):
        row = inv.iloc[i]; raw = RD.mne.io.read_raw_fif(row.fif_path, preload=True)
        if "sleep" not in raw.ch_names:
            continue
        sf = raw.info["sfreq"]; br = RD.block_concat(raw, "stimulation 1")
        if br is None:
            continue
        bad = bad_annot_mask(br); chs = [c for c in RD.CH if c in br.ch_names]
        cand = []
        for ch in chs:
            x = br.get_data(picks=ch)[0] * 1e6
            evs = SD.detect_spindles(x, sf, bad)
            for ev in evs:
                rows.append(dict(session_id=row["Session ID"], ch=ch, **event_props(x, sf, ev)))
            if len(evs) >= 3:
                cand.append((ch, x, evs))
        if len(examples) < 24 and cand:
            ch, x, evs = cand[RNG.integers(len(cand))]
            for ev in RNG.choice(len(evs), 3, replace=False):
                examples.append((row["Session ID"], ch, x, sf, evs[ev]))
        print("%d/40 %s events so far %d" % (n + 1, row["Session ID"], len(rows)), flush=True)
    df = pd.DataFrame(rows); df.to_csv(T / "REV_spindle_specificity_events.csv", index=False)
    summ = dict(n_recordings=df.session_id.nunique(), n_events=len(df),
                duration_median=df.duration_s.median(), duration_q1=df.duration_s.quantile(.25), duration_q3=df.duration_s.quantile(.75),
                peak_freq_median=df.peak_freq_hz.median(), peak_freq_q1=df.peak_freq_hz.quantile(.25), peak_freq_q3=df.peak_freq_hz.quantile(.75),
                peak_freq_in_12_16_pct=100 * df.peak_freq_hz.between(12, 16).mean(), peak_freq_in_11_17_pct=100 * df.peak_freq_hz.between(11, 17).mean(),
                ptp_median_uV=df.ptp_uV.median(), ptp_q1=df.ptp_uV.quantile(.25), ptp_q3=df.ptp_uV.quantile(.75),
                sigma_over_both_neighbours_pct=100 * df.sigma_over_both_neighbours.mean(), sigma_over_low_pct=100 * df.sigma_over_low.mean(), sigma_over_high_pct=100 * df.sigma_over_high.mean())
    pd.DataFrame([summ]).to_csv(T / "REV_spindle_specificity.csv", index=False); print(summ)
    # ── figure: 24 examples (6 rows × 4 cols) + 3 distribution panels
    f = plt.figure(figsize=(7.09, 9.2)); gs = f.add_gridspec(8, 4, hspace=.9, wspace=.35, left=.07, right=.98, top=.96, bottom=.05)
    ex = examples[:24]
    for k, (sid, ch, x, sf, ev) in enumerate(ex):
        ax = f.add_subplot(gs[k // 4, k % 4]); c = ev["center"]; w = int(1.5 * sf)
        seg = x[max(0, c - w):c + w]; t = (np.arange(len(seg)) - min(c, w)) / sf
        xf = bp(x, sf, 12, 16)[max(0, c - w):c + w]
        ax.plot(t, seg, color="#9AA0A4", lw=.6); ax.plot(t, xf - np.ptp(seg) * .6, color=PS.C["stim"], lw=.7)
        ax.axvspan((ev["onset"] - c) / sf, (ev["end"] - c) / sf, color=PS.C["stim"], alpha=.12, lw=0)
        ax.set_xlim(-1.5, 1.5); ax.set_yticks([]); ax.tick_params(labelsize=5.5); ax.spines["left"].set_visible(False)
        ax.set_title("%s %s" % (sid, ch), fontsize=6, pad=2)
        if k // 4 == 5: ax.set_xlabel("Time from spindle centre (s)", fontsize=6)
    ax = f.add_subplot(gs[6:8, 0:2]); ax.hist(df.peak_freq_hz, bins=np.arange(9, 20.5, .5), color=PS.C["stim"], alpha=.85); ax.axvspan(12, 16, color="#999", alpha=.15, lw=0)
    ax.set_xlabel("Peak frequency of raw segment (Hz)", fontsize=6.5); ax.set_ylabel("Events", fontsize=6.5); ax.tick_params(labelsize=6)
    ax = f.add_subplot(gs[6:8, 2]); ax.hist(df.duration_s, bins=np.arange(.4, 3.05, .1), color=PS.C["stim"], alpha=.85); ax.set_xlabel("Duration (s)", fontsize=6.5); ax.tick_params(labelsize=6)
    ax = f.add_subplot(gs[6:8, 3]); ax.hist(df.ptp_uV.clip(upper=df.ptp_uV.quantile(.99)), bins=30, color=PS.C["stim"], alpha=.85); ax.set_xlabel("12–16 Hz peak-to-peak (µV)", fontsize=6.5); ax.tick_params(labelsize=6)
    f.text(.07, .985, "a", fontsize=10, fontweight="bold"); f.text(.07, .245, "b", fontsize=10, fontweight="bold"); f.text(.52, .245, "c", fontsize=10, fontweight="bold"); f.text(.76, .245, "d", fontsize=10, fontweight="bold")
    f.savefig(FIG.with_suffix(".png"), dpi=300, bbox_inches="tight"); f.savefig(FIG.with_suffix(".pdf"), bbox_inches="tight"); print("saved figure", len(ex), "examples")


if __name__ == "__main__":
    main()
