# -*- coding: utf-8 -*-
"""Revision analyses — answer reviewer points 5 and 7 using the rev_redetect.py outputs.

  A  replication          are the new n_SO / n_coupled identical to S1?
  B  threshold sensitivity condition x window interaction of coupling prevalence re-detected with recording-wide thresholds
  C  chance level         interaction of surrogate-mean-subtracted prevalence (observed - chance), per-cell excess over chance
  D  polarity             distribution of spindle-centre phase (which extremum it clusters at), interaction under the opposite rule
  E  window length, SO count  distributions by condition and window (supplementary table)
  F  stimulation artefact high-resolution PSD: line-component index at multiples of 5 Hz, stimulation window - pre window, both conditions

Output  tables/REV_*.csv,  manuscript/figures/v14oct2026/Supplementary_Figure_7_stimulation_artefact.(png|pdf)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

R = Path(r"D:/SO-Spindle논문/Final analysis")
T = R / "tables"
FIG = R / "manuscript/figures/v14oct2026"
S1 = pd.read_csv(R / "Brain submission/supplementary tables/Supplementary_Table_01_S1_session_metrics.csv")
S1 = S1[S1.stage == "N2"]
REG = {"Frontal": ["Fp1", "Fp2", "F3", "F4"], "Temporal": ["F7", "F8", "T3", "T4"],
       "Central": ["C3", "Cz", "C4"], "Parietal": ["P3", "P4"]}
CH2REG = {c: r for r, cs in REG.items() for c in cs}
REGS = list(REG)
HARM = np.arange(5, 56, 5)          # 60 Hz excluded because it lies within the anti-aliasing roll-off


def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = np.isfinite(p)
    pp = p[ok]; o = np.argsort(pp); m = len(pp); out = np.empty(m); prev = 1.0
    for rank, i in enumerate(o[::-1], 1):
        prev = min(prev, pp[i] * m / (m - rank + 1)); out[i] = prev
    q[ok] = out; return q


def lmm(d, y, label):
    rows = []
    for reg in REGS:
        x = d[d.ROI == reg].dropna(subset=[y]).copy()
        x["trt"] = (x.arm == "Treatment").astype(int)
        x["win"] = pd.Categorical(x.win, ["pre", "stim", "post"])
        try:
            md = smf.mixedlm("%s ~ trt*C(win)" % y, x, groups=x.participant).fit()
            k = "trt:C(win)[T.stim]"
            rows.append(dict(analysis=label, ROI=reg, n_part=x.participant.nunique(), n_rows=len(x),
                             beta=md.params[k], ci_lo=md.conf_int().loc[k, 0], ci_hi=md.conf_int().loc[k, 1],
                             p=md.pvalues[k]))
        except Exception as exc:                                   # noqa: BLE001
            rows.append(dict(analysis=label, ROI=reg, n_part=np.nan, n_rows=len(x), beta=np.nan,
                             ci_lo=np.nan, ci_hi=np.nan, p=np.nan, note=str(exc)[:40]))
    o = pd.DataFrame(rows); o["q"] = bh(o.p.values); return o


def main():
    c = pd.read_csv(T / "REV_redetect_cells.csv")
    c["ROI"] = c.ch.map(CH2REG)
    agg = dict(n_SO=("n_SO", "sum"), n_coupled=("n_coupled", "sum"), n_any=("n_any_spindle", "sum"),
               n_opp=("n_coupled_opposite", "sum"), n_surr=("n_coupled_surr_mean", "sum"),
               n_SO_rw=("n_SO_rw", "sum"), n_cp_rw=("n_coupled_rw", "sum"), n_sp=("n_spindle", "sum"),
               n_sp_rw=("n_spindle_rw", "sum"), duration_sec=("duration_sec", "first"))
    g = c.groupby(["session_id", "participant", "arm", "cohort", "win", "ROI"], as_index=False).agg(**agg)
    g["prevalence"] = g.n_coupled / g.n_SO.replace(0, np.nan)
    g["prev_rw"] = g.n_cp_rw / g.n_SO_rw.replace(0, np.nan)
    g["prev_opp"] = g.n_opp / g.n_SO.replace(0, np.nan)
    g["chance"] = g.n_surr / g.n_SO.replace(0, np.nan)
    g["prev_excess"] = g.prevalence - g.chance
    g["any_frac"] = g.n_any / g.n_SO.replace(0, np.nan)
    g["phase_pass"] = g.n_coupled / g.n_any.replace(0, np.nan)
    # same cell criteria as the manuscript (>=20 s, >=5 SO)
    ok = (g.duration_sec >= 20) & (g.n_SO >= 5)
    g = g[ok].copy()
    g.to_csv(T / "REV_region_cells.csv", index=False)

    # ── A replication
    m = g.merge(S1, on=["session_id", "win", "ROI"], suffixes=("", "_s1"))
    print("A  replication: cells %d | n_SO identical %.3f | n_coupled identical %.3f | prevalence r = %.4f"
          % (len(m), (m.n_SO == m.n_SO_s1).mean(), (m.n_coupled == m.n_coupled_s1).mean(),
             np.corrcoef(m.prevalence.fillna(0), m.prevalence_s1.fillna(0))[0, 1]))

    # ── B, C, D interactions
    out = pd.concat([lmm(g, "prevalence", "window-specific thresholds (primary)"),
                     lmm(g, "prev_rw", "recording-wide thresholds"),
                     lmm(g, "prev_excess", "chance-corrected (observed - surrogate)"),
                     lmm(g, "prev_opp", "opposite phase convention"),
                     lmm(g, "any_frac", "spindle inside SO, no phase criterion"),
                     lmm(g, "phase_pass", "phase criterion given a spindle inside")], ignore_index=True)
    out.to_csv(T / "REV_interaction_sensitivity.csv", index=False)
    print("\nB-D  condition x window interaction (stim vs pre)")
    print(out.to_string(index=False, float_format=lambda v: "%.4f" % v))

    # ── C relative to chance: observed / chance, per cell and by condition and window
    g["ratio_to_chance"] = g.prevalence / g.chance.replace(0, np.nan)
    cc = g.groupby(["arm", "win"]).agg(prev=("prevalence", "mean"), chance=("chance", "mean"),
                                       excess=("prev_excess", "mean"), ratio=("ratio_to_chance", "median"),
                                       phase_pass=("phase_pass", "mean"), any_frac=("any_frac", "mean"),
                                       n=("prevalence", "size")).reset_index()
    cc.to_csv(T / "REV_chance_level.csv", index=False)
    print("\nC  observed vs chance (cell means)"); print(cc.round(4).to_string(index=False))
    # per-cell surrogate P (fraction of cells where the observed value lies above the surrogate distribution)
    cp = c.dropna(subset=["n_coupled_surr_p"])
    print("   channel-window cells with observed > 95th pct of surrogates: %.3f (of %d)"
          % ((cp.n_coupled_surr_p < 0.05).mean(), len(cp)))

    # ── D polarity: mean angle of spindle-centre phase (channel x window cells, spindles >= 20)
    ph = c[(c.n_sp_phase >= 20)].copy()
    v = np.exp(1j * np.radians(ph.sp_phase_deg)) * ph.sp_phase_mrl
    mu = np.degrees(np.angle(v.mean()))
    near0 = (np.abs(ph.sp_phase_deg) <= 90).mean()
    print("\nD  spindle-centre phase on the SO filtered signal as recorded: grand mean %+.1f deg; %.1f%% of cells cluster within +-90 of the positive peak (n=%d cells)"
          % (mu, near0 * 100, len(ph)))
    pd.DataFrame(dict(grand_mean_deg=[mu], frac_cells_near_positive_peak=[near0], n_cells=[len(ph)])).to_csv(T / "REV_polarity_summary.csv", index=False)

    # ── E window length and SO count
    e = g.groupby(["arm", "win", "ROI"]).agg(
        n_cells=("n_SO", "size"), dur_median=("duration_sec", "median"),
        dur_q1=("duration_sec", lambda x: x.quantile(.25)), dur_q3=("duration_sec", lambda x: x.quantile(.75)),
        nSO_median=("n_SO", "median"), nSO_q1=("n_SO", lambda x: x.quantile(.25)),
        nSO_q3=("n_SO", lambda x: x.quantile(.75)), frac_nSO_lt20=("n_SO", lambda x: (x < 20).mean()),
        frac_dur_lt120s=("duration_sec", lambda x: (x < 120).mean())).reset_index()
    e.to_csv(T / "REV_window_durations_counts.csv", index=False)

    # ── F artefact: line-component index
    z = np.load(T / "REV_psd.npz")
    f = z["freqs"]
    keys = [k for k in z.files if k != "freqs"]
    arm = dict(zip(c.session_id, c.arm)); coh = dict(zip(c.session_id, c.cohort))
    rows, spec = [], {}
    for k in keys:
        sid, win, ch = k.split("|")
        p = z[k].astype(float)
        reg = CH2REG.get(ch)
        if reg is None:
            continue
        spec.setdefault((sid, win, reg), []).append(p)
    recs = []
    for (sid, win, reg), ps in spec.items():
        p = np.mean(ps, 0)
        r = dict(session_id=sid, arm=arm[sid], cohort=coh[sid], win=win, ROI=reg)
        for h in HARM:
            at = (np.abs(f - h) <= 0.13)
            nb = (np.abs(f - h) <= 1.0) & (np.abs(f - h) > 0.3)
            r["line_%d" % h] = float(np.log10(p[at].mean()) - np.log10(np.median(p[nb])))
        # sharpest line component within the sigma band (0.125 Hz resolution, 12-16 Hz)
        sig = (f >= 12) & (f <= 16)
        ls = np.log10(p[sig]); r["sigma_peakiness"] = float(ls.max() - np.median(ls))
        r["sigma_power"] = float(p[sig].mean()); r["so_band_power"] = float(p[(f >= 0.5) & (f <= 1.0)].mean())
        recs.append(r)
        spec[(sid, win, reg)] = p
    L = pd.DataFrame(recs)
    L.to_csv(T / "REV_line_components.csv", index=False)
    # window change (stim - pre), by condition, region mean
    w = L.pivot_table(index=["session_id", "arm", "ROI"], columns="win",
                      values=["line_%d" % h for h in HARM] + ["sigma_peakiness"])
    rows = []
    for col in ["line_%d" % h for h in HARM] + ["sigma_peakiness"]:
        if ("pre" not in w[col]) or ("stim" not in w[col]):
            continue
        d = (w[col]["stim"] - w[col]["pre"]).dropna().reset_index()
        t = d[d.arm == "Treatment"][0]; s = d[d.arm == "Sham"][0]
        rows.append(dict(measure=col, trt_change=t.mean(), trt_P=stats.wilcoxon(t).pvalue if len(t) > 5 else np.nan, n_t=len(t),
                         sham_change=s.mean(), sham_P=stats.wilcoxon(s).pvalue if len(s) > 5 else np.nan, n_s=len(s),
                         between_P=stats.mannwhitneyu(t, s).pvalue if len(t) > 5 and len(s) > 5 else np.nan,
                         trt_stim_level=w[col]["stim"].reset_index().query("arm=='Treatment'")[("stim")].mean() if False else np.nan))
    LC = pd.DataFrame(rows); LC["between_q"] = bh(LC.between_P.values)
    LC.to_csv(T / "REV_line_components_change.csv", index=False)
    print("\nF  line-component index (log10 ratio to 1-Hz neighbourhood), stim - pre")
    print(LC.to_string(index=False, float_format=lambda v: "%.3f" % v))
    # absolute level: condition comparison in the stimulation window
    lv = L[L.win == "stim"].groupby(["arm"])[["line_%d" % h for h in HARM] + ["sigma_peakiness"]].mean().T
    print("\n   stimulation-window level by arm:"); print(lv.round(3).to_string())

    # ── Figure: Supplementary Figure 7
    import paperstyle as PS
    PS.use()
    STIM, NOST, INK, INK2 = PS.C["stim"], PS.C["nost"], PS.C["ink"], PS.C["ink2"]
    fig = plt.figure(figsize=(7.09, 5.6))
    gs = fig.add_gridspec(2, 2, wspace=.3, hspace=.55, left=.08, right=.98, top=.93, bottom=.1)
    # a,b: mean spectra (frontal, parietal), stimulation window, both conditions, 0-62.5 Hz, log scale
    for j, reg in enumerate(["Frontal", "Parietal"]):
        ax = fig.add_subplot(gs[0, j])
        for a, col in (("Treatment", STIM), ("Sham", NOST)):
            ps = [v for (sid, win, r), v in spec.items() if r == reg and win == "stim" and arm[sid] == a]
            m_ = np.median(ps, 0)
            ax.plot(f, m_, color=col, lw=1.0, label="Stimulation" if a == "Treatment" else "No stimulation")
            pp = [v for (sid, win, r), v in spec.items() if r == reg and win == "pre" and arm[sid] == a]
            ax.plot(f, np.median(pp, 0), color=col, lw=.8, ls=":", alpha=.9)
        for h in HARM:
            ax.axvline(h, color=INK2, lw=.4, alpha=.35)
        ax.axvspan(12, 16, color="#999", alpha=.12, lw=0)
        ax.set_yscale("log"); ax.set_xlim(0, 62.5)
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power (µV² Hz⁻¹), median")
        ax.set_title("%s — stimulation window (solid) and pre-stimulation (dotted)" % reg, fontsize=7.6)
        if j == 0:
            ax.legend(fontsize=6.2, frameon=False, loc="upper right")
    # c: line-component index, stimulation window, both conditions, region mean
    ax = fig.add_subplot(gs[1, 0])
    lvl = L[L.win == "stim"].groupby("arm")[["line_%d" % h for h in HARM]]
    for a, col, dx in (("Treatment", STIM, -.9), ("Sham", NOST, .9)):
        mm = lvl.mean().loc[a].values; se = lvl.sem().loc[a].values
        ax.bar(HARM + dx, mm, width=1.7, color=col, alpha=.85, lw=0)
        ax.errorbar(HARM + dx, mm, yerr=1.96 * se, fmt="none", ecolor=INK, lw=.8, capsize=1.5)
    ax.axhline(0, color=INK, lw=.7)
    ax.set_xticks(HARM); ax.set_xticklabels(HARM, fontsize=6.4)
    ax.set_xlabel("Multiple of 5 Hz (Hz)"); ax.set_ylabel("Line index, log₁₀(peak / neighbourhood)")
    ax.set_title("Narrow-band excess at multiples of 5 Hz, stimulation window", fontsize=7.6)
    # d: change stim - pre at 5-Hz harmonics, two arms
    ax = fig.add_subplot(gs[1, 1])
    sub = LC[LC.measure.str.startswith("line_")]
    hh = np.array([int(s.split("_")[1]) for s in sub.measure])
    ax.plot(hh, sub.trt_change, "o-", color=STIM, ms=3.5, lw=1, label="Stimulation")
    ax.plot(hh, sub.sham_change, "s-", color=NOST, ms=3.5, lw=1, label="No stimulation")
    ax.axhline(0, color=INK, lw=.7)
    ax.set_xticks(HARM); ax.set_xticklabels(HARM, fontsize=6.4)
    ax.set_xlabel("Multiple of 5 Hz (Hz)"); ax.set_ylabel("Change in line index, stimulation − pre")
    ax.set_title("Within-recording change at the same frequencies", fontsize=7.6)
    ax.legend(fontsize=6.2, frameon=False)
    for ax_, lt in zip(fig.axes, "abcd"):
        PS.tag(fig, ax_, lt, dx=.055, dy=.025)
    out = FIG / "Supplementary_Figure_7_stimulation_artefact"
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("\nsaved", out.name)


if __name__ == "__main__":
    main()
