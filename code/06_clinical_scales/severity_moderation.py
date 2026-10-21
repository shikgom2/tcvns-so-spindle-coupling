# -*- coding: utf-8 -*-
"""Does baseline symptom severity (PSQI, ISI; first visit) moderate the coupling effect?

 1  Continuous: baseline PSQI/ISI vs per-participant Δcoupling (during − before, mean over stimulation recordings), Spearman
    + baseline PSQI/ISI vs coupling level in the pre-stimulation window
 2  Dichotomous: split by PSQI ≤ 10 / > 10 (median) and ISI < 15 / ≥ 15 (clinical moderate),
    Δcoupling in each group (stimulation, no stimulation) and the between-condition difference; tertiles as well
 3  LMM: prevalence ~ arm × window × severity (continuous, standardised) + (1|participant)
 4  Within each severity group, does Δcoupling predict PSQI/ISI improvement?
Output: SEV_*.csv, Figure_S_severity_moderation.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
import statsmodels.formula.api as smf                             # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from scipy import stats                                           # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402
import paperstyle as PS                                           # noqa: E402
import figures_questionnaire as FQ                                # noqa: E402

R = LP.PROJECT_ROOT
T = R / "Final analysis/tables"
FIG = R / "Final analysis/manuscript/figures"
C = PS.C; STIM, NOST = C["stim"], C["nost"]; RC = PS.ROI
ROIS = ["Frontal", "Temporal", "Central", "Parietal"]
RNG = np.random.default_rng(20261012)


def load():
    inv = pd.read_csv(T / "FIF_index.csv")
    idmap = inv.drop_duplicates("Participant").set_index("Source subject ID").Participant
    q = FQ.recode(FQ.load_q()); q["participant"] = q.ID.map(idmap)
    base = q[q.Order == 1].set_index("participant")[["PSQI", "ISI", "ESS", "BAI", "BDI-2"]]
    last = q.sort_values("Order").groupby("participant").tail(1).set_index("participant")[["PSQI", "ISI"]]
    chg = (last - base[["PSQI", "ISI"]]).rename(columns={"PSQI": "dPSQI", "ISI": "dISI"})
    g = pd.read_csv(R / "이전/outputs/paper/S_Table_session_metrics.csv")
    g = g[g.detector == "staresina_N2"]
    w = g.pivot_table(index=["participant", "session_id", "arm", "ROI"], columns="win",
                      values="prevalence").reset_index()
    w["d"] = (w.stim - w.pre) * 100
    w = w.merge(base, left_on="participant", right_index=True, how="left").merge(
        chg, left_on="participant", right_index=True, how="left")
    return g, w, base, chg


def part1(w):
    rows = []
    for roi in ROIS:
        for scale in ("PSQI", "ISI"):
            for arm in ("Treatment", "Sham"):
                p = (w[(w.ROI == roi) & (w.arm == arm)].groupby("participant")
                     .agg(d=("d", "mean"), pre=("pre", "mean"), sev=(scale, "first")).dropna())
                r1 = stats.spearmanr(p.sev, p.d); r2 = stats.spearmanr(p.sev, p.pre * 100)
                rows.append(dict(ROI=roi, scale=scale, arm=arm, n=len(p),
                                 rho_sev_change=r1[0], p_sev_change=r1[1],
                                 rho_sev_baseline=r2[0], p_sev_baseline=r2[1]))
    S = pd.DataFrame(rows)
    print("\n[1] baseline severity vs Δcoupling (during−before) and vs pre-window coupling, participant level")
    print(S.round(3).to_string(index=False))
    S.to_csv(T / "SEV_1_continuous.csv", index=False)


def part2(w):
    rows = []
    splits = {"PSQI": [("PSQI ≤ 10", lambda s: s <= 10), ("PSQI > 10", lambda s: s > 10)],
              "ISI": [("ISI < 15", lambda s: s < 15), ("ISI ≥ 15", lambda s: s >= 15)]}
    for roi in ROIS:
        for scale, grp in splits.items():
            for lab, fn in grp:
                sub = w[(w.ROI == roi) & fn(w[scale])]
                pa = sub[sub.arm == "Treatment"].groupby("participant").d.mean().dropna()
                pb = sub[sub.arm == "Sham"].groupby("participant").d.mean().dropna()
                pmw = stats.mannwhitneyu(pa, pb).pvalue if len(pa) > 3 and len(pb) > 3 else np.nan
                rows.append(dict(ROI=roi, split=lab, n_stim=len(pa), n_sham=len(pb),
                                 change_stim=pa.mean(), p_stim=stats.wilcoxon(pa).pvalue if len(pa) > 5 else np.nan,
                                 change_sham=pb.mean(), p_sham=stats.wilcoxon(pb).pvalue if len(pb) > 5 else np.nan,
                                 diff=pa.mean() - pb.mean(), p_between=pmw))
    S = pd.DataFrame(rows)
    print("\n[2] Δcoupling by severity group (participant level, pp)")
    print(S.round(3).to_string(index=False))
    S.to_csv(T / "SEV_2_groups.csv", index=False)
    return S


def part3(g, base):
    t = g[g.win.isin(["pre", "stim"])].merge(base, left_on="participant", right_index=True, how="inner").copy()
    t["arm01"] = (t.arm == "Treatment").astype(int); t["w01"] = (t.win == "stim").astype(int)
    rows = []
    for roi in ROIS:
        for scale in ("PSQI", "ISI"):
            s = t[t.ROI == roi].dropna(subset=["prevalence", scale]).copy()
            s["sev"] = (s[scale] - s[scale].mean()) / s[scale].std()
            m = smf.mixedlm("prevalence ~ arm01 * w01 * sev", s, groups=s.participant).fit(reml=False)
            rows.append(dict(ROI=roi, scale=scale, n=s.participant.nunique(),
                             beta_arm_x_win=m.params["arm01:w01"], p_arm_x_win=m.pvalues["arm01:w01"],
                             beta_3way=m.params["arm01:w01:sev"], p_3way=m.pvalues["arm01:w01:sev"],
                             beta_win_x_sev=m.params["w01:sev"], p_win_x_sev=m.pvalues["w01:sev"]))
    S = pd.DataFrame(rows)
    print("\n[3] arm × window × severity (standardised) mixed model")
    print(S.round(4).to_string(index=False))
    S.to_csv(T / "SEV_3_lmm_3way.csv", index=False)


def part4(w):
    rows = []
    for roi in ("Frontal", "Parietal"):
        for scale, cut, lab in (("PSQI", 10, "PSQI > 10"), ("ISI", 15, "ISI ≥ 15")):
            for hi in (False, True):
                sub = w[(w.ROI == roi) & (w.arm == "Treatment") & ((w[scale] > cut) if scale == "PSQI" else (w[scale] >= cut)) == hi] \
                    if False else w[(w.ROI == roi) & (w.arm == "Treatment")]
                sub = sub[(sub[scale] > cut) == hi] if scale == "PSQI" else sub[(sub[scale] >= cut) == hi]
                p = sub.groupby("participant").agg(d=("d", "mean"), dP=("dPSQI", "first"), dI=("dISI", "first")).dropna()
                if len(p) < 6:
                    continue
                r1 = stats.spearmanr(p.d, p.dP); r2 = stats.spearmanr(p.d, p.dI)
                rows.append(dict(ROI=roi, group=(lab if hi else "not " + lab), n=len(p),
                                 rho_dcoupling_dPSQI=r1[0], p1=r1[1], rho_dcoupling_dISI=r2[0], p2=r2[1]))
    S = pd.DataFrame(rows)
    print("\n[4] within severity group: Δcoupling (stim) vs symptom change first→last")
    print(S.round(3).to_string(index=False))
    S.to_csv(T / "SEV_4_brain_behaviour_by_group.csv", index=False)


def figure(w, S2):
    PS.use()
    f, axes = plt.subplots(2, 2, figsize=(7.09, 5.6))
    for j, scale in enumerate(("PSQI", "ISI")):
        ax = axes[0, j]
        for roi, col, dx in (("Parietal", RC["Parietal"], 0), ("Frontal", RC["Frontal"], .25)):
            p = (w[(w.ROI == roi) & (w.arm == "Treatment")].groupby("participant")
                 .agg(d=("d", "mean"), sev=(scale, "first")).dropna())
            ax.scatter(p.sev + dx + RNG.uniform(-.12, .12, len(p)), p.d, s=12, color=col, alpha=.55, lw=0)
            sl, ic, r, pv, _ = stats.linregress(p.sev, p.d)
            xx = np.linspace(p.sev.min(), p.sev.max(), 20)
            ax.plot(xx, ic + sl * xx, color=col, lw=1.6)
            rho, pr = stats.spearmanr(p.sev, p.d)
            ax.text(.02, .97 - (0 if roi == "Parietal" else .09), "%s ρ = %+.2f, P = %.2f" % (roi, rho, pr),
                    transform=ax.transAxes, fontsize=6.5, color=col, va="top")
        ax.axhline(0, color=C["ink"], lw=.7, ls=(0, (3, 2)))
        ax.set_xlabel("Baseline %s" % scale); ax.set_ylabel("Δ coupling, during − before (pp)")
        PS.title(ax, "Stimulation recordings")
        ax = axes[1, j]
        labs = ["PSQI ≤ 10", "PSQI > 10"] if scale == "PSQI" else ["ISI < 15", "ISI ≥ 15"]
        for k, lab in enumerate(labs):
            for roi, dx in (("Frontal", -.3), ("Parietal", .3)):
                r = S2[(S2.ROI == roi) & (S2.split == lab)].iloc[0]
                x = k * 2 + dx
                for arm, col, ddx in (("sham", NOST, -.1), ("stim", STIM, .1)):
                    v = r["change_" + arm]
                    ax.bar(x + ddx, v, width=.18, color=col, alpha=.9 if roi == "Parietal" else .5)
                ax.text(x, max(r.change_stim, r.change_sham) + .6, "P = %.2f" % r.p_between,
                        ha="center", fontsize=6, color=C["ink2"])
                ax.text(x, -8.5, roi[:4], ha="center", fontsize=6, color=C["ink2"])
        ax.axhline(0, color=C["ink"], lw=.7)
        ax.set_xticks([0, 2]); ax.set_xticklabels(labs)
        ax.set_ylim(-9.5, 12); ax.set_ylabel("Δ coupling (pp)")
        PS.title(ax, "By severity group")
    h = [Line2D([], [], color=STIM, lw=6, label="Stimulation"), Line2D([], [], color=NOST, lw=6, label="No stimulation")]
    f.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(.5, -.005))
    for ax, lt in zip(axes.ravel(), "abcd"):
        PS.tag(f, ax, lt); PS.panel(ax); ax.grid(False)
    f.tight_layout(rect=(0, .03, 1, 1))
    f.savefig(FIG / "Figure_S_severity_moderation.png", dpi=300)
    f.savefig(FIG / "Figure_S_severity_moderation.pdf")
    print("saved figure")


if __name__ == "__main__":
    g, w, base, chg = load()
    print("participants with baseline scales:", base.index.notna().sum(), "| matched to EEG:", w.participant.isin(base.index).sum() > 0)
    part1(w); S2 = part2(w); part3(g, base); part4(w); figure(w, S2)
