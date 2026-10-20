# -*- coding: utf-8 -*-
"""Figure 7 (clinical, merged) — a–c are the same plots as figures_questionnaire.fig_scales,
d–e add the stimulation effect by baseline insomnia severity (ISI ≥ 15).

 d  parietal coupling prevalence during − before, ISI < 15 / ≥ 15 × condition (participant means, bootstrap 95 %)
 e  frontal 'widespread slow wave' fraction during − before, same layout
Output: Figure_07_clinical_scales_and_severity.png/pdf, SEV_6_widespread_by_ISI.csv
"""
from __future__ import annotations

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
import paperstyle as PS                                           # noqa: E402
import figures_questionnaire as FQ                                # noqa: E402
import severity_moderation as SM                                  # noqa: E402

R = Path(r"D:/SO-Spindle논문")
T = R / "Final analysis/tables"
FIG = R / "Final analysis/manuscript/figures"
C = PS.C; STIM, NOST = C["stim"], C["nost"]
RNG = np.random.default_rng(20261012)
SCALES, SLAB = FQ.SCALES, FQ.SLAB


def boot_ci(v, n=4000):
    v = np.asarray(v, float); b = RNG.choice(v, (n, v.size), True).mean(1)
    return np.percentile(b, [2.5, 97.5])


def severity_frames():
    g, w, base, chg = SM.load()                      # w: recording × ROI, d = Δprevalence(pp), includes ISI
    inv = pd.read_csv(T / "FIF_index.csv")[["Session ID", "Participant"]]
    inv.columns = ["session", "participant"]
    m2 = pd.read_csv(T / "M2_globality_recording.csv").merge(inv, on="session")
    fw = m2.pivot_table(index=["session", "participant", "arm", "ROI"], columns="win", values="frac_global").reset_index()
    fw["d"] = (fw.stim - fw.pre) * 100
    fw = fw.merge(base[["ISI"]], left_on="participant", right_index=True, how="left")
    return w, fw


def group_panel(ax, df, roi, ylab, title, stats_rows, key):
    """df: recording × ROI with d, arm, participant, ISI. Participant means -> group × condition bootstrap."""
    t = df[df.ROI == roi].dropna(subset=["d", "ISI"])
    x = np.arange(2)
    res = {}
    for k, (lab, mask) in enumerate((("ISI < 15", t.ISI < 15), ("ISI ≥ 15", t.ISI >= 15))):
        for arm, col, dx in (("Sham", NOST, -.18), ("Treatment", STIM, .18)):
            v = t[mask & (t.arm == arm)].groupby("participant").d.mean().dropna().values
            res[(lab, arm)] = v
            lo, hi = boot_ci(v)
            ax.plot([x[k] + dx] * 2, [lo, hi], color=col, lw=1.8, solid_capstyle="round", zorder=3)
            PS.dot(ax, [x[k] + dx], [v.mean()], col, s=34, zorder=5)
            ax.scatter(x[k] + dx + RNG.uniform(-.06, .06, v.size), v, s=5, color=col, alpha=.3, lw=0, zorder=2)
        a, b = res[(lab, "Treatment")], res[(lab, "Sham")]
        p = stats.mannwhitneyu(a, b).pvalue
        stats_rows.append(dict(measure=key, ROI=roi, group=lab, n_stim=a.size, n_sham=b.size,
                               change_stim=a.mean(), change_sham=b.mean(), diff=a.mean() - b.mean(), p_between=p))
        ax.text(x[k], 1.0, "P = %.3f" % p, ha="center", va="top", fontsize=6.5, color=C["ink2"],
                transform=ax.get_xaxis_transform())
        ax.text(x[k], .02, "n = %d / %d" % (b.size, a.size), ha="center", va="bottom", fontsize=5.8,
                color=C["ink2"], transform=ax.get_xaxis_transform())
    ax.axhline(0, color=C["ink"], lw=.8, ls=(0, (3, 2)))
    ax.set_xticks(x); ax.set_xticklabels(["ISI < 15", "ISI ≥ 15"]); ax.set_xlim(-.6, 1.6)
    ax.set_ylabel(ylab, linespacing=1.3)
    lo_, hi_ = np.percentile(t.d, [2, 100]); ax.set_ylim(lo_ - 4, hi_ + 10)
    ax.text(.98, .97, "1 point below axis", ha="right", va="top", fontsize=5.6, color=C["ink2"], transform=ax.transAxes)
    PS.title(ax, title)


def main():
    PS.use()
    d = FQ.recode(FQ.load_q()); res = pd.read_csv(T / "Q5_change_primary.csv")
    SC = {"PSQI": PS.C["stim"], "ISI": "#2E8F8F", "ESS": "#3F7A4A", "BAI": PS.C["nost"], "BDI-2": "#6B4E8F"}
    WORSE, INK, INK2 = "#9AA2A8", PS.C["ink"], PS.C["ink2"]

    f = plt.figure(figsize=(7.09, 10.4))
    gs = f.add_gridspec(3, 1, height_ratios=[1, 1.22, 1.05], hspace=.55, left=.082, right=.975, top=.965, bottom=.055)
    g0 = gs[0].subgridspec(1, 5, wspace=.62)
    g1 = gs[1].subgridspec(1, 2, width_ratios=[1.52, 1], wspace=.34)
    g2 = gs[2].subgridspec(1, 3, width_ratios=[1.15, 1.25, .9], wspace=.55)

    # ── a (as in figures_questionnaire)
    axs0 = []
    for i, s in enumerate(SCALES):
        ax = f.add_subplot(g0[i]); axs0.append(ax); c = SC[s]
        t = FQ.first_last(d, s); imp = (t["last"] < t["first"]).values
        for (a_, b_), ok in zip(t[["first", "last"]].values, imp):
            ax.plot([0, 1], [a_, b_], color=c if ok else WORSE, alpha=.17 if ok else .22, lw=.6, zorder=1)
        for k, col in enumerate(("first", "last")):
            v = t[col].values; lo, hi = FQ._boot_ci(v)
            ax.plot([k, k], [lo, hi], color=INK, lw=1.4, zorder=4, solid_capstyle="round")
            PS.dot(ax, [k], [v.mean()], c, s=30, zorder=6)
        PS.line(ax, [0, 1], [t["first"].mean(), t["last"].mean()], c, lw=2.0, zorder=4)
        ax.set_xlim(-.42, 1.42); ax.set_xticks([0, 1]); ax.set_xticklabels(["First", "Last"], fontsize=6.8)
        ax.set_title(SLAB[s], fontsize=8.4, pad=5, color=c, fontweight="bold"); ax.tick_params(labelsize=6.8)
        if i == 0:
            ax.set_ylabel("Score", fontsize=7.6)
    # ── b
    axb = f.add_subplot(g1[0]); y = np.arange(len(SCALES))[::-1]; xmax = 0.0
    for i, s in enumerate(SCALES):
        v = FQ.change_per_id(d, s).values; yy, c = y[i], SC[s]
        FQ._half_violin(axb, v, yy + .06, width=.40, color=c, alpha=.26)
        axb.scatter(v, yy - .17 + FQ.RNG.uniform(-.075, .075, v.size), s=5.5,
                    color=[c if x < 0 else WORSE for x in v], alpha=.48, lw=0, zorder=4)
        lo, hi = FQ._boot_ci(v)
        axb.plot([lo, hi], [yy - .17] * 2, color=INK, lw=1.6, zorder=6, solid_capstyle="round")
        PS.dot(axb, [v.mean()], [yy - .17], c, s=28, zorder=7, halo=False); xmax = max(xmax, abs(v).max())
    axb.axvline(0, color=INK, lw=.9, ls=(0, (3, 2)), zorder=1)
    axb.set_yticks(y[::-1]); axb.set_yticklabels([SLAB[s] for s in SCALES[::-1]], fontsize=8)
    for tick, s in zip(axb.get_yticklabels(), SCALES[::-1]):
        tick.set_color(SC[s]); tick.set_fontweight("bold")
    axb.set_ylim(-1.0, len(SCALES) - .25); axb.set_xlim(-xmax * 1.08, xmax * 1.08)
    axb.set_xlabel("Change in score, first to last visit", fontsize=7.6)
    axb.annotate("improvement", xy=(-xmax * .68, -.80), xytext=(-xmax * .16, -.80), fontsize=6.6, color=INK2,
                 va="center", ha="right", arrowprops=dict(arrowstyle="->", lw=.8, color=INK2))
    # ── c
    axc = f.add_subplot(g1[1])
    for i, s in enumerate(SCALES):
        r = res[res.scale == s].iloc[0]; v = FQ.change_per_id(d, s).values
        bs = FQ.RNG.choice(v, (4000, v.size), True); dzb = bs.mean(1) / bs.std(1, ddof=1)
        lo, hi = np.percentile(dzb, [2.5, 97.5]); _, wt, _ = FQ.style(r.p, r.q); col = SC[s]
        PS.grad_ci(axc, lo, hi, y[i], col, center=r.dz, h=.30); PS.dot(axc, [r.dz], [y[i]], col, s=38, zorder=6)
        axc.text(.72, y[i] + .055, "$d_z$ = %.2f" % r.dz, fontsize=7.0, va="bottom", ha="right", color=col, fontweight=wt)
        axc.text(.72, y[i] - .075, "%d%% improved" % round(100 * r.n_improved / r.n), fontsize=6.2, va="top", ha="right", color=INK2)
    axc.axvline(0, color=INK, lw=.9, ls=(0, (3, 2)), zorder=1)
    for ref in (-.2, -.5, -.8):
        axc.axvline(ref, color="#E3E0DA", lw=.7, zorder=0)
    axc.set_yticks(y[::-1]); axc.set_yticklabels([]); axc.set_ylim(-1.0, len(SCALES) - .05)
    axc.set_xlim(-1.45, .76); axc.set_xticks([-1.2, -.8, -.4, 0]); axc.tick_params(labelsize=6.8)
    axc.set_xlabel("Standardized change $d_z$", fontsize=7.6)
    for ref, lab, dy in ((-.2, "small", 0), (-.5, "medium", .17), (-.8, "large", 0)):
        axc.text(ref, len(SCALES) - .36 + dy, lab, fontsize=5.8, color=INK2, ha="center", va="bottom")
    # ── d, e
    w, fw = severity_frames()
    rows = []
    axd = f.add_subplot(g2[0])
    # d: baseline-normalised Before/During/After, stimulation sessions, parietal, by ISI group
    tA = w[(w.ROI == "Parietal")].dropna(subset=["pre", "ISI"]).copy()
    tA["d_stim"] = (tA.stim - tA.pre) * 100; tA["d_post"] = (tA.post - tA.pre) * 100
    tP = tA[tA.arm == "Treatment"]; tS = tA[tA.arm == "Sham"]
    for lab, mask, dx in (("ISI < 15", tS.ISI < 15, -.05), ("ISI ≥ 15", tS.ISI >= 15, .05)):
        g_ = tS[mask].groupby("participant")[["d_stim", "d_post"]].mean()
        m = [0, g_.d_stim.mean(), g_.d_post.mean()]
        axd.plot(np.array([0, 1, 2]) + dx, m, color=NOST, lw=1.4, alpha=.8 if lab == "ISI ≥ 15" else .45,
                 ls=(0, (4, 2)), zorder=2, marker="o", ms=3.2, mec="white", mew=.5)
    axd.axvspan(.5, 1.5, color=STIM, alpha=.07, lw=0, zorder=0)
    axd.axhline(0, color=C["ink2"], lw=.7, ls=(0, (3, 2)), zorder=1)
    cols = {"ISI ≥ 15": PS.ROI["Parietal"], "ISI < 15": C["ns"]}
    for lab, mask, dx in (("ISI < 15", tP.ISI < 15, -.05), ("ISI ≥ 15", tP.ISI >= 15, .05)):
        g_ = tP[mask].groupby("participant")[["d_stim", "d_post"]].mean()
        m = [0, g_.d_stim.mean(), g_.d_post.mean()]
        e = [0, g_.d_stim.std(ddof=1) / np.sqrt(g_.d_stim.notna().sum()), g_.d_post.std(ddof=1) / np.sqrt(g_.d_post.notna().sum())]
        xx = np.array([0, 1, 2]) + dx
        axd.errorbar(xx, m, yerr=e, fmt="none", ecolor=cols[lab], elinewidth=1.3, capsize=3, zorder=3)
        PS.line(axd, xx, m, cols[lab], lw=2.2 if lab == "ISI ≥ 15" else 1.8, zorder=4)
        PS.dot(axd, xx, m, cols[lab], s=34, zorder=6)
        axd.text(2.32, m[2] + (0.4 if lab == "ISI ≥ 15" else -1.6), "n = %d" % len(g_), va="center", fontsize=6.0, color=cols[lab] if lab == "ISI ≥ 15" else C["ink2"], linespacing=1.2)
    axd.set_xticks([0, 1, 2]); axd.set_xticklabels(["Before", "During", "After"]); axd.set_xlim(-.3, 2.95); axd.set_ylim(-4.5, 16.5)
    axd.set_ylabel("Parietal coupling,\nchange from own baseline (pp)", linespacing=1.3)
    hd = [Line2D([], [], color=PS.ROI["Parietal"], lw=2.2, marker="o", ms=4, label="Stim., ISI ≥ 15"),
          Line2D([], [], color=C["ns"], lw=1.8, marker="o", ms=4, label="Stim., ISI < 15"),
          Line2D([], [], color=NOST, lw=1.4, ls=(0, (4, 2)), marker="o", ms=3, alpha=.8, label="No stim., ISI ≥ 15"),
          Line2D([], [], color=NOST, lw=1.4, ls=(0, (4, 2)), marker="o", ms=3, alpha=.45, label="No stim., ISI < 15")]
    axd.legend(handles=hd, loc="upper left", fontsize=5.2, frameon=False, borderpad=.1, handlelength=1.6, labelspacing=.25, ncol=2, columnspacing=.8)
    PS.title(axd, "By severity and condition")
    axe = f.add_subplot(g2[1])
    # e: baseline-adjusted condition effect on during-window coupling, by ISI group, parietal & frontal
    erows = []; yy = 0
    ylabels = []
    for roi in ("Parietal", "Frontal"):
        for lab, mask in (("ISI ≥ 15", w.ISI >= 15), ("ISI < 15", w.ISI < 15)):
            t = w[(w.ROI == roi) & mask].dropna(subset=["pre", "stim"]).copy()
            t["arm01"] = (t.arm == "Treatment").astype(int); t["s100"] = t.stim * 100; t["p100"] = t.pre * 100
            m = smf.mixedlm("s100 ~ arm01 + p100", t, groups=t.participant).fit(reml=False)
            b_, se = m.params["arm01"], m.bse["arm01"]; pv = m.pvalues["arm01"]
            col = PS.ROI[roi]; sig = pv < .05
            axe.plot([b_ - 1.96 * se, b_ + 1.96 * se], [yy, yy], color=col if sig else C["ns"], lw=1.9, solid_capstyle="round")
            PS.dot(axe, [b_], [yy], col if sig else C["ns"], s=36, zorder=6)
            axe.text(b_ + 1.96 * se + .8, yy, "%+.1f, P = %.3f" % (b_, pv), va="center", fontsize=6.0,
                     color=col if sig else C["ink2"], fontweight="bold" if sig else "normal")
            ylabels.append("%s\n%s" % (roi, lab)); erows.append(dict(ROI=roi, group=lab, n_part=t.participant.nunique(), beta_pp=b_, se=se, p=pv))
            yy += 1
    axe.axvline(0, color=C["ink"], lw=.8, ls=(0, (3, 2)))
    axe.set_yticks(range(len(ylabels))); axe.set_yticklabels(ylabels, fontsize=6.6, linespacing=1.15)
    axe.set_ylim(-.6, len(ylabels) - .4); axe.set_xlim(-8, 46); axe.set_xticks([0, 10, 20, 30])
    axe.set_xlabel("Condition effect on during-window coupling,\nadjusted for baseline (pp)", linespacing=1.3)
    PS.title(axe, "Baseline-adjusted effect by group")
    pd.DataFrame(erows).to_csv(T / "SEV_7_fig7_adjusted.csv", index=False); print(pd.DataFrame(erows).round(3).to_string(index=False))
    axf = f.add_subplot(g2[2])
    pp_ = w[w.ROI == "Parietal"].dropna(subset=["d", "ISI"])
    sham_sd = pp_[pp_.arm == "Sham"].groupby("participant").d.mean().std(ddof=1)
    frows = []; xt = []; xl = []
    pos = {("ISI < 15", "Sham"): 0, ("ISI < 15", "Treatment"): 1.3, ("ISI ≥ 15", "Sham"): 3.1, ("ISI ≥ 15", "Treatment"): 4.4}
    for lab, mask in (("ISI < 15", pp_.ISI < 15), ("ISI ≥ 15", pp_.ISI >= 15)):
        for arm, col in (("Sham", NOST), ("Treatment", STIM)):
            v = pp_[mask & (pp_.arm == arm)].groupby("participant").d.mean().dropna().values
            r_ = int((v > sham_sd).sum()); n_ = len(v); p_ = r_ / n_; x0 = pos[(lab, arm)]
            z = 1.96; den = 1 + z**2 / n_; cen = (p_ + z**2 / (2 * n_)) / den; hw = z * np.sqrt(p_ * (1 - p_) / n_ + z**2 / (4 * n_**2)) / den
            resp = v > sham_sd
            axf.scatter(x0 + RNG.uniform(-.22, .22, (~resp).sum()), v[~resp], s=9, facecolor="white", edgecolor=col, lw=.7, alpha=.8, zorder=3)
            axf.scatter(x0 + RNG.uniform(-.22, .22, resp.sum()), v[resp], s=11, color=col, alpha=.9, lw=0, zorder=4)
            lo, hi = boot_ci(v)
            axf.plot([x0 - .3, x0 + .3], [v.mean()] * 2, color=C["ink"], lw=1.4, zorder=6)
            axf.plot([x0 - .3, x0 - .3], [lo, hi], color=C["ink"], lw=.8, zorder=6, alpha=.6); axf.plot([x0 + .3, x0 + .3], [lo, hi], color=C["ink"], lw=.8, zorder=6, alpha=.6)
            xt.append(x0); xl.append(("%s" + chr(10) + "%d/%d") % ("No stim." if arm == "Sham" else "Stim.", r_, n_))
            frows.append(dict(group=lab, arm=arm, responders=r_, n=n_, rate=p_, ci_lo=cen - hw, ci_hi=cen + hw, threshold_pp=sham_sd))
    ft = pd.DataFrame(frows)
    a1 = ft[(ft.arm == "Treatment") & (ft.group == "ISI ≥ 15")].iloc[0]; a0 = ft[(ft.arm == "Treatment") & (ft.group == "ISI < 15")].iloc[0]
    pf = stats.fisher_exact([[a1.responders, a1.n - a1.responders], [a0.responders, a0.n - a0.responders]])[1]
    axf.axhline(sham_sd, color=C["ink2"], lw=.8, ls=(0, (3, 2)), zorder=1)
    axf.axhline(0, color=C["ink"], lw=.6, zorder=1)
    allv = pp_.groupby(["participant", "arm"]).d.mean(); lo_, hi_ = np.percentile(allv, [2, 100])
    axf.set_ylim(lo_ - 6, hi_ + 9); axf.set_xlim(-.8, 5.2)
    pass
    axf.set_xticks(xt); axf.set_xticklabels(xl, fontsize=5.6, linespacing=1.15)
    for x0, lab in ((0.65, "ISI < 15"), (3.75, "ISI ≥ 15")):
        axf.text(x0, .99, lab, transform=axf.get_xaxis_transform(), ha="center", va="top", fontsize=6.6, fontweight="bold", color=C["ink"])
    axf.set_ylabel("Parietal coupling change per participant" + chr(10) + "(during − before, pp)", linespacing=1.3)
    PS.title(axf, "Individual responses")
    ft.to_csv(T / "SEV_8_responders.csv", index=False); print(ft.round(3).to_string(index=False), "Fisher P", round(pf, 3))
    pd.DataFrame(rows).to_csv(T / "SEV_6_fig7_groups.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    for ax, lt in zip([axs0[0], axb, axc, axd, axe, axf], "abcdef"):
        p = ax.get_position()
        f.text(max(p.x0 - .058, .002), p.y1 + .022, lt, fontsize=10.5, fontweight="bold", va="bottom", ha="left", color=INK)
    for a_ in axs0 + [axb, axc, axd, axe, axf]:
        PS.panel(a_); a_.grid(False); a_.set_axisbelow(True)
    out = FIG / "Figure_07_clinical_scales_and_severity_v3"
    f.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight"); f.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
