# -*- coding: utf-8 -*-
"""Questionnaire figures + brain-behaviour correlations — 4 scales (ESS excluded), 75 participants, no values removed.

Display rules as for the other results: q<.05 bold+★, passing only before correction ☆,
otherwise grey, but the value is still shown.
"""
from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402

warnings.filterwarnings("ignore")
_R = Path(r"D:/SO-Spindle논문")
P = _R if (_R / "outputs/paper").exists() else _R / "이전"
OUT = P / "outputs/paper"
T = _R / "Final analysis/tables"
FIG = _R / "Final analysis/figures"
FIG.mkdir(parents=True, exist_ok=True)
XLSB = Path(r"D:/HJ/vns/rawfiles_fromJY/DATA/Data/Data/tVNS_Sleep.xlsb")
IITF = Path(r"D:/HJ/vns/rawfiles_fromJY/DATA/Data/iit_delivery/Data"
            "/IIT_Assessments.xlsx")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.linewidth": .7, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.major.width": .7,
    "ytick.major.width": .7, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "legend.frameon": False,
    "savefig.dpi": 400, "figure.dpi": 140,
})
import paperstyle as PS
SIG, NOM, NS = PS.C["stim"], "#7FA8BC", PS.C["ns"]
POSC, NEGC = "#2E7257", "#9A3B3B"
SCALES = ["PSQI", "ISI", "ESS", "BAI", "BDI-2"]
SLAB = {"PSQI": "PSQI", "ISI": "ISI", "ESS": "ESS", "BAI": "BAI",
        "BDI-2": "BDI-II"}
RANGE = {"PSQI": (0, 21), "ISI": (0, 28), "ESS": (0, 24),
         "BAI": (0, 63), "BDI-2": (0, 63)}
ROIS = ["Frontal", "Temporal", "Central", "Parietal"]
DETS = ["staresina_N2"]
DL = {"jy_N2": "Reference · N2", "staresina_N2": "Staresina · N2"}
# Participants excluded from the questionnaire analyses (no analysable EEG).
# The ID list is kept outside version control: config/excluded_participants.txt
# (one ID per line; see config/excluded_participants.example.txt).
DROP = [ln.strip() for ln in (Path(__file__).resolve().parents[2] / "config" / "excluded_participants.txt")
        .read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
RNG = np.random.default_rng(20261003)


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


def style(p, q):
    if np.isfinite(q) and q < .05:
        return SIG, "bold", "★"
    if np.isfinite(p) and p < .05:
        return NOM, "normal", "☆"
    return NS, "normal", ""


def load_q():
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
    d = pd.concat([a[["ID", "cohort", "Order"] + SCALES],
                   b[["ID", "cohort", "Order"] + SCALES]], ignore_index=True)
    for s in SCALES:
        d[s] = pd.to_numeric(d[s], errors="coerce")
    return d[~d.ID.isin(DROP)].copy()


def recode(d):
    """Primary-analysis handling — values outside the scale range (= the ESS -800) set to missing."""
    out = d.copy()
    for s, (lo, hi) in RANGE.items():
        v = out[s]
        out[s] = v.mask(v.notna() & ((v < lo) | (v > hi)))
    return out


def change_per_id(d, s):
    w = d.dropna(subset=[s]).sort_values("Order")
    n = w.groupby("ID").Order.nunique()
    keep = n[n >= 2].index
    return (w.groupby("ID")[s].last()[keep]
            - w.groupby("ID")[s].first()[keep]).dropna()


def first_last(d, s):
    """First-visit and last-visit scores of participants measured at least twice on scale s."""
    w = d.dropna(subset=[s]).sort_values("Order")
    n = w.groupby("ID").Order.nunique()
    keep = n[n >= 2].index
    return pd.DataFrame({"first": w.groupby("ID")[s].first()[keep],
                         "last": w.groupby("ID")[s].last()[keep]}).dropna()


def _boot_ci(v, fn=np.mean, n=4000):
    b = fn(RNG.choice(v, (n, v.size), True), axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _half_violin(ax, v, y0, width=.38, color="#1D6B8C", alpha=.30):
    """Violin drawn as the upper half only (top of the raincloud)."""
    if v.size < 3 or np.ptp(v) == 0:
        return
    k = stats.gaussian_kde(v)
    xs = np.linspace(v.min() - np.ptp(v) * .08,
                     v.max() + np.ptp(v) * .08, 256)
    dn = k(xs)
    dn = dn / dn.max() * width
    ax.fill_between(xs, y0, y0 + dn, color=color, alpha=alpha, lw=0, zorder=2)
    ax.plot(xs, y0 + dn, color=color, lw=.8, alpha=.75, zorder=3)


def fig_scales(d, res):
    """Three-part layout in the style of Nature Communications.

    a  individual first-visit → last-visit trajectories per scale (five small panels)
    b  raincloud of individual change — half-violin + points + mean with 95 % CI
    c  standardized effect size d_z on a common axis — direct comparison across scales
    """
    # One colour per scale — ordered from sleep (teal family) to mood/anxiety (purple/magenta)
    PS.use()
    SC = {"PSQI": PS.C["stim"], "ISI": "#2E8F8F", "ESS": "#3F7A4A",
          "BAI": PS.C["nost"], "BDI-2": "#6B4E8F"}
    WORSE, INK, INK2 = "#9AA2A8", PS.C["ink"], PS.C["ink2"]

    f = plt.figure(figsize=(7.2, 6.25))
    gs = f.add_gridspec(2, 1, height_ratios=[1, 1.22], hspace=.54,
                        left=.082, right=.975, top=.930, bottom=.095)
    g0 = gs[0].subgridspec(1, 5, wspace=.62)
    g1 = gs[1].subgridspec(1, 2, width_ratios=[1.52, 1], wspace=.34)

    # ---------------------------------------------------------------- a
    axs0 = []
    for i, s in enumerate(SCALES):
        ax = f.add_subplot(g0[i])
        axs0.append(ax)
        c = SC[s]
        t = first_last(d, s)
        imp = (t["last"] < t["first"]).values
        for (a_, b_), ok in zip(t[["first", "last"]].values, imp):
            ax.plot([0, 1], [a_, b_], color=c if ok else WORSE,
                    alpha=.17 if ok else .22, lw=.6, zorder=1)
        for k, col in enumerate(("first", "last")):
            v = t[col].values
            lo, hi = _boot_ci(v)
            ax.plot([k, k], [lo, hi], color=INK, lw=1.4, zorder=4,
                    solid_capstyle="round")
            PS.dot(ax, [k], [v.mean()], c, s=30, zorder=6)
        PS.line(ax, [0, 1], [t["first"].mean(), t["last"].mean()], c,
                lw=2.0, zorder=4)
        ax.set_xlim(-.42, 1.42)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["First", "Last"], fontsize=6.8)
        ax.set_title(SLAB[s], fontsize=8.4, pad=5, color=c,
                     fontweight="bold")
        ax.tick_params(labelsize=6.8)
        if i == 0:
            ax.set_ylabel("Score", fontsize=7.6)

    # ---------------------------------------------------------------- b
    axb = f.add_subplot(g1[0])
    y = np.arange(len(SCALES))[::-1]
    xmax = 0.0
    for i, s in enumerate(SCALES):
        v = change_per_id(d, s).values
        yy, c = y[i], SC[s]
        _half_violin(axb, v, yy + .06, width=.40, color=c, alpha=.26)
        axb.scatter(v, yy - .17 + RNG.uniform(-.075, .075, v.size), s=5.5,
                    color=[c if x < 0 else WORSE for x in v], alpha=.48,
                    lw=0, zorder=4)
        lo, hi = _boot_ci(v)
        axb.plot([lo, hi], [yy - .17] * 2, color=INK, lw=1.6, zorder=6,
                 solid_capstyle="round")
        PS.dot(axb, [v.mean()], [yy - .17], c, s=28, zorder=7, halo=False)
        xmax = max(xmax, abs(v).max())
    axb.axvline(0, color=INK, lw=.9, ls=(0, (3, 2)), zorder=1)
    axb.set_yticks(y[::-1])
    axb.set_yticklabels([SLAB[s] for s in SCALES[::-1]], fontsize=8)
    for tick, s in zip(axb.get_yticklabels(), SCALES[::-1]):
        tick.set_color(SC[s]); tick.set_fontweight("bold")
    axb.set_ylim(-1.00, len(SCALES) - .25)
    axb.set_xlim(-xmax * 1.08, xmax * 1.08)
    axb.set_xlabel("Change in score, first to last visit", fontsize=7.6)
    axb.annotate("improvement", xy=(-xmax * .68, -.80),
                 xytext=(-xmax * .16, -.80), fontsize=6.6, color=INK2,
                 va="center", ha="right",
                 arrowprops=dict(arrowstyle="->", lw=.8, color=INK2))

    # ---------------------------------------------------------------- c
    axc = f.add_subplot(g1[1])
    for i, s in enumerate(SCALES):
        r = res[res.scale == s].iloc[0]
        v = change_per_id(d, s).values
        bs = RNG.choice(v, (4000, v.size), True)
        dzb = bs.mean(1) / bs.std(1, ddof=1)
        lo, hi = np.percentile(dzb, [2.5, 97.5])
        _, wt, _ = style(r.p, r.q)
        col = SC[s]
        PS.grad_ci(axc, lo, hi, y[i], col, center=r.dz, h=.30)
        PS.dot(axc, [r.dz], [y[i]], col, s=38, zorder=6)
        axc.text(.72, y[i] + .055, "$d_z$ = %.2f" % r.dz, fontsize=7.0,
                 va="bottom", ha="right", color=col, fontweight=wt)
        axc.text(.72, y[i] - .075, "%d%% improved"
                 % round(100 * r.n_improved / r.n), fontsize=6.2,
                 va="top", ha="right", color=INK2)
    axc.axvline(0, color=INK, lw=.9, ls=(0, (3, 2)), zorder=1)
    for ref in (-.2, -.5, -.8):
        axc.axvline(ref, color="#E3E0DA", lw=.7, zorder=0)
    axc.set_yticks(y[::-1]); axc.set_yticklabels([])
    axc.set_ylim(-1.00, len(SCALES) - .05)
    axc.set_xlim(-1.45, .76)
    axc.set_xticks([-1.2, -.8, -.4, 0])
    axc.tick_params(labelsize=6.8)
    axc.set_xlabel("Standardized change $d_z$", fontsize=7.6)
    for ref, lab, dy in ((-.2, "small", 0), (-.5, "medium", .17),
                         (-.8, "large", 0)):
        axc.text(ref, len(SCALES) - .36 + dy, lab, fontsize=5.8, color=INK2,
                 ha="center", va="bottom")

    for ax, lt in zip([axs0[0], axb, axc], "abc"):
        p = ax.get_position()
        f.text(max(p.x0 - .058, .002), p.y1 + .028, lt, fontsize=10.5,
               fontweight="bold", va="bottom", ha="left", color=INK)
    for a_ in axs0 + [axb, axc]:
        PS.panel(a_); a_.grid(False); a_.set_axisbelow(True)
    f.savefig(FIG / "FigQ_scales.png", bbox_inches="tight")
    f.savefig(FIG / "FigQ_scales.pdf", bbox_inches="tight")
    plt.close(f)
    print("FigQ_scales  (redrawn)")


def brain_behaviour(d):
    g = pd.read_csv(OUT / "S_Table_session_metrics.csv")
    idm = pd.read_csv(OUT / "INTERNAL_session_idmap.csv")
    g = g.merge(idm[["session_id", "ID", "Order"]], on="session_id",
                how="left")
    g["ID"] = g.ID.astype(str)
    rows = []
    for det in DETS:
        for roi in ROIS:
            t = g[(g.detector == det) & (g.ROI == roi) & (g.win == "stim")]
            t = t.dropna(subset=["prevalence"]).sort_values("Order")
            n = t.groupby("ID").Order.nunique()
            keep = n[n >= 2].index
            dc = (t.groupby("ID").prevalence.last()[keep]
                  - t.groupby("ID").prevalence.first()[keep]).dropna()
            for s in SCALES:
                dq = change_per_id(d, s)
                j = dc.index.intersection(dq.index)
                if len(j) < 10:
                    continue
                r, p = stats.spearmanr(dc[j], dq[j])
                rows.append(dict(detector=det, ROI=roi, scale=s, n=len(j),
                                 rho=float(r), p=float(p)))
    B = pd.DataFrame(rows)
    B["q"] = B.groupby(["detector", "scale"]).p.transform(
        lambda p: fdr(p.values))
    B.to_csv(T / "Q4_brain_behaviour.csv", index=False, encoding="utf-8-sig")

    f, axs = plt.subplots(1, len(DETS), figsize=(3.9 * len(DETS) + .3, 2.6),
                          gridspec_kw=dict(wspace=.34, left=.16, right=.985,
                                           top=.84, bottom=.22))
    axs = np.atleast_1d(axs)
    for ax, det, lt in zip(axs, DETS, "ab"):
        if len(DETS) > 1:
            ax.text(-.22, 1.08, lt, transform=ax.transAxes, fontsize=10,
                    fontweight="bold")
        t = B[B.detector == det]
        M = t.pivot(index="scale", columns="ROI", values="rho").reindex(
            index=SCALES, columns=ROIS)
        Q = t.pivot(index="scale", columns="ROI", values="q").reindex(
            index=SCALES, columns=ROIS)
        Pv = t.pivot(index="scale", columns="ROI", values="p").reindex(
            index=SCALES, columns=ROIS)
        im = ax.imshow(M.values, cmap="RdBu_r", vmin=-.45, vmax=.45,
                       aspect="auto")
        for i in range(len(SCALES)):
            for j in range(len(ROIS)):
                v, q, p = M.values[i, j], Q.values[i, j], Pv.values[i, j]
                if not np.isfinite(v):
                    continue
                _, w, star = style(p, q)
                ax.text(j, i, "%+.2f%s" % (v, "\n" + star if star else ""),
                        ha="center", va="center", fontsize=6.6,
                        fontweight=w,
                        color="white" if abs(v) > .28 else "#1A1D26")
        ax.set_xticks(range(len(ROIS)))
        ax.set_xticklabels(ROIS, fontsize=7, rotation=18, ha="right")
        ax.set_yticks(range(len(SCALES)))
        ax.set_yticklabels([SLAB[s] for s in SCALES], fontsize=7.5)
        ax.set_title(DL[det], fontsize=8.5, pad=4)
        ax.set_xticks(np.arange(-.5, len(ROIS)), minor=True)
        ax.set_yticks(np.arange(-.5, len(SCALES)), minor=True)
        ax.grid(which="minor", color="white", lw=1.4)
        ax.tick_params(which="minor", length=0)
    cb = f.colorbar(im, ax=axs, fraction=.022, pad=.015)
    cb.set_label("Spearman ρ", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    f.text(.105, .025, "Change in coupling prevalence (stimulation window) "
           "versus change in scale · all cells shown · ★ q < .05  ☆ P < .05",
           fontsize=6.4, color="#6B7378")
    f.savefig(FIG / "FigQ_brain_behaviour.png", bbox_inches="tight")
    f.savefig(FIG / "FigQ_brain_behaviour.pdf", bbox_inches="tight")
    plt.close(f)
    print("FigQ_brain_behaviour   cells=%d  q<.05=%d  nominal=%d"
          % (len(B), int((B.q < .05).sum()),
             int(((B.p < .05) & ~(B.q < .05)).sum())))
    return B


def fig_ess_handling(raw):
    """Supplementary figure — what the ESS -800 is and how it was handled."""
    V = {t: pd.read_csv(T / ("Q5_change_%s.csv" % t))
         for t in ("raw", "primary", "mad")}
    VL = [("raw", "No values altered", "#B4651A"),
          ("primary", "−800 recoded as missing\n(primary analysis)", SIG),
          ("mad", "Plus median ± 3 MAD removal\n(sensitivity)", "#6B8E6B")]

    f = plt.figure(figsize=(7.2, 4.5))
    gs = f.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1],
                        wspace=.34, hspace=.66, left=.095, right=.975,
                        top=.855, bottom=.185)

    # a — raw-value distribution: shows that -800 is a single code, not the tail of a continuous distribution
    ax = f.add_subplot(gs[0, 0])
    ax.text(-.19, 1.10, "a", transform=ax.transAxes, fontsize=10,
            fontweight="bold")
    v = raw["ESS"].dropna().values
    bad, good = v[v < 0], v[v >= 0]
    ax.scatter(good, RNG.uniform(-.28, .28, good.size), s=9, color="#4F6472",
               alpha=.45, lw=0)
    ax.scatter(bad, RNG.uniform(-.28, .28, bad.size), s=16, color=NEGC,
               alpha=.8, lw=0)
    ax.axvspan(0, 24, color=SIG, alpha=.07, lw=0)
    ax.annotate("−800 × %d\nsingle value, no spread" % bad.size,
                xy=(-800, .34), xytext=(-660, .78), fontsize=6.8,
                color=NEGC, ha="left",
                arrowprops=dict(arrowstyle="-", lw=.6, color=NEGC))
    ax.text(12, -.74, "valid range 0–24", fontsize=6.8, ha="center",
            color=SIG)
    ax.set_yticks([]); ax.set_ylim(-1.0, 1.1)
    ax.set_xlabel("ESS value as stored")
    ax.set_title("Every ESS value in the dataset", fontsize=8.5, pad=4)
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)

    # b — where the -800 entries come from
    ax = f.add_subplot(gs[0, 1])
    ax.text(-.17, 1.10, "b", transform=ax.transAxes, fontsize=10,
            fontweight="bold")
    oor = pd.read_csv(T / "Q5_out_of_range_values.csv")
    cats = ["Cohort A\nvisit 1", "Cohort A\nlater visits",
            "Cohort B\nany visit"]
    vals = [int((oor.Order == 1).sum()), int((oor.Order != 1).sum()),
            int((oor.cohort == "Cohort B").sum())]
    ax.bar(range(3), vals, width=.58, color=[NEGC, "#D9D3CC", "#D9D3CC"],
           lw=0)
    for i, vv in enumerate(vals):
        ax.text(i, vv + .6, str(vv), ha="center", fontsize=8,
                fontweight="bold" if vv else "normal",
                color=NEGC if vv else "#9AA2A8")
    ax.set_xticks(range(3)); ax.set_xticklabels(cats, fontsize=6.8)
    ax.set_ylabel("Entries coded −800")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_title("All at the first visit, one cohort", fontsize=8.5, pad=4)

    # c — ESS estimates, three versions
    ax = f.add_subplot(gs[1, 0])
    ax.text(-.19, 1.10, "c", transform=ax.transAxes, fontsize=10,
            fontweight="bold")
    y = np.arange(len(VL))[::-1]
    for i, (t, lab, col) in enumerate(VL):
        r = V[t][V[t].scale == "ESS"].iloc[0]
        ax.plot([r.ci_lo, r.ci_hi], [y[i], y[i]], color=col, lw=2.0,
                solid_capstyle="round")
        ax.scatter([r.change], [y[i]], s=40, color=col, lw=.7,
                   edgecolor="white", zorder=3)
        _, w, star = style(r.p, r.q)
        ax.text(.975, .885 - i * .345, "%+.2f   q = %s%s"
                % (r.change, ("%.3f" % r.q) if r.q >= .001 else "<.001",
                   "  " + star if star else ""),
                transform=ax.transAxes, fontsize=7, color=col, fontweight=w,
                ha="right")
    ax.axvline(0, color="k", lw=.7, ls="--", dashes=(3, 2))
    ax.set_yticks(y[::-1])
    ax.set_yticklabels([l for _, l, _ in VL][::-1], fontsize=6.6)
    ax.set_ylim(-.6, len(VL) - .4)
    ax.set_xlabel("ESS change, first to last visit")
    ax.set_title("What the coded values do to ESS", fontsize=8.5, pad=4)

    # d — five scales × three versions
    ax = f.add_subplot(gs[1, 1])
    ax.text(-.17, 1.10, "d", transform=ax.transAxes, fontsize=10,
            fontweight="bold")
    x = np.arange(len(SCALES))
    wd = .26
    for k, (t, lab, col) in enumerate(VL):
        dz = [V[t][V[t].scale == s].iloc[0].dz for s in SCALES]
        ax.bar(x + (k - 1) * wd, dz, width=wd * .9, color=col, lw=0,
               label=lab.split("\n")[0])
    ax.axhline(0, color="k", lw=.7)
    ax.set_xticks(x)
    ax.set_xticklabels([SLAB[s] for s in SCALES], fontsize=6.8, rotation=16,
                       ha="right")
    ax.set_ylabel("$d_z$")
    ax.set_title("Only ESS is affected", fontsize=8.5, pad=4)
    # Legend placed below, outside the axes, to keep clear of the bars
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.26), ncol=3,
              fontsize=5.9, handlelength=1.0, columnspacing=1.2,
              borderpad=.2)

    f.suptitle("Handling of the ESS missing-data code",
               fontsize=10.5, x=.095, y=.945, ha="left")
    f.text(.095, .012,
           "22 first-visit ESS entries in Cohort A were stored as −800, a "
           "missing-data code rather than a measured score. They were set to "
           "missing;\nall 22 participants retain two or more valid ESS "
           "visits, so no participant is lost and n = 75 throughout. No "
           "other value on any scale fell outside its range.",
           fontsize=6.5, color="#6B7378", va="bottom")
    f.savefig(FIG / "FigS_ESS_handling.png", bbox_inches="tight")
    f.savefig(FIG / "FigS_ESS_handling.pdf", bbox_inches="tight")
    plt.close(f)
    print("FigS_ESS_handling")


if __name__ == "__main__":
    raw = load_q()
    d = recode(raw)
    res = pd.read_csv(T / "Q5_change_primary.csv")
    fig_scales(d, res)
    fig_ess_handling(raw)
    B = brain_behaviour(d)
    print("\n", B.sort_values("p").head(8).round(4).to_string(index=False))
    print("\n[saved] %s" % FIG)
