# -*- coding: utf-8 -*-
"""Applies the reviewer figure comments — re-runs the existing figure scripts in patched form (original scripts and figures untouched).
  · topomaps: extrapolate="head", which filled colour over the electrode-free occipital area -> "local" (Fig 1d,e · Fig 2e · Fig 3f–h)
  · Fig 2 c,d: P<.05 colour from orange (= sham colour) to grey
  · 'No stimulation' -> 'Sham' inside the figures
  · Fig 3b title 'No drift …' -> descriptive
  · remove 'N2+N3' from the Fig 4a y label (a Methods sentence explains it instead)
  · Fig 5: stimulation-window shading 0–30 min, P = 0.068 no longer bold
  · Supp (SWA, SO-locked/PPC): 'No stimulation' -> 'Sham'
Output  manuscript/figures/v19oct2026/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"D:/SO-Spindle논문/code/paper")
import matplotlib
matplotlib.use("Agg")

R = Path(r"D:/SO-Spindle논문/Final analysis")
V19 = R / "manuscript/figures/v19oct2026"
V19.mkdir(exist_ok=True)
RUN_ALL = "--all" in sys.argv


def reload_patched(mod, reps):
    src = open(mod.__file__, encoding="utf-8").read()
    for a, b in reps:
        assert a in src, (mod.__name__, a[:60])
        src = src.replace(a, b)
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


import figures_topography as TP                                   # noqa: E402
# topography kept as the original (head-extrapolated) version; the captions state that the occipital area is interpolated

# ── Figure 1 · 2 ───────────────────────────────────────────────────────────────
import figures_fig1_merged as FM1                                 # noqa: E402
reload_patched(FM1, [('NOMC = "#B4651A"', 'NOMC = "#6B7378"')])
import figures_fig1_split as S12                                  # noqa: E402
reload_patched(S12, [('NOMC = "#B4651A"', 'NOMC = "#6B7378"'),
                     ('label="No stimulation"', 'label="Sham"'),
                     ('["Stimulation", "No\\nstimulation"]', '["Stimulation", "Sham"]'),
                     ('"Every electrode: no stimulation \\u2192 stimulation"', '"Every electrode: sham \\u2192 stimulation"')])
S12.OUTDIR = V19
if RUN_ALL: S12.main(); print("Figure 1, 2 done")

# ── Figure 3 ───────────────────────────────────────────────────────────────────
import figures_result4_v2 as R4                                   # noqa: E402
reload_patched(R4, [('"No stimulation\\n($n$ = %d)"', '"Sham\\n($n$ = %d)"'),
                    ('ax.set_title("No drift across the recording"', 'ax.set_title("Drift in coupling prevalence across the recording"'),
                    ('"Fig4_withinnight_v2.png"', '"Fig4_withinnight_v3.png"'), ('"Fig4_withinnight_v2.pdf"', '"Fig4_withinnight_v3.pdf"')])
if RUN_ALL: R4.main(); print("Figure 3 done")

# ── Figure 4 ───────────────────────────────────────────────────────────────────
import figures_fig4_v14 as F4                                     # noqa: E402
reload_patched(F4, [('"(% of own pre-window, N2+N3)"', '"(% of own pre-window)"'),
                    ('label="No stimulation (reference)"', 'label="Sham (reference)"'),
                    ('label="No stimulation"', 'label="Sham"'),
                    ('Figure_4_slow_oscillation_population_v14"', 'Figure_4_slow_oscillation_population_v15"')])
if RUN_ALL: F4.main(); print("Figure 4 done")

# ── Figure 5 ───────────────────────────────────────────────────────────────────
import figures_timecourse as FT                                   # noqa: E402
reload_patched(FT, [('"No stimulation"', '"Sham"'), ("STIM_END = 20.0", "STIM_END = 30.0"), ("ax.set_xlim(-15, 35)", "ax.set_xlim(-15, 45)"), ('def series(ax, B, y, ylab, title, scale=1.0, baseline=False, robust=False,\n           min_n=8):', 'def series(ax, B, y, ylab, title, scale=1.0, baseline=False, robust=False,\n           min_n=8, faint=True):'),
                    ('        k = n >= min_n\n        ax.errorbar(m.index[k], m[k]',
                     "        k = n >= min_n\n        if faint:\n            ax.errorbar(m.index, m, yerr=se, fmt='o-', ms=3.0, lw=1.1, capsize=1.8, color=c, mec='white', mew=.5, alpha=.32, zorder=2)\n        ax.errorbar(m.index[k], m[k]")])
import figures_fig8_full as F8                                    # noqa: E402
reload_patched(F8, [('label="No stimulation"', 'label="Sham"')])
import figures_mechanism as FMc                                   # noqa: E402
reload_patched(FMc, [('color=col if pool.P < .10 else C["ink2"]', 'color=col if pool.P < .05 else C["ink2"]'),
                     ('fontweight="bold" if pool.P < .10 else "normal"', 'fontweight="bold" if pool.P < .05 else "normal"')])
import figures_fig6_spread_story as SS                            # noqa: E402
reload_patched(SS, [('(.900, "No stimulation", NOST)', '(.900, "Sham", NOST)'),
                    ('                    label="No stimulation")]', '                    label="Sham")]'),
                    ('            t.set_fontweight("bold"); t.set_color(PS.C["up"])', '            pass'),
                    ('"Spatial extent, aligned to onset", baseline=True, robust=True,\n              min_n=12)', '"Spatial extent, aligned to onset", baseline=True, robust=True,\n              min_n=12, faint=False)'),
                    ('"M7_time_course_bins.csv"', '"M7_time_course_bins_v2.csv"'), ('"M7_lag_correlation.csv"', '"M7_lag_correlation_v2.csv"'),
                    ])
SS.OUT = V19 / "Figure_5_spatial_spread_and_coupling_v2"
SS.main(); print("Figure 5 done")

# ── Supplementary: SWA (new 4), SO-locked/PPC (new 3) ──────────────────────────
import figures_swa_supp as SW                                     # noqa: E402
reload_patched(SW, [('label="No stimulation"', 'label="Sham"')])
SW.OUT = V19 / "Supplementary_Figure_4_slow_wave_activity_v2"
if RUN_ALL: SW.main(); print("Supp SWA done")
try:
    if not RUN_ALL: raise RuntimeError("skipped")
    import figures_fig4_merged_envelope_ppc as EP                 # noqa: E402
    src = open(EP.__file__, encoding="utf-8").read()
    reps = [('"No stimulation"', '"Sham"')] if '"No stimulation"' in src else []
    reload_patched(EP, reps)
    if hasattr(EP, "OUT"):
        EP.OUT = V19 / (Path(EP.OUT).name + "_v2")
    EP.main(); print("Supp envelope/PPC done ->", getattr(EP, "OUT", "?"))
except Exception as exc:                                          # noqa: BLE001
    print("!! envelope/PPC supplementary not regenerated:", exc)
print("all done")
