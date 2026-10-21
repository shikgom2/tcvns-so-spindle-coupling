# -*- coding: utf-8 -*-
"""Final draft for submission (19 Oct master). All changes are marked in red.
  1 References: fill placeholders, number in order of appearance, rewrite the list
  2 Abbreviations: define once at first appearance in each section, abbreviation thereafter
  3 Supplementary material: only items cited in the text, renumbered in order of appearance (13 tables, 8 figures), copy/rename files, rewrite the supplementary docx
  4 N: unify to 70 / 47 / 230 (remove '117', '234', '228')
  5 'no-current' etc. -> 'sham'
  6 Rewrite the Discussion
  7 Delete instructions, notes and duplicate paragraphs; 'within-night' -> 'within-session'
Output  Brain submission/final draft for submission/
"""
from __future__ import annotations

import copy
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph

import final_second_pass as SP
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import local_paths as LP  # noqa: E402

R = LP.PROJECT_ROOT / "Final analysis"
B = R / "Brain submission"
T = R / "tables"
SRC = B / "Manuscript_Brain_submission_19Oct2026_EIC_R2_comments.docx"
OUT = B / "final draft for submission"
OUT.mkdir(exist_ok=True)
(OUT / "supplementary tables").mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)
RED = RGBColor(0xC0, 0x00, 0x00)

# ───────────────────────── helpers ─────────────────────────

def find(doc, prefix, start=0):
    for i, p in enumerate(doc.paragraphs):
        if i >= start and p.text.strip().startswith(prefix):
            return i, p
    raise KeyError(prefix[:70])


def _add_marked(p, text, size=None, red=True):
    for tok in re.split(r"(\^\{[^}]*\})", text):
        if not tok:
            continue
        if tok.startswith("^{"):
            r = p.add_run(tok[2:-1]); r.font.superscript = True
        else:
            r = p.add_run(tok)
        if red:
            r.font.color.rgb = RED
        if size:
            r.font.size = Pt(size)


def drop_hyperlinks(p):
    for h in list(p._p.iter(qn("w:hyperlink"))):
        h.getparent().remove(h)


def rewrite(p, text, size=None, red=True):
    drop_hyperlinks(p)
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    _add_marked(p, text, size, red)


def insert_after(p, text, style=None, size=None, red=True):
    el = OxmlElement("w:p"); p._p.addnext(el)
    np_ = Paragraph(el, p._parent)
    if style is not None:
        np_.style = style
    elif p.style is not None:
        np_.style = p.style
    _add_marked(np_, text, size, red)
    return np_


def delete(p):
    p._p.getparent().remove(p._p)


def replace_sub(p, old, new):
    """Replace text even across run boundaries while preserving other runs' formatting (superscript etc.). New text is red; ^{} is superscript."""
    runs = p.runs
    texts = [r.text for r in runs]
    full = "".join(texts)
    idx = full.find(old)
    if idx < 0:
        raise KeyError("substring not found: %s | in: %s" % (old[:50], p.text[:60]))
    end = idx + len(old)
    starts = []; pos = 0
    for t in texts:
        starts.append(pos); pos += len(t)
    first = max(k for k in range(len(runs)) if starts[k] <= idx)
    last = max(k for k in range(len(runs)) if starts[k] < end) if end > idx else first
    before = texts[first][:idx - starts[first]]
    after = texts[last][end - starts[last]:]
    base = runs[first]
    if last == first:
        tail_el = copy.deepcopy(base._r); base._r.addnext(tail_el)
        tail = type(base)(tail_el, p); tail.text = after
    else:
        for k in range(first + 1, last):
            runs[k]._r.getparent().remove(runs[k]._r)
        runs[last].text = after
    base.text = before
    anchor = base._r
    for tok in re.split(r"(\^\{[^}]*\})", new):
        if not tok:
            continue
        nr = copy.deepcopy(base._r); anchor.addnext(nr); anchor = nr
        nrun = type(base)(nr, p)
        if tok.startswith("^{"):
            nrun.text = tok[2:-1]; nrun.font.superscript = True
        else:
            nrun.text = tok; nrun.font.superscript = False
        nrun.font.color.rgb = RED
    return True


def replace_all(p, old, new):
    n = 0
    while old in p.text and n < 20:
        replace_sub(p, old, new); n += 1
    return n


def strip_comments(doc):
    body = doc.element.body
    for tag in ("w:commentRangeStart", "w:commentRangeEnd", "w:commentReference"):
        for el in list(body.iter(qn(tag))):
            par = el.getparent()
            if tag == "w:commentReference":
                par.getparent().remove(par)
            else:
                par.remove(el)
    for rid, rel in list(doc.part.rels.items()):
        if rel.reltype.endswith("/comments") or rel.reltype.endswith("/commentsExtended") or rel.reltype.endswith("/commentsIds") or rel.reltype.endswith("/commentsExtensible"):
            doc.part.drop_rel(rid)


# ───────────────────────── references ─────────────────────────
REFS = {
    1: "Riemann D, Spiegelhalder K, Feige B, Voderholzer U, Berger M, Perlis M, et al. The hyperarousal model of insomnia: a review of the concept and its evidence. Sleep Med Rev. 2010;14:19–31.",
    2: "Dressle RJ, Spiegelhalder K, Schiel JE, Benz F, Johann A, Feige B, et al. The future of insomnia research — there’s still work to be done. J Sleep Res. 2025;34:e70091.",
    3: "Bonnet MH, Arand DL. Hyperarousal and insomnia: state of the science. Sleep Med Rev. 2010;14:9–15.",
    4: "Rosenberg R, Murphy P, Zammit G, Mayleben D, Kumar D, Dhadda S, et al. Comparison of lemborexant with placebo and zolpidem tartrate extended release for the treatment of older adults with insomnia disorder: a phase 3 randomized clinical trial. JAMA Netw Open. 2019;2:e1918254.",
    5: "Schifano F, Chiappini S, Corkery JM, Guirguis A. An insight into Z-drug abuse and dependence: an examination of reports to the European Medicines Agency database of suspected adverse drug reactions. Int J Neuropsychopharmacol. 2019;22:270–7.",
    6: "Koffel E, Bramoweth AD, Ulmer CS. Increasing access to and utilization of cognitive behavioral therapy for insomnia (CBT-I): a narrative review. J Gen Intern Med. 2018;33:955–62.",
    7: "Yap JYY, Keatch C, Lambert E, Woods W, Stoddart PR, Kameneva T. Critical review of transcutaneous vagus nerve stimulation: challenges for translation to clinical practice. Front Neurosci. 2020;14:284.",
    8: "Wang X, Sun Q, Zhang Y, Wang Y, Cai L, Wang Z. Vagus nerve stimulation for sleep disorders: emerging evidence and future directions. Nat Sci Sleep. 2026;18:630809.",
    9: "Sharon O, Fahoum F, Nir Y. Transcutaneous vagus nerve stimulation in humans induces pupil dilation and attenuates alpha oscillations. J Neurosci. 2021;41:320–30.",
    10: "Frangos E, Komisaruk BR. Access to vagal projections via cutaneous electrical stimulation of the neck: fMRI evidence in healthy humans. Brain Stimul. 2017;10:19–27.",
    11: "Nonis R, D’Ostilio K, Schoenen J, Magis D. Evidence of activation of vagal afferents by non-invasive vagus nerve stimulation: an electrophysiological study in healthy volunteers. Cephalalgia. 2017;37:1285–93.",
    12: "McIntire LK, McKinley RA, Goodyear C, McIntire JP, Brown RD. Cervical transcutaneous vagal nerve stimulation (ctVNS) improves human cognitive performance under sleep deprivation stress. Commun Biol. 2021;4:634.",
    13: "Bottenheft C, Fonken YM, Hendrikse LF, Koedijk M, Landman A, Binsch O. Effects of cervical transcutaneous vagus nerve stimulation (ctVNS) on military cognitive performance during sleep deprivation. Front Physiol. 2025;16:1542791.",
    14: "Bremner JD, Wittbrodt MT, Gurel NZ, Shandhi MH, Gazi AH, Jiao Y, et al. Transcutaneous cervical vagal nerve stimulation in patients with posttraumatic stress disorder (PTSD): a pilot study of effects on PTSD symptoms and interleukin-6 response to stress. J Affect Disord Rep. 2021;6:100190.",
    15: "Gazi AH, Harrison AB, Lambert TP, Obideen M, Alavi P, Murrah N, et al. Transcutaneous cervical vagus nerve stimulation reduces behavioral and physiological manifestations of withdrawal in patients with opioid use disorder: a double-blind, randomized, sham-controlled pilot study. Brain Stimul. 2022;15:1206–14.",
    16: "de Oliveira HM, Gallo Ruelas M, Viana Diaz CA, Oliveira de Paula G, Fruett da Costa PR, Pilitsis JG. Transcutaneous auricular vagus nerve stimulation in insomnia: a systematic review and meta-analysis. Neuromodulation. 2025;28:1332–40.",
    17: "Choi BJ, Choi H, Kang N. Transcutaneous vagus nerve stimulation influences sleep quality and insomnia: a systematic review and meta-analysis. Sleep Med Rev. 2026;89:102311.",
    18: "Lin Z, Wang D, Nie Y, Tan C, Liu Z. Transcutaneous auricular vagus nerve stimulation for sleep disorders: a systematic review and meta-analysis of sleep, anxiety, depression, and safety outcomes. Psychol Health Med. 2026;1–26.",
    19: "Staresina BP, Bergmann TO, Bonnefond M, van der Meij R, Jensen O, Deuker L, et al. Hierarchical nesting of slow oscillations, spindles and ripples in the human hippocampus during sleep. Nat Neurosci. 2015;18:1679–86.",
    20: "Staresina BP, Niediek J, Borger V, Surges R, Mormann F. How coupled slow oscillations, spindles and ripples coordinate neuronal processing and communication during human sleep. Nat Neurosci. 2023;26:1429–37.",
    21: "Mölle M, Bergmann TO, Marshall L, Born J. Fast and slow spindles during the sleep slow oscillation: disparate coalescence and engagement in memory processing. Sleep. 2011;34:1411–21.",
    22: "Klinzing JG, Niethard N, Born J. Mechanisms of systems memory consolidation during sleep. Nat Neurosci. 2019;22:1598–610.",
    23: "Helfrich RF, Mander BA, Jagust WJ, Knight RT, Walker MP. Old brains come uncoupled in sleep: slow wave–spindle synchrony, brain atrophy, and forgetting. Neuron. 2018;97:221–230.e4.",
    24: "Ng T, Noh E, Spencer RMC. Bayesian meta-analysis reveals the mechanistic role of slow oscillation–spindle coupling in sleep-dependent memory consolidation. Elife. 2025;13:e101992.",
    25: "Ngo HV, Martinetz T, Born J, Mölle M. Auditory closed-loop stimulation of the sleep slow oscillation enhances memory. Neuron. 2013;78:545–53.",
    26: "Vallat R, Walker MP. An open-source, high-performance tool for automated sleep staging. Elife. 2021;10:e70092.",
    27: "Denis D, Mylonas D, Poskanzer C, Bursal V, Payne JD, Stickgold R. Sleep spindles preferentially consolidate weakly encoded memories. J Neurosci. 2021;41:4088–99.",
    28: "Amiri M, Frauscher B, Gotman J. Phase-amplitude coupling is elevated in deep sleep and in the onset zone of focal epileptic seizures. Front Hum Neurosci. 2016;10:387.",
    29: "Vinck M, van Wingerden M, Womelsdorf T, Fries P, Pennartz CMA. The pairwise phase consistency: a bias-free measure of rhythmic neuronal synchronization. Neuroimage. 2010;51:112–22.",
    30: "Dijk DJ, Beersma DGM, van den Hoofdakker RH. All night spectral analysis of EEG sleep in young adult and middle-aged male subjects. Neurobiol Aging. 1989;10:677–82.",
    31: "Massimini M, Huber R, Ferrarelli F, Hill S, Tononi G. The sleep slow oscillation as a traveling wave. J Neurosci. 2004;24:6862–70.",
    32: "Achermann P, Borbély AA. Mathematical models of sleep regulation. Front Biosci. 2003;8:s683–93.",
    33: "Welch PD. The use of fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms. IEEE Trans Audio Electroacoust. 1967;15:70–3.",
    34: "Muehlroth BE, Sander MC, Fandakova Y, Grandy TH, Rasch B, Shing YL, et al. Precise slow oscillation–spindle coupling promotes memory consolidation in younger and older adults. Sci Rep. 2019;9:1940.",
    35: "Lustenberger C, Boyle MR, Alagapan S, Mellin JM, Vaughn BV, Fröhlich F. Feedback-controlled transcranial alternating current stimulation reveals a functional role of sleep spindles in motor memory consolidation. Curr Biol. 2016;26:2127–36.",
    36: "Niknazar M, Krishnan GP, Bazhenov M, Mednick SC. Coupling of thalamocortical sleep oscillations are important for memory consolidation in humans. PLoS One. 2015;10:e0144720.",
    37: "Carbone J, Bibián C, Reischl P, Born J, Forcato C, Diekelmann S. The effect of zolpidem on targeted memory reactivation during sleep. Learn Mem. 2021;28:307–18.",
    38: "Zhang S, He JK, Zhong GL, Wang Y, Zhao B, Zhao YN, et al. Transcutaneous auricular vagus nerve stimulation for chronic insomnia disorder: a randomized clinical trial. JAMA Netw Open. 2024;7:e2451217.",
    39: "Yeom JW, Kim H, Park S, Yoon Y, Seo JY, Cho CH, et al. Transcutaneous auricular vagus nerve stimulation (taVNS) improves sleep quality in chronic insomnia disorder: a double-blind, randomized, sham-controlled trial. Sleep Med. 2025;133:106579.",
    40: "McCormick DA. Neurotransmitter actions in the thalamus and cerebral cortex and their role in neuromodulation of thalamocortical activity. Prog Neurobiol. 1992;39:337–88.",
    41: "Osorio-Forero A, Cardis R, Vantomme G, Guillaume-Gentil A, Katsioudi G, Devenoges C, et al. Noradrenergic circuit control of non-REM sleep substates. Curr Biol. 2021;31:5009–5023.e7.",
    42: "Kjaerby C, Andersen M, Hauglund N, Untiet V, Dall C, Sigurdsson B, et al. Memory-enhancing properties of sleep depend on the oscillatory amplitude of norepinephrine. Nat Neurosci. 2022;25:1059–70.",
    43: "Swift KM, Gross BA, Frazer MA, Bauer DS, Clark KJD, Vazey EM, et al. Abnormal locus coeruleus sleep activity alters sleep signatures of memory consolidation and impairs place cell stability and spatial memory. Curr Biol. 2018;28:3599–3609.e4.",
    44: "Zhao W, Van Someren EJW, Li C, Chen X, Gui W, Tian Y, et al. EEG spectral analysis in insomnia disorder: a systematic review and meta-analysis. Sleep Med Rev. 2021;59:101457.",
}

# ───────────────────────── supplementary renumbering ─────────────────────────
TAB_MAP = {30: 1, 34: 1, 45: None, 38: 2, 40: 2, 44: 3, 46: 3, 49: 3, 36: 4, 35: 4, 43: 4, 37: 5, 1: 6, 2: 6, 3: 7, 4: 7, 6: 7, 7: 8, 32: 9, 12: 10, 13: 10, 18: 11, 22: 11}
FIG_MAP = {7: 1, 8: 2, 2: 3, 6: 4, 4: 5, 5: 6}          # old 1 (sensitivity) and 3 (spindle phase) removed
TAB_SRC = {   # new number -> (output file name, [(part label, source file)])
    1: ("window_durations_and_event_counts.csv", [(None, "Supplementary_Table_34_REV_window_durations_counts.csv")]),
    2: ("event_density_decomposition.csv", [("regional, N2+N3", "Supplementary_Table_38_REV_fig4_decomposition.csv"),
                                            ("regional, N2 only", "Supplementary_Table_38_REV_fig4_decomposition_N2.csv"),
                                            ("pooled over regions", "Supplementary_Table_40_REV_fig4_pooled_regions.csv")]),
    3: ("coupling_rule_and_extent_thresholds.csv", [("coupling rule", "Supplementary_Table_44_REV_coupling_rule_sensitivity.csv"),
                                                    ("extent thresholds", "Supplementary_Table_49_REV_extent_thresholds.csv")]),
    4: ("sensitivity_analyses.csv", [("model specification", "Supplementary_Table_35_REV_T35a_model_specifications.csv"),
                                     ("detection and criterion", "Supplementary_Table_35_REV_T35b_detection_and_criterion.csv"),
                                     ("chance level", "Supplementary_Table_36_REV_chance_level.csv"),
                                     ("sleep staging", "Supplementary_Table_43_REV_staging_sensitivity.csv")]),
    5: ("stimulation_artefact_line_components.csv", [(None, "Supplementary_Table_37_REV_line_components.csv")]),
    6: ("session_metrics.csv", [(None, "Supplementary_Table_01_S1_session_metrics.csv")]),
    7: ("channel_level_effects.csv", [("condition x window interaction", "Supplementary_Table_04_S4_channel_level_interaction.csv"),
                                      ("between-condition, stimulation window", "Supplementary_Table_06_S6_channel_level_between.csv")]),
    8: ("within_session_contrasts.csv", [(None, "Supplementary_Table_07_S7_within_night_contrasts.csv")]),
    9: ("slow_wave_activity.csv", [(None, "Supplementary_Table_32_SWA_stats.csv")]),
    10: ("spatial_extent_and_so_properties.csv", [("coupling by spatial extent", "Supplementary_Table_12_S10_coupling_by_spatial_extent.csv"),
                                                  ("coupled vs uncoupled properties", "Supplementary_Table_13_S9_coupled_vs_uncoupled_properties.csv")]),
    11: ("clinical_scales_and_severity.csv", [("clinical scale change", "Supplementary_Table_18_S19_clinical_scale_change.csv"),
                                              ("coupling change vs symptom change by severity", "Supplementary_Table_22_SEV_4_brain_behaviour_by_group.csv")]),
}
N_TAB = 11


def tab_file(new):
    return "Supplementary_Table_%02d_%s" % (new, TAB_SRC[new][0])


def build_tables(src_dir, out_dir):
    """Read the source CSVs, prepend a part column and merge them into one (union of columns)."""
    for new, (name, parts) in TAB_SRC.items():
        frames = []
        for label, f in parts:
            df = pd.read_csv(src_dir / f, encoding="utf-8-sig")
            if label is not None:
                df.insert(0, "part", label)
            frames.append(df)
        out = pd.concat(frames, ignore_index=True, sort=False) if len(frames) > 1 else frames[0]
        out.to_csv(out_dir / tab_file(new), index=False, encoding="utf-8-sig")


LEGFIX = [("Slow-wave activity does not differ between conditions.", "Slow-wave activity by condition and window."),
          ("Coupling probability shows a weak association with SO amplitude.", "Coupling probability by within-recording SO amplitude quintile."),
          ("Within-recording coupling and spatial spread were not correlated across adjacent time bins.", "Lagged correlation between within-recording coupling and spatial spread."),
          ("Stimulation artefact check:", "Narrow-band components at stimulation frequencies:"),
          ("No stimulation", "Sham"), ("no stimulation", "sham")]
TAB_LEG = {
    1: "Artefact-free N2 duration and detected SO counts per region × window cell, by condition (median and interquartile range), with the fraction of cells below 20 SOs or 120 s.",
    2: "Condition × window interactions for coupled SO, uncoupled SO and spindle density, and for coupling prevalence: regional models under window-specific and recording-wide thresholds (N2+N3), the same recording-wide models restricted to N2, and the pooled-region models (part column).",
    3: "Coupling prevalence and the condition × window interaction under the primary coupling rule and a ±1 s co-occurrence rule, and spatial extent under window-specific and recording-wide detection thresholds (part column).",
    4: "Sensitivity of the primary interaction: alternative model specifications, detection and criterion variants (recording-wide thresholds, chance-corrected prevalence, opposite half-cycle), observed prevalence against the time-shift surrogate chance level, and the interaction with every recording staged by YASA (part column).",
    5: "Stimulation artefact: line-component index at multiples of 5 Hz per recording, window and region.",
    6: "Session-level derived metrics for every analysed recording.",
    7: "Channel-level estimates at the 13 derivations: condition × window interaction for every contrast, and the between-condition effect size in the stimulation window (part column).",
    8: "Within-session contrasts in stimulation recordings, N2 and N2+N3.",
    9: "Slow-wave activity: condition × window interaction for four frequency bands.",
    10: "Coupling probability by spatial extent, and properties of spindle-carrying against uncoupled SOs (part column).",
    11: "Clinical scale change from first to last visit, and coupling change against symptom change by insomnia-severity group (part column).",
}
FIG_FILES = {1: "Supplementary_Figure_7.png", 2: "Supplementary_Figure_8.png", 3: "Supplementary_Figure_2.png", 4: "Supplementary_Figure_6.png",
             5: "Supplementary_Figure_4.png", 6: "Supplementary_Figure_5.png"}


def renum_supp(text):
    def tab(m):
        nums = [int(x) for x in re.findall(r"\d+", m.group(2))]
        new = sorted(set(TAB_MAP.get(n, n) for n in nums) - {None})
        if not new:
            return m.group(0)
        word = "Table" if len(new) == 1 else "Tables"
        return "Supplementary %s %s%s" % (word, " and ".join(str(n) for n in new), m.group(3) or "")
    def fig(m):
        nums = [int(x) for x in re.findall(r"\d+", m.group(2))]
        new = sorted(set(FIG_MAP.get(n, n) for n in nums) - {None})
        if not new:
            return m.group(0)
        word = "Figure" if len(new) == 1 else "Figures"
        return "Supplementary %s %s%s" % (word, " and ".join(str(n) for n in new), m.group(3) or "")
    text = re.sub(r"Supplementary (Tables?) ((?:\d+)(?: and \d+)?)([a-z](?:[,–][a-z])*)?", tab, text)
    text = re.sub(r"Supplementary (Fig(?:ures?|\.)) ((?:\d+)(?: and \d+)?)([a-z](?:[,–][a-z])*)?", fig, text)
    return text


# ───────────────────────── abbreviations ─────────────────────────
ABBR = [  # (regex of full form, abbreviation, key)
    (r"transcutaneous cervical vagus nerve stimulation", "tcVNS", "tcVNS"),
    (r"transcutaneous vagus nerve stimulation", "tVNS", "tVNS"),
    (r"non-rapid eye movement", "NREM", "NREM"),
    (r"[Ss]low oscillation[–-][Ss]pindle|SO[–-]Spindle|SO-spindle", "SO–spindle", "SO"),
    (r"[Ss]low oscillations(?![–-])", "SOs", "SO"),
    (r"[Ss]low oscillation(?![s–-])", "SO", "SO"),
    (r"[Ss]low-wave activity", "SWA", "SWA"),
    (r"[Ee]lectroencephalography|[Ee]lectroencephalographic|[Ee]lectroencephalogram", "EEG", "EEG"),
    (r"Pittsburgh Sleep Quality Index", "PSQI", "PSQI"),
    (r"Insomnia Severity Index", "ISI", "ISI"),
    (r"Epworth Sleepiness Scale", "ESS", "ESS"),
    (r"Beck Anxiety Inventory", "BAI", "BAI"),
    (r"Beck Depression Inventory-II", "BDI-II", "BDI-II"),
    (r"pairwise phase consistency", "PPC", "PPC"),
    (r"generalised estimating equation", "GEE", "GEE"),
    (r"confidence intervals", "CIs", "CI"),
    (r"confidence interval(?!s)", "CI", "CI"),
]


def abbreviate_section(paras):
    """paras: list of Paragraph of one section. First appearance becomes 'full form (abbreviation)', later ones the abbreviation. Heading paragraphs are left untouched."""
    seen = {}                                                     # key -> (paragraph element, position of definition)
    for p in paras:
        if p.style is not None and p.style.name.startswith("Heading"):
            continue
        for pat, ab, key in ABBR:
            pos = 0
            for _ in range(60):
                m = re.compile(pat).search(p.text, pos)
                if not m:
                    break
                full = m.group(0); after = p.text[m.end():]
                defined_here = re.match(r"\s*\(%s\)" % re.escape(ab), after) or re.match(r"\s*\(%s" % re.escape(key), after)
                if key not in seen:
                    if defined_here:
                        seen[key] = (p._p, m.start()); pos = m.end() + len(defined_here.group(0)); continue
                    if key == "SO" and ab == "SO–spindle":           # 'slow oscillation (SO)–spindle'
                        rep = "%s oscillation (SO)–spindle" % ("Slow" if full.startswith("Slow") else "slow")
                    else:
                        rep = "%s (%s)" % (full, ab)
                    replace_sub(p, full, rep) if p.text.find(full, pos) == m.start() else None
                    seen[key] = (p._p, m.start()); pos = m.start() + len(rep); continue
                if seen[key][0] is p._p and seen[key][1] == m.start():      # the definition of this section itself -> skip
                    pos = m.end() + (len(defined_here.group(0)) if defined_here else 0); continue
                target = full + (defined_here.group(0) if defined_here else "")
                _replace_at(p, target, m.start(), ab)
                pos = m.start() + len(ab)


def _replace_at(p, old, start, new):
    """When p.text[start:] begins with old, replace only at that position (earlier occurrences of the same string are untouched)."""
    runs = p.runs; texts = [r.text for r in runs]; full = "".join(texts)
    assert full[start:start + len(old)] == old, (old, full[start:start + 40])
    # Temporarily protect the earlier part: replace_sub replaces the first occurrence, so align the position if old occurs before start
    k = full.find(old)
    if k == start:
        replace_sub(p, old, new); return
    # The same string occurs earlier: split the paragraph text by run and handle it directly
    marker = "⁣"                                            # invisible separator
    # Temporarily replace earlier occurrences with marker, then restore them
    n_before = full[:start].count(old)
    for _ in range(n_before):
        replace_sub(p, old, marker + old[1:])
    replace_sub(p, old, new)
    for _ in range(n_before):
        replace_sub(p, marker + old[1:], old)


# ───────────────────────── main ─────────────────────────

def main():
    d = Document(SRC)
    strip_comments(d)
    P = d.paragraphs

    # 7. Delete instructions, notes and duplicate paragraphs ---------------------------------------------------
    i_abs, _ = find(d, "Abstract")
    for p in P[2:i_abs]:
        delete(p)
    for prefix in ("References to cite", "Primary methodological reference:", "Original algorithmic basis:", "Optional supporting reference:",
                   "Suggested references", "SWA, 0.5–4 Hz:", "SO band, 0.5–1 Hz:", "Delta, 1–4 Hz:", "Sigma, 12–16 Hz:",
                   "Sleep spindles were detected on the same channels using criteria adapted from Staresina et al. [ref]"):
        try:
            _, p = find(d, prefix); delete(p)
        except KeyError:
            pass
    _, p = find(d, "Within stimulation recordings, coupling prevalence increased numerically from the pre-stimulation to the stimulation window in all four regions: by 2.7")
    delete(p)                                                     # duplicate (old-version) paragraph; the new version starts with 'Within stimulation recordings, coupling prevalence was compared between windows'

    # 1. Reference placeholders ------------------------------------------------------------
    # 1a. Existing superscript numbers -> new numbers (before filling the placeholders!)
    OLD2NEW = {"1": "1", "2": "2", "7": "10", "8": "11", "22": "12", "23": "13", "25": "14", "24": "15", "3,4,5": "16–18"}
    for prefix in ("Insomnia is conceptualized as a disorder of hyperarousal", "Transcutaneous vagus nerve stimulation (tVNS) has emerged", "In the context of insomnia, theoretically"):
        _, p = find(d, prefix)
        for r in p.runs:
            if r.font.superscript and r.text.strip() in OLD2NEW and OLD2NEW[r.text.strip()] != r.text.strip():
                r.text = OLD2NEW[r.text.strip()]; r.font.color.rgb = RED
    _, p = find(d, "Since a ±90° window admits half")
    drop_hyperlinks(p)
    cites = [
        ("Insomnia is conceptualized as a disorder of hyperarousal", [("[4,5]", "^{1,3}"), ("GABAergic hypnotics [ref]", "GABAergic hypnotics^{4,5}"), ("remains limited in many settings [ref]", "remains limited in many settings^{6}")]),
        ("Transcutaneous vagus nerve stimulation (tVNS) has emerged", [("[6,7]", "^{7,8}"), ("[8]", "^{9}")]),
        ("In the context of insomnia, theoretically", [("sleep disorders [ref]", "sleep disorders^{8}")]),
        ("Non-rapid eye movement (NREM) sleep provides", [("[19–21]", "^{19–21}"), ("[22–25]", "^{22–25}")]),
        ("Slow oscillations were detected with the criteria of Staresina", [("Staresina and colleagues (Nat Neurosci 2023;26:1429–37) [ref]", "Staresina and colleagues^{20}"), ("Staresina et al. (Nat Neurosci 2015;18:1679–86) [ref]", "Staresina et al.^{19}")]),
        ("Each channel was zero-phase filtered at 0.3–1.25 Hz", [("third-order Butterworth filter [ref]", "third-order Butterworth filter^{20}"), ("open-source pipeline 30", "open-source pipeline^{26}")]),
        ("Since a ±90° window admits half of a uniformly distributed phase", [(" [ref , ] .", "^{27,28}.")]),
        ("To establish that the phase clustering reported here is specific", [("[ref https://pubmed.ncbi.nlm.nih.gov/20114076/]", "^{29}")]),
        ("Spectral power was estimated in each window from artefact-free N2 only", [("[REFs]", "^{21,30–32}")]),
        ("Electrical stimulation at 5–40 Hz could introduce", [("Welch power spectral density estimates [ref]", "Welch power spectral density estimates^{33}")]),
        ("Figure 1 | Changes in slow oscillation–spindle coupling", [("Staresina et al. (2015, 2023) [ref]", "Staresina et al.^{19,20}")]),
    ]
    for prefix, pairs in cites:
        _, p = find(d, prefix)
        for a, b in pairs:
            replace_sub(p, a, b)
    # 4. Unify N ----------------------------------------------------------------------
    _, p = find(d, "We analysed 230 laboratory sessions from 117 adults")
    replace_sub(p, "230 laboratory sessions from 117 adults reporting persistent sleep difficulties, comprising 70 participants during active stimulation and 47 participants without the stimulation",
                   "230 laboratory sessions from adults reporting persistent sleep difficulties, 176 with active stimulation from 70 participants and 54 sham sessions from 47 participants")
    _, p = find(d, "No study to our knowledge has taken slow oscillation–spindle coupling")
    replace_sub(p, "in 230 laboratory recordings from 117 adults reporting persistent sleep difficulties. The dataset comprised 176 active-stimulation recordings from 70 participants and 54 no-current recordings from 47 participants.",
                   "in 230 laboratory recordings from adults reporting persistent sleep difficulties, 176 active-stimulation recordings from 70 participants and 54 sham recordings from 47 participants.")
    _, p = find(d, "Spindle-band signal quality was assessed using N2 spectra from 228 recordings")
    replace_sub(p, "from 228 recordings", "from the analysed recordings")
    _, p = find(d, "Spectral power was estimated in each window from artefact-free N2 only")
    replace_sub(p, "giving 487 window-by-recording observations from 234 recordings", "giving 487 window-by-recording observations")

    # 5. Unify to 'sham' ---------------------------------------------------------------------
    SHAM = [("no-current recordings", "sham recordings"), ("No-current", "Sham"), ("no-current", "sham"), ("no current", "sham"),
            ("no-stimulation recordings", "sham recordings"), ("no-stimulation sessions", "sham sessions"), ("no-stimulation condition", "sham condition"),
            ("no-stimulation", "sham"), ("No stimulation", "Sham"), ("no stimulation", "sham"), ("within-night", "within-session"), ("Within-night", "Within-session"),
            ("(SO) –spindle", "(SO)–spindle"), ("SO-Spindle", "SO–spindle"), ("SO-spindle", "SO–spindle")]
    for p in d.paragraphs:
        for a, b in SHAM:
            if a in p.text:
                replace_all(p, a, b)
    # 'without current' is kept as is (descriptive)
    # Unify the stimulation name (Methods, Results): tVNS -> tcVNS, before abbreviation handling
    i_m = find(d, "Materials and methods")[0]; i_d = find(d, "Discussion")[0]
    for p in d.paragraphs[i_m:i_d]:
        if re.search(r"\btVNS\b", p.text):
            replace_all(p, "tVNS", "tcVNS")

    # 8. EIC/reviewer second pass (old supplementary numbers) ---------------------------------
    SP.second_pass(d, find, replace_sub, rewrite, insert_after, delete)

    # 3. Renumber supplementary material (text and legends) ------------------------------------------------------
    i_supp, _ = find(d, "Supplementary material")
    for p in d.paragraphs[:i_supp]:
        if "Supplementary" in p.text:
            new = renum_supp(p.text)
            if new != p.text:
                # token-by-token replacement
                for m in re.finditer(r"Supplementary (?:Tables?|Fig(?:ures?|\.)) (?:\d+)(?: and \d+)?(?:[a-z](?:[,–][a-z])*)?", p.text):
                    tok = m.group(0); rep = renum_supp(tok)
                    if rep != tok:
                        replace_sub(p, tok, rep)
    # Rewrite the supplementary material section
    _, p0 = find(d, "Supplementary Figures 1–7 and Supplementary Tables 1–37 are provided")
    for p in list(d.paragraphs[i_supp + 1:]):
        if p._p is not p0._p:
            delete(p)
    rewrite(p0, "Supplementary Figures 1–6 and Supplementary Tables 1–11 are provided as separate files, numbered in the order of first citation.")
    src = Document(SRC)
    old_leg = {}
    for p in src.paragraphs:
        m = re.match(r"Supplementary Figure (\d+) \|", p.text)
        if m:
            old_leg[int(m.group(1))] = p.text
    last = p0
    for new in range(1, 7):
        old = [k for k, v in FIG_MAP.items() if v == new][0]
        txt = renum_supp(re.sub(r"^Supplementary Figure \d+ \|", "Supplementary Figure X |", old_leg[old])).replace("Supplementary Figure X |", "Supplementary Figure %d |" % new, 1)
        for a, b in SHAM + LEGFIX:
            txt = txt.replace(a, b)
        last = insert_after(last, txt)
    for new in range(1, N_TAB + 1):
        last = insert_after(last, "Supplementary Table %d | %s %s" % (new, TAB_LEG[new], tab_file(new)))

    # 2. Abbreviations (per section) ------------------------------------------------------------------
    heads = ["Abstract", "Introduction", "Materials and methods", "Results", "Discussion", "Data availability", "Supplementary material"]
    idx = {}
    for i, p in enumerate(d.paragraphs):
        if p.text.strip() in heads and p.style is not None and p.style.name.startswith("Heading") and p.text.strip() not in idx:
            idx[p.text.strip()] = i
    bounds = sorted(idx.values()) + [len(d.paragraphs)]
    for k, h in enumerate(sorted(idx, key=lambda h: idx[h])):
        s0, s1 = idx[h], bounds[k + 1]
        if h in ("Data availability",):
            continue
        abbreviate_section(d.paragraphs[s0 + 1:s1])

    # 6. Discussion --------------------------------------------------------------------
    i_dis, _ = find(d, "Discussion")
    i_da, _ = find(d, "Data availability")
    disc = d.paragraphs[i_dis + 1:i_da]
    first = disc[0]
    for p in disc[1:]:
        delete(p)
    DISC = SP.DISC
    rewrite(first, DISC[0]); last = first
    for t in DISC[1:]:
        last = insert_after(last, t)

    # 1b. Reference list (before Supplementary material) ------------------------------------------
    i_supp, p_supp = find(d, "Supplementary material")
    prev = d.paragraphs[i_supp - 1]
    h = insert_after(prev, "References"); h.style = p_supp.style
    last = h
    for n in sorted(REFS):
        last = insert_after(last, "%d. %s" % (n, REFS[n]), style=d.paragraphs[i_supp + 1].style if i_supp + 1 < len(d.paragraphs) else None)

    for p in d.paragraphs:
        if "more severe insomnia" in p.text and "more severe insomnia symptoms" not in p.text:
            replace_sub(p, "more severe insomnia", "more severe insomnia symptoms")
    # swap main figures for the current files (Fig 3 = 'Stimulation sessions' version, Fig 6 = panel f re-laid)
    V19 = R / "manuscript/figures/v19oct2026"
    NEWFIG = {1: V19 / "Figure_1_condition_by_window_interaction", 2: V19 / "Figure_2_between_condition_during_stimulation",
              3: R / "figures/Fig4_withinnight_v3", 4: R / "manuscript/figures/v14oct2026/Figure_4_slow_oscillation_population_v15",
              5: V19 / "Figure_5_spatial_spread_and_coupling_v2", 6: R / "manuscript/figures/Figure_07_clinical_scales_and_severity_v3"}
    for k, src in NEWFIG.items():
        for ext in (".png", ".pdf"):
            shutil.copy(src.with_suffix(ext), OUT / "figures" / ("Figure_%d%s" % (k, ext)))
    for new, src in ((3, V19 / "Figure_04_spindle_envelope_and_frequency_specificity_v2"), (4, V19 / "Supplementary_Figure_4_slow_wave_activity_v2")):
        for ext in (".png", ".pdf"):
            shutil.copy(src.with_suffix(ext), OUT / "figures" / ("Supplementary_Figure_%d%s" % (new, ext)))
    paras = d.paragraphs
    for k in range(1, 7):
        i_cap = find(d, "Figure %d |" % k)[0]
        for j in range(i_cap - 1, max(i_cap - 4, 0), -1):
            pics = paras[j]._p.xpath(".//a:blip")
            if pics:
                rid = pics[0].get(qn("r:embed")); d.part.related_parts[rid]._blob = (OUT / "figures" / ("Figure_%d.png" % k)).read_bytes()
                print("swapped Figure", k, "at para", j); break
        else:
            print("!! no picture found for Figure", k)
    out = OUT / "Manuscript_Brain_final_draft.docx"
    d.save(out); print("saved", out)

    # ───────── Supplementary files and docx ─────────
    for old in (OUT / "supplementary tables").glob("*.csv"):
        old.unlink()
    build_tables(B / "supplementary tables", OUT / "supplementary tables")
    for new, f in FIG_FILES.items():
        shutil.copy(B / "figures" / f, OUT / "figures" / ("Supplementary_Figure_%d.png" % new))
        pdf = (B / "figures" / f).with_suffix(".pdf")
        if pdf.exists():
            shutil.copy(pdf, OUT / "figures" / ("Supplementary_Figure_%d.pdf" % new))
    s = Document()
    s.add_heading("Supplementary material", 0)
    s.add_paragraph("Transcutaneous cervical vagus nerve stimulation and slow oscillation–spindle coupling. Supplementary Figures 1–6 and Supplementary Tables 1–11, numbered in the order of first citation in the manuscript.")
    for new in range(1, 7):
        old = [k for k, v in FIG_MAP.items() if v == new][0]
        txt = renum_supp(re.sub(r"^Supplementary Figure \d+ \|", "Supplementary Figure X |", old_leg[old])).replace("Supplementary Figure X |", "Supplementary Figure %d |" % new, 1)
        for a, b in SHAM + LEGFIX:
            txt = txt.replace(a, b)
        pp = s.add_paragraph(); pp.add_run().add_picture(str(OUT / "figures" / ("Supplementary_Figure_%d.png" % new)), width=Inches(6.2))
        q = s.add_paragraph(); r = q.add_run(txt); r.font.size = Pt(9)
    s.add_heading("Supplementary tables", 1)
    for new in range(1, N_TAB + 1):
        q = s.add_paragraph(); r = q.add_run("Supplementary Table %d | %s File: %s" % (new, TAB_LEG[new], tab_file(new))); r.font.size = Pt(9)
    s.save(OUT / "Supplementary_material_final_draft.docx"); print("saved supplementary")


if __name__ == "__main__":
    main()
