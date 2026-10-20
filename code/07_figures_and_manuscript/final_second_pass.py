# -*- coding: utf-8 -*-
"""Applies the EIC/reviewer second-round comments (called from build_final_19oct.py). Run **before** supplementary renumbering (old numbering).
  Title, abstract and introduction sentences, design (parallel, randomised), window schedule (15/30/15), measurement wording, polarity, removal of the baseline-adjusted model,
  Fig 1/2/3/5/6 legends, Fig 1 interaction composition, spindle density per derivation, envelope wording, extent wording,
  responder wording (removing the crossover impression), sensitivity (event amplitude removed, FPR wording), removal of text tied to old Supp Fig 1 and 3.
"""
from __future__ import annotations

import re

TITLE = "Transcutaneous cervical vagus nerve stimulation increases slow oscillation–spindle coupling, a marker of sleep microstructure, in adults with insomnia symptoms with fragmented sleep microarchitecture"
RUNNING = "Running title: tcVNS increases slow oscillation–spindle coupling"


def second_pass(d, find_prefix, replace_sub, rewrite, insert_after, delete):
    P = d.paragraphs

    def find(doc, prefix):                                        # substring match (runs before abbreviation pass)
        for i, p in enumerate(doc.paragraphs):
            if prefix in p.text:
                return i, p
        raise KeyError(prefix[:70])
    # Title -------------------------------------------------------------------------
    rewrite(P[0], TITLE)
    _, p = find(d, "Running title:")
    rewrite(p, RUNNING)
    # Abstract: frequency-sweep sentence ---------------------------------------------
    _, p = find(d, "During stimulation, coupling prevalence increased relative to the sham condition")
    replace_sub(p, "Phase consistency between spindle timing and low-frequency phase was concentrated below 1.5 Hz and vanished above 2 Hz.",
                   "Phase consistency between spindle timing and low-frequency phase was maximal within the slow-oscillation band and declined at higher carrier frequencies.")
    m = re.search(r"Phase consistency between spindle timing and low-frequency phase.*?q < 0\.001\)\.", p.text)
    assert m, p.text[:120]
    replace_sub(p, m.group(0),
                "Spindle timing was locked specifically to the slow-oscillation rhythm: phase consistency peaked within the 0.5–1.25 Hz band "
                "and fell at higher frequencies, so the coupling measure does not reflect a filter artefact. Which slow oscillations recruited a "
                "spindle was predicted by their spatial extent: across 151,111 events, coupling probability rose from 9.8% for waves confined to "
                "one derivation to 17.6% for waves spanning nine. Across the protocol, participants who received tcVNS improved on every clinical "
                "scale (Pittsburgh Sleep Quality Index (PSQI), −3.3 points; Insomnia Severity Index (ISI), −4.3; Beck Anxiety Inventory (BAI), −4.6; "
                "Beck Depression Inventory-II (BDI-II), −2.8; Epworth Sleepiness Scale (ESS), −1.6; all q < 0.001).")
    _, p = find(d, "Together, these findings suggest that tcVNS is associated with altered SO–spindle coordination")
    rewrite(p, "These results indicate that tcVNS increases the proportion of slow oscillations that are coordinated with spindles, over frontal "
               "and parietal cortex, without a general increase in spindle production. Because this coupling is the motif through which slow "
               "oscillations and spindles organise neuronal communication during human sleep, it offers an objective physiological target for "
               "future trials of tcVNS in insomnia.")
    # Introduction -------------------------------------------------------------------
    _, p = find(d, "Insomnia is conceptualized as a disorder of hyperarousal")
    replace_sub(p, "Insomnia is conceptualized as a disorder of hyperarousal that persists into sleep which is an interdependent risk factor for depression and anxiety disorders",
                   "Insomnia is conceptualised as a disorder of hyperarousal that persists into sleep, and it is an interdependent risk factor for depression and anxiety disorders")
    replace_sub(p, ", and despite considerable progress its pathophysiology remains incompletely specified", "; despite considerable progress, its pathophysiology remains incompletely specified")
    _, p = find(d, "In the context of insomnia, theoretically")
    replace_sub(p, "In the context of insomnia, theoretically tVNS is understood to be an effective solution for sleep disorders",
                   "tVNS has been proposed as a treatment for insomnia and other sleep disorders on the basis of its effects on arousal")
    # Participants and design --------------------------------------------------------
    _, p = find(d, "Participants were adults recruited at Korea University")
    if "sham medication" in p.text:
        replace_sub(p, "sham medication", "no current medication")
    _, p = find(d, "Seventy participants (41 women, 29 men")
    replace_sub(p, "Allocation of groups for tcVNS conditions was randomised.",
                   "The design was parallel-group: each participant was allocated at random to either the stimulation or the sham condition and received only that condition in every session, so no participant contributed recordings to both conditions.")
    _, p = find(d, "Each recording is divided by annotations into three consecutive windows")
    replace_sub(p, "according to the fixed schedule of each stream, 10, 20 and 10 min in stream A and 15, 30 and 15 min in stream B, and the same schedule",
                   "according to a fixed schedule of 15, 30 and 15 min, and the same schedule")
    replace_sub(p, " The annotated stimulation windows had a median duration of 30.1 minutes.", "")
    # Measurement description --------------------------------------------------------
    _, p = find(d, "was recorded using a 16-channel OpenBCI")
    rewrite(p, "EEG was recorded with a 16-channel OpenBCI Cyton + Daisy amplifier (OpenBCI, New York, USA), whose ADS1299 front end provides 24-bit "
               "resolution, an input impedance above 1 GΩ and a common-mode rejection ratio of 110 dB. Signals were sampled at 125 Hz after on-chip "
               "anti-alias decimation (−3 dB at approximately 65 Hz), which places the Nyquist limit at 62.5 Hz, well above the 0.5–16 Hz range in "
               "which SOs and spindles were analysed; the spindle-band noise floor is quantified below. Thirteen scalp derivations (Fp1, Fp2, F7, F3, "
               "F4, F8, T3, C3, Cz, C4, T4, P3, P4) were referenced to Pz with the driven-bias electrode at Fz, and one electrocardiographic channel "
               "was recorded. Signals were filtered at 0.5–50 Hz "
               "with a zero-phase finite-impulse-response filter (Hamming window) and a 60-Hz notch filter. Bad channels were interpolated with "
               "spherical splines, signals were re-referenced to the common average of the remaining EEG channels, and independent component "
               "analysis removed electrocardiographic and ocular components. Residual narrow-band components were removed with ZapLine-plus"
               ". Segments exceeding predefined amplitude or delta-band criteria were excluded from all "
               "analyses. Sleep-stage annotations were retained and aligned with the processed signals. The 0.5-Hz high-pass filter defined the "
               "effective lower limit for SO detection. T3/T4 labels are displayed as T7/T8 in topographic plots.")
    _, p = find(d, "Sleep stages were assigned automatically in 30-s epochs")
    rewrite(p, "Sleep stages were assigned automatically in 30-s epochs with YASA (v0.7.0)^{26}, an open-source classifier validated against expert "
               "consensus scoring on large polysomnography datasets, applied to the average-referenced signal with identical settings in every "
               "recording. No manual scoring was performed, so staging could not be influenced by knowledge of condition. In the recordings for "
               "which a second, independently trained classifier was also available, epoch-wise agreement between the two was 60.1% (Cohen’s "
               "κ = 0.44; 9,022 epochs), highest for N2 (κ = 0.55), the stage on which the primary analysis rests, and lower for N1 (0.19) and REM "
               "(0.18), stages that contribute no events; the direction of the primary estimates was unchanged when every recording was staged "
               "with YASA (Sensitivity analyses). Coupling prevalence analyses were restricted to N2; event-density and SO property analyses also "
               "included N3.")
    _, p = find(d, "instantaneous phase of the")
    replace_sub(p, "As the recording polarity could not be verified from the device documentation, this peak was treated as the putative up-state.",
                   "The polarity of the recordings was established from the data rather than from the device documentation: spindle-band activity "
                   "rose on the ascending limb towards this positive peak and spindle peaks clustered around it (Figure 4g; Supplementary Figure 2), "
                   "the configuration expected for the depolarising up-state^{19,21}, and the positive peak was therefore treated as the up-state.")
    # Remove the baseline-adjusted model (tied to old Supp Fig 1) ------------------------------
    _, p = find(d, "For the during-stimulation comparison, recordings were averaged within participant and condition")
    m = re.search(r" A baseline-adjusted mixed-effects model.*?subsequent change\.", p.text)
    if m:
        replace_sub(p, m.group(0), "")
    # Figure 1 text and legend ----------------------------------------------------------
    _, p = find(d, "The condition × window interaction indicated a higher coupling prevalence during stimulation")
    replace_sub(p, "indicating that the most consistent evidence for stimulation-related enhancement was observed over frontal and parietal cortex.",
                   "indicating that the most consistent evidence for stimulation-related enhancement was observed over frontal and parietal cortex. "
                   "The interaction combines an increase from baseline within stimulation recordings with a small decrease within sham recordings "
                   "(Figure 1a,b); the two components are examined separately below, and the between-condition difference in the level of coupling "
                   "during stimulation, which does not depend on the baseline change, is reported in the next section.")
    _, p = find(d, "Figure 1 | Changes in")
    replace_sub(p, "for the stimulation-versus-pre-stimulation (e) Topography", "for the stimulation-versus-pre-stimulation contrast. (e) Topography")
    # Figure 2 legend -------------------------------------------------------------------
    _, p = find(d, "Figure 2 | Between-condition comparison")
    replace_sub(p, "parietal 10,673).", "parietal 10,673). In (c) and (d), marker colours encode significance (blue, q < 0.05 after Benjamini–Hochberg "
                   "correction across the four regions; dark grey, P < 0.05 uncorrected; light grey, not significant), not condition; a filled star marks "
                   "q < 0.05 and an open star P < 0.05.")
    replace_sub(p, "the montage contains no occipital electrode and posterior values are interpolated for display.",
                   "the montage contains no occipital electrode, and values over the occipital area are interpolated from the surrounding electrodes.")
    _, p = find(d, "Figure 1 | Changes in")
    replace_sub(p, "No occipital electrode was included; posterior values on the topographic display are interpolated.",
                   "No occipital electrode was included; values over the occipital area of the topographic display are interpolated from the surrounding electrodes.")
    _, p = find(d, "a bin was displayed only when at least 12 recordings contributed to it")
    replace_sub(p, "a bin was displayed only when at least 12 recordings contributed to it", "a bin was displayed only when at least 12 recordings of a condition contributed to it, except in the coupling time course, where bins with fewer recordings are drawn in a lighter tone")
    _, p = find(d, "Figure 5 | Associations between SO spatial extent")
    replace_sub(p, "lines show means ± s.e.m.", "lines show means ± s.e.m.; bins to which fewer than 12 recordings of a condition contributed are omitted in (d) and drawn in a lighter tone in (e).")
    _, p = find(d, "values shown over posterior scalp in the topographic figures are interpolated")
    replace_sub(p, "values shown over posterior scalp in the topographic figures are interpolated and are explicitly labelled as such",
                   "values shown over the occipital area in the topographic figures are interpolated from the surrounding electrodes and are labelled as such in the figure legends")
    _, p = find(d, "Figure 3 | Within-session changes")
    replace_sub(p, "Coupling prevalence in 5-min bins across with stimulation and without stimulation recordings", "Coupling prevalence in 5-min bins across stimulation and sham recordings")
    replace_sub(p, "(f–h) Corresponding estimates at the 13 derivations for the three contrasts;", "(f–h) Corresponding estimates at the 13 derivations for the three contrasts; no occipital electrode was recorded and values over the occipital area are interpolated from the surrounding electrodes;")
    _, p = find(d, "Figure 4 | Decomposition of coupling prevalence")
    replace_sub(p, "under recording-wide thresholds.", "under recording-wide thresholds. Event densities in (a–d) are computed over artefact-free N2 and N3 sleep (Methods); the prevalence analyses of Figures 1–3 use N2 only.")
    # Figure 3 legend -------------------------------------------------------------------
    _, p = find(d, "Figure 3 | Within-session changes")
    replace_sub(p, "shading marks the median stimulation window", "shading marks the stimulation window")
    # Density: per-derivation values -----------------------------------------------------
    _, p = find(d, "Relative to the corresponding change in the sham recordings")
    replace_sub(p, "(β = +10.7, 95% CI 4.2 to 17.2; q = 0.005; Figure 4c)",
                   "(β = +10.7, 95% CI 4.2 to 17.2; q = 0.005; Figure 4c), regional sums over four derivations that correspond to approximately "
                   "+2.2 and +2.7 spindles min⁻¹ per derivation")
    # Envelope wording; remove old Supp Fig 3 ---------------------------------------------
    _, p = find(d, "The spindle-band envelope time-locked to the")
    replace_sub(p, "rising on the ascending phase and peaking shortly before the up-state", "rising after the trough and peaking on the ascending phase before the up-state")
    m = re.search(r"Between conditions, by contrast, the baseline-corrected envelopes did not differ in peak amplitude or latency, and at the event level spindle peaks clustered on the ascending phase of the SO under both conditions \(Supplementary Figures? \d+ and \d+\)\.", p.text)
    assert m, p.text[:200]
    replace_sub(p, m.group(0), "Between conditions, by contrast, the baseline-corrected envelopes did not differ in peak amplitude or latency (Supplementary Figure 2).")
    # extent -------------------------------------------------------------------------
    _, p = find(d, "Coupling probability increased with")
    replace_sub(p, "(all P ≥ 0.53; Figure 5c).", "(all P ≥ 0.53; Figure 5c). Because coupling was assessed at the derivation on which each SO was detected, "
                   "the number of derivations at which a wave appeared did not alter the opportunity to detect its spindle, so the graded relation is "
                   "not an artefact of detection opportunity.")
    _, p = find(d, "We next examined changes in")
    replace_sub(p, "Spatial extent increased over time to a greater extent during stimulation in the active group compared to the sham group (Figure 5d,f).",
                   "Spatial extent increased over the session in both conditions, and the between-condition difference was not significant (Figure 5d,f).")
    _, p = find(d, "Figure 5 | Associations between SO spatial extent")
    replace_sub(p, "Shading in (d,e) marks the median stimulation window", "Shading in (d,e) marks the stimulation window")
    # Figure 6 legend and responder wording ------------------------------------------------
    _, p = find(d, "Figure 6 | Clinical symptom changes")
    replace_sub(p, "markers show means with bootstrap 95% CIs.", "markers show means with bootstrap 95% CIs; in (b), coloured points mark participants whose score improved and grey points those whose score worsened.")
    replace_sub(p, "coloured points indicate responders, with counts shown above each distribution.",
                   "filled points indicate responders and open points non-responders; responder counts are given beneath each group (stimulation, 5 of 32 versus 9 of 26; Fisher’s exact P = 0.13).")
    _, p = find(d, "Interestingly, the physiological effect was not uniform across the cohort")
    replace_sub(p, "9 of 26 participants with severe insomnia met the threshold during stimulation, compared with 1 of 11 in their sham sessions. Among participants with milder insomnia,",
                   "9 of 26 participants with ISI ≥ 15 met the threshold during stimulation, compared with 1 of 11 participants with sham sessions. Among participants with ISI < 15,")
    # Tidy citations of unsubmitted supplementary tables (inventory, YASA benchmark, channel metrics) ----
    _, p = find(d, "The full inventory of recordings, including window durations, is provided in Supplementary Table 30.")
    replace_sub(p, "The full inventory of recordings, including window durations, is provided in Supplementary Table 30.",
                   "Window durations and event counts by condition are summarised in Supplementary Table 34.")
    _, p = find(d, "with absolute SO and spindle densities reported in Supplementary Table 45")
    replace_sub(p, "with absolute SO and spindle densities reported in Supplementary Table 45", "with absolute SO and spindle densities reported under Sensitivity analyses")
    for _ in range(2):
        try:
            _, p = find(d, "Session-level and channel-level derived tables, the membership lists for all five participant sets, the recording inventory linking every analysed session to its condition and visit order, and the code producing every figure and table are provided with the supplementary material.")
        except KeyError:
            break
        replace_sub(p, "Session-level and channel-level derived tables, the membership lists for all five participant sets, the recording inventory linking every analysed session to its condition and visit order, and the code producing every figure and table are provided with the supplementary material.",
                       "Session-level derived tables are provided with the supplementary material; channel-level tables, the recording inventory linking every analysed session to its condition and visit order, and the code producing every figure and table are available from the corresponding author.")
    # Sensitivity ----------------------------------------------------------------------
    _, p = find(d, "We examined the amplitude of the detected events, sleep state and arousal")
    replace_sub(p, "We examined the amplitude of the detected events, sleep state and arousal", "We examined sleep state and arousal")
    _, p = find(d, "Event amplitude. The interaction was present in the lowest within-recording SO-amplitude quintile")
    delete(p)
    _, p = find(d, "Coupling rule and detector. Under a ±1 s co-occurrence rule")
    replace_sub(p, "Raising the spindle threshold from mean + 1 to mean + 2 standard deviations, where the detector’s false-positive share falls from 81% to 17%, left",
                   "Raising the spindle threshold from mean + 1 to mean + 2 standard deviations, a stricter criterion under which fewer than one event in five "
                   "is attributable to background sigma fluctuations in phase-randomised surrogates, left")
    p2 = insert_after(p, "Stage definition. Restricted to N2, the density estimates under recording-wide thresholds kept their direction but lost "
                     "regional significance: coupled SO density rose in frontal cortex after stimulation (β = +0.57 events min⁻¹; P = 0.043; "
                     "q = 0.17) and, pooled across regions, during stimulation (+0.33; P = 0.006); spindle density rose when pooled (+4.1 events "
                     "min⁻¹ after stimulation; P = 0.003) but in no single region after correction (q ≥ 0.15); and the reduction in uncoupled SOs "
                     "was not significant (pooled −0.65; P = 0.32; Supplementary Table 38). The density analyses therefore draw their power from "
                     "the N3 epochs available in about half of the recordings, whereas the prevalence interaction keeps its direction and size "
                     "under both stage definitions (recording-wide thresholds: frontal β = 0.054 and 0.071, parietal β = 0.084 and 0.096 for N2 "
                     "and N2+N3, respectively; Supplementary Tables 35 and 38).")
    insert_after(p, "Sleep staging. With every recording staged by YASA, including those originally staged by the second classifier, the "
                    "stimulation-window interaction retained its direction and magnitude in frontal (β = 0.042; P = 0.074) and parietal cortex "
                    "(β = 0.048; P = 0.12; Supplementary Table 43).")


DISC = [
    "Slow oscillation (SO)–spindle coupling is a fundamental feature of non-rapid eye movement (NREM) sleep and a candidate mechanism through which sleep supports memory "
    "consolidation. Spindles nested within the SO up-state are thought to provide a temporal window for the redistribution of "
    "hippocampal reactivation to neocortical networks^{19,20,22}. The precision of this coordination has been associated with "
    "overnight memory retention and is altered in the context of cortical atrophy^{23,34}, while a recent Bayesian meta-analysis "
    "further supports a general contribution of SO–spindle coupling to memory consolidation^{24}. Importantly, this coordination "
    "is modifiable by interventions delivered during sleep, including phase-locked auditory stimulation, transcranial alternating "
    "current stimulation and pharmacological manipulation^{25,35–37}. Whether peripheral vagal neuromodulation can influence this "
    "endogenous sleep architecture, however, has remained largely unexplored. Clinical studies of transcutaneous vagus nerve stimulation (tVNS) for sleep have focused "
    "predominantly on subjective outcomes and have generally targeted the auricular branch^{16,17,38,39}. By examining SO–spindle "
    "coordination during transcutaneous cervical vagus nerve stimulation (tcVNS), the present study extends this literature to an objective measure of sleep microstructure and "
    "identifies a potential physiological pathway through which cervical vagal stimulation could influence sleep.",

    "The principal finding is that tcVNS was associated with a higher prevalence of SO–spindle coupling rather than with a generalised "
    "increase in sleep oscillatory activity. The proportion of SOs accompanied by a spindle in the up-state half-cycle rose by five to six "
    "percentage points relative to the concurrent change in sham sessions, a relative increment of roughly one half against a prevalence of "
    "10–14%, with the clearest evidence over frontal and parietal cortex. The estimate kept its sign across three levels of aggregation, in a "
    "mixed model that treats repeated recordings of a participant as dependent, in the between-condition comparison of stimulation-window "
    "levels and in an event-level model in which every SO contributes a binary outcome; it exceeded the chance level derived from "
    "time-shifted surrogates by a factor of about two; and it was invariant to whether detection thresholds were defined within each window "
    "or across the whole recording. Interventions previously shown to act on this motif have driven the SO itself, through sound timed to "
    "the up-state, or have altered thalamocortical excitability directly through current or drug^{25,35,36}. A peripheral afferent volley "
    "that is neither phase-locked to the SO nor applied to cortex is a categorically different perturbation, and its association with the "
    "proportion of coupled SOs rather than with the abundance of either rhythm reframes the question such stimulation poses: not whether "
    "vagal input changes how much oscillatory activity sleep contains, but whether it changes how that activity is organised in time.",

    "Prevalence and event density interrogate different aspects of the signal, and the data hold them apart. With thresholds fixed once per "
    "recording, so that events in every window are counted against a single yardstick, the rise in prevalence was accompanied by a greater "
    "density of spindle-carrying SOs in frontal cortex, of spindles in frontal and temporal cortex, and a scalp-wide reduction in SOs devoid "
    "of a spindle; these gains persisted under a stricter spindle criterion. With thresholds re-estimated within each window, the same change "
    "manifested as a loss of uncoupled SOs, because a criterion that ascends with the amplitude of a window excludes events that a fixed "
    "criterion admits as slow-wave and sigma amplitude rise across the session. Under both scopes the proportion rose and under neither did "
    "uncoupled events increase. The constitution of the change thus depends on the yardstick, whereas the change itself does not, and the "
    "lesson is methodological as much as physiological: an intervention delivered during sleep must be judged not only by the number of "
    "events a detector returns but by which of those events it classifies as coupled, since the two quantities can dissociate.",

    "The spatial organisation of the coupled events supplies the proportion with a physiological substrate. Coupling probability increased "
    "almost monotonically from 9.8% for SOs confined to a single derivation to 17.6% for SOs expressed at nine, and coupled SOs had steeper "
    "trough-to-peak transitions in every region. Both relations were reproduced with thresholds fixed across channels on "
    "common-average-referenced data, and coupling was assessed on the derivation at which each SO was detected, so neither channel-specific "
    "noise, a shared reference nor detection opportunity accounts for them. Coupling is concentrated on waves that are both widespread and "
    "abrupt in their ascent to the up-state, precisely the configuration in which a synchronous cortical depolarisation delivers "
    "corticothalamic drive coherent enough to initiate a spindle within the same cycle^{20,21}. The frontal and parietal distribution of the "
    "condition difference is congruent with the topography of this motif, in which frontal SOs couple to centroparietal fast spindles "
    "synchronised to the up-state^{21}, and its presence in two non-adjacent regions argues against a phenomenon confined to a single "
    "derivation. Extent is the standing determinant of recruitment; it increased within recordings of both conditions, so whether "
    "stimulation acts on extent, on the steepness of the up-state or on a thalamic gate that extent merely indexes is the question this "
    "result hands to the next study.",

    "What did not change delineates the effect as sharply as what did. Slow-wave activity (SWA) and sigma power rose within the stimulation "
    "recordings from the pre-stimulation to the stimulation window, yet sham recordings moved in the same direction and no band differed "
    "between conditions in any region. The spindle-band envelope locked to the SO trough retained its time course in every window and "
    "condition, rising after the trough and peaking on the ascending phase before the up-state, with greater amplitude during and after "
    "stimulation but no between-condition difference in peak amplitude or latency, and indices of sleep depth and arousal neither differed "
    "between conditions nor altered the estimate when adjusted for. The condition difference therefore resides in the co-occurrence of the "
    "two rhythms, not in the quantity of either or in the timing of spindle activity within the SO cycle.",

    "A single candidate mechanism deserves to be stated with its difficulty intact. Transcutaneous stimulation of the antero-lateral neck "
    "engages the nucleus of the solitary tract and the parabrachial complex^{10}, the principal visceral afferents to the locus coeruleus. "
    "A sustained elevation of noradrenergic tone would be expected to curtail spindles rather than to increase coupling: noradrenaline "
    "depolarises thalamocortical neurons out of the burst-firing mode that generates spindles^{40}, spindles cluster in the troughs of the "
    "infraslow noradrenergic oscillation that structures NREM sleep^{41,42}, and optogenetic activation of the locus coeruleus during sleep "
    "suppresses spindles and impairs consolidation^{43}. The observations are reconciled if stimulation modulates the amplitude or phase of "
    "that infraslow oscillation rather than its mean, prolonging the troughs within which an up-state can recruit a spindle; this would yield "
    "exactly a gain of coupled events with spindle form unaltered, and it is testable with autonomic indices time-locked to events and with "
    "stimulation phase-locked to the SO. Clinically, all five symptom scales improved across the protocol with medium to large "
    "within-participant effects, and participants above the conventional severity threshold on the Insomnia Severity Index were more often "
    "coupling responders under stimulation. Given the cortical hyperarousal that characterises insomnia during NREM sleep^{1,44}, a shift of "
    "the SO population towards coupled events is the direction in which a sleep-promoting intervention would be expected to act, and coupling "
    "prevalence offers a quantifiable physiological correlate of exposure against which clinical response can be tested prospectively.",

    "Several limitations define the next experiment. Sessions were daytime recordings in young, symptom-selected adults rather than "
    "overnight polysomnography in diagnosed patients, and the sham condition, although delivered with the same device and schedule at 0 mA, "
    "was not an active sham. Staging was automated, the detector was deliberately inclusive at the primary threshold, and the parietal "
    "estimate, unlike the frontal estimate and the density results, did not survive the stricter spindle criterion. The clinical improvement "
    "was uncontrolled and uncorrelated with the change in coupling, and the severity moderation rests on subgroups specified after the "
    "primary result, so coupling is proposed as a correlate of exposure rather than as a demonstrated mediator of benefit; coupling precision "
    "and strength, the properties most consistently predictive of memory^{23,24}, were not shown to change. Within these bounds, the study "
    "establishes that cervical stimulation is associated with a higher proportion of SOs that carry a spindle, with the spindle itself "
    "unchanged and coupling concentrated on widespread, steep waves. The key finding is not a uniform increase in sleep-related activity but "
    "a change in how that activity is organised in time, and that is the property a preregistered, phase-locked, overnight trial with an "
    "active sham and prespecified physiological and clinical outcomes should now test.",
]


# ── Length adjustment (reviewer: to about two thirds) ─────────────────────────────
_TRIM = [
 ("The estimate kept its sign across three levels of aggregation, in a mixed model that treats repeated recordings of a participant as dependent, in the between-condition comparison of stimulation-window levels and in an event-level model in which every SO contributes a binary outcome; it exceeded the chance level derived from time-shifted surrogates by a factor of about two; and it was invariant to whether detection thresholds were defined within each window or across the whole recording.",
  "The estimate kept its sign in a mixed model that treats repeated recordings as dependent, in the between-condition comparison of stimulation-window levels and in an event-level model in which every SO contributes a binary outcome; it exceeded the chance level from time-shifted surrogates by a factor of about two; and it was invariant to the scope of the detection thresholds."),
 ("reframes the question such stimulation poses: not whether", "reframes the question: not whether"),
 ("Under both scopes the proportion rose and under neither did uncoupled events increase. The constitution of the change thus depends on the yardstick, whereas the change itself does not, and the lesson is methodological as much as physiological: an intervention delivered during sleep must be judged not only by the number of events a detector returns but by which of those events it classifies as coupled, since the two quantities can dissociate.",
  "Under both scopes the proportion rose and under neither did uncoupled events increase: the constitution of the change depends on the yardstick, whereas the change itself does not, so an intervention delivered during sleep must be judged not only by the number of events a detector returns but by which of them it classifies as coupled."),
 (", and its presence in two non-adjacent regions argues against a phenomenon confined to a single derivation.", "."),
 ("The observations are reconciled if stimulation modulates the amplitude or phase of that infraslow oscillation rather than its mean, prolonging the troughs within which an up-state can recruit a spindle; this would yield exactly a gain of coupled events with spindle form unaltered, and it is testable with autonomic indices time-locked to events and with stimulation phase-locked to the SO.",
  "The observations are reconciled if stimulation modulates the amplitude or phase of that infraslow oscillation rather than its mean, prolonging the troughs within which an up-state can recruit a spindle, which would yield exactly a gain of coupled events with spindle form unaltered; the hypothesis is testable with autonomic indices time-locked to events and stimulation phase-locked to the SO."),
 ("Both relations were reproduced with thresholds fixed across channels on common-average-referenced data, and coupling was assessed on the derivation at which each SO was detected, so neither channel-specific noise, a shared reference nor detection opportunity accounts for them.",
  "Both relations were reproduced with thresholds fixed across channels on common-average-referenced data, and coupling was assessed on the derivation at which each SO was detected, so neither channel-specific noise nor detection opportunity accounts for them."),
]


def _apply(t):
    for a, b in _TRIM:
        if a in t:
            t = t.replace(a, b, 1)
    return t


_n0 = sum(len(t.split()) for t in DISC)
DISC = [_apply(t) for t in DISC]
assert sum(len(t.split()) for t in DISC) < _n0 - 40, "trim did not apply"
