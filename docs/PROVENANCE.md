# Provenance

Final draft, 19 Oct 2026. Script names refer to `code/`.

## Main figures and tables

| Item | Produced by |
|---|---|
| Table 1 | 02_coupling_prevalence/did_interaction.py; sensitivity rows from rev_analysis.py |
| Table 2 | 03_event_densities_and_spindles/figures_fig4_v14.py (recording-wide thresholds, N2+N3) |
| Figure 1, Figure 2 | 07_figures_and_manuscript/figures_fig1_split.py (via refig_19oct.py) |
| Figure 3 | 07_figures_and_manuscript/figures_result4_v2.py |
| Figure 4 | 03_event_densities_and_spindles/figures_fig4_v14.py; envelope panel from so_locked_windows.py |
| Figure 5 | 07_figures_and_manuscript/figures_fig6_spread_story.py; data from 04_spatial_extent_and_so_properties |
| Figure 6 | 07_figures_and_manuscript/figures_fig7_clinical_merged_v3.py; data from 06_clinical_scales |

## Supplementary figures

| Figure | Produced by |
|---|---|
| Supplementary Figure 1 (stimulation artefact) | 02_coupling_prevalence/rev_analysis.py |
| Supplementary Figure 2 (spindle detector specificity) | 03_event_densities_and_spindles/rev_spindle_specificity.py |
| Supplementary Figure 3 (SO-locked envelope, PPC) | figures_fig4_merged_envelope_ppc.py; so_locked_average.py, so_ppc_spectrum.py |
| Supplementary Figure 4 (slow-wave activity) | figures_swa_supp.py; swa_stats.py |
| Supplementary Figure 5 (coupling by SO amplitude) | figures_supp_from_fig6.py; alt_absolute_threshold.py |
| Supplementary Figure 6 (lag correlation) | figures_supp_from_fig6.py; mech_time_course_v2.py |

## Supplementary tables

| No. | File | Produced by |
|---|---|---|
| 1 | Supplementary_Table_01_window_durations_and_event_counts.csv | rev_analysis.py |
| 2 | Supplementary_Table_02_event_density_decomposition.csv | figures_fig4_v14.py (N2+N3), rev_fig4_decomposition_n2.py (N2) |
| 3 | Supplementary_Table_03_coupling_rule_and_extent_thresholds.csv | rev_coupling_rule_stats.py; rev_extent_thresholds.py + rev_extent_thresholds_stats.py |
| 4 | Supplementary_Table_04_sensitivity_analyses.csv | rev_primary_sensitivity_tables.py (random slopes, stream-specific, detection/criterion), rev_analysis.py (chance level), rev_yasaB_stats.py (staging) |
| 5 | Supplementary_Table_05_stimulation_artefact_line_components.csv | rev_analysis.py |
| 6 | Supplementary_Table_06_session_metrics.csv | final_cohort.py |
| 7 | Supplementary_Table_07_channel_level_effects.csv | did_interaction.py (interaction), final_cohort.py (between-condition) |
| 8 | Supplementary_Table_08_within_session_contrasts.csv | final_crossover.py |
| 9 | Supplementary_Table_09_slow_wave_activity.csv | swa_stats.py |
| 10 | Supplementary_Table_10_spatial_extent_and_so_properties.csv | mech_so_properties.py |
| 11 | Supplementary_Table_11_clinical_scales_and_severity.csv | questionnaires_final.py, severity_moderation.py |

## Scripts

| Folder | Script | Role | Manuscript section |
|---|---|---|---|
| 01_event_detection | staresina_detector.py | SO and spindle detector (Staresina 2015/2023 rules), coupling criterion, window-specific thresholds | Methods: Slow-oscillation and spindle detection; Coupling criterion |
| 01_event_detection | redetect_multi.py | runs the detector over every recording, window and channel; also counts the +-1 s co-occurrence rule | Methods: detection; Sensitivity: coupling rule |
| 01_event_detection | build_fif_index.py | locates the preprocessed .fif files and builds the recording list (not distributed) | Methods: participants and recordings |
| 01_event_detection | rev_redetect.py | one pass per recording: window-specific and recording-wide thresholds, time-shift surrogates, opposite half-cycle, 0.125 Hz spectra (N2) | Methods: Two threshold scopes; Chance level; Stimulation artefact |
| 01_event_detection | rev_redetect_n2n3.py | same pass restricted to N2+N3 for the event-density analyses | Methods: event densities (Figure 4, Table 2) |
| 02_coupling_prevalence | final_cohort.py | session-level metrics (Supplementary Table 6) and between-condition comparisons in the stimulation window (Figure 2; Supplementary Table 7 part 2) | Results: between-condition comparison |
| 02_coupling_prevalence | did_interaction.py | condition x window mixed model per region and per derivation (Table 1; Figure 1; Supplementary Table 7 part 1) | Methods: primary estimand; Results: primary interaction |
| 02_coupling_prevalence | build_fixed_frame.py | fixes the analysis population (230 sessions, 70/47 participants) | Methods: participants |
| 02_coupling_prevalence | fixed_frame.py | helper for the fixed analysis frame | Methods: participants |
| 02_coupling_prevalence | final_crossover.py | within-session contrasts in stimulation recordings (Figure 3; Supplementary Table 8) | Results: within-session trajectory |
| 02_coupling_prevalence | so_time_course.py | coupling against elapsed recording time (Figure 3b) | Results: within-session trajectory |
| 02_coupling_prevalence | rev_analysis.py | window durations and event counts (Supplementary Table 1), interaction under recording-wide thresholds / chance correction / opposite half-cycle, chance level (Supplementary Table 4), line components (Supplementary Table 5, Supplementary Figure 1) | Methods and Sensitivity analyses |
| 02_coupling_prevalence | rev_primary_sensitivity_tables.py | assembles the sensitivity table (random slopes, stream-specific models, detection and criterion variants) from the model outputs | Sensitivity analyses; Supplementary Table 4 |
| 03_event_densities_and_spindles | figures_fig4_v14.py | event-density models under recording-wide thresholds (Table 2; Supplementary Table 2) and Figure 4 | Results: composition of the SO population |
| 03_event_densities_and_spindles | rev_fig4_decomposition_n2.py | the same density models restricted to N2 (Supplementary Table 2, part 'regional, N2 only') | Sensitivity analyses: stage definition |
| 03_event_densities_and_spindles | rev_density_threshold_sweep.py | density models with the spindle threshold at 1-2.5 SD | Sensitivity analyses: spindle threshold |
| 03_event_densities_and_spindles | rev_spindle_threshold_sweep.py | primary interaction with the spindle threshold at 1-2.5 SD | Sensitivity analyses: spindle threshold |
| 03_event_densities_and_spindles | rev_spindle_fpr.py | false-positive share of the spindle detector against phase-randomised surrogates (mean + 1 SD) | Methods: detector specificity |
| 03_event_densities_and_spindles | rev_spindle_fpr_threshold_sweep.py | false-positive share across threshold multiples | Sensitivity analyses: spindle threshold |
| 03_event_densities_and_spindles | rev_spindle_specificity.py | random-sample inspection of detected spindles (Supplementary Figure 2) | Methods: detector specificity |
| 03_event_densities_and_spindles | rev_yasa_benchmark.py | YASA detector run on the same recordings (absolute densities) | Methods: detector benchmark; Sensitivity: detector |
| 03_event_densities_and_spindles | rev_yasa_stats.py | summary of the YASA benchmark | Sensitivity analyses: detector |
| 03_event_densities_and_spindles | swa_windows.py | slow-wave, SO-band, delta and sigma power per window and region | Methods: slow-wave activity |
| 03_event_densities_and_spindles | swa_stats.py | condition x window models for the four bands (Supplementary Table 9; Figure 4e,f; Supplementary Figure 4) | Results: SWA |
| 03_event_densities_and_spindles | so_locked_windows.py | SO-locked spindle-band envelope per window (Figure 4g) | Methods: SO-locked activity |
| 03_event_densities_and_spindles | so_locked_average.py | SO-locked averages, cluster permutation test, peak amplitude and latency (Supplementary Figure 3) | Results: SO-locked envelope |
| 03_event_densities_and_spindles | so_ppc_spectrum.py | pairwise phase consistency across carrier frequency 0.5-3 Hz (Supplementary Figure 3) | Methods: frequency specificity |
| 04_spatial_extent_and_so_properties | mech_so_properties.py | spatial extent, amplitude and slope per SO; coupling by extent; coupled vs uncoupled properties (Supplementary Table 10) | Results: properties of coupled SOs |
| 04_spatial_extent_and_so_properties | mech_tables.py | condition x window models for six SO properties (Figure 5c) | Results: properties of coupled SOs |
| 04_spatial_extent_and_so_properties | mech_time_course_v2.py | 5-min time course of coupling and extent aligned to stimulation onset, -15 to 45 min; lag correlation (Figure 5d-g; Supplementary Figure 6) | Results: spatial extent over time |
| 04_spatial_extent_and_so_properties | mech_tvns_specificity.py | coupling change for local vs widespread SOs | Results: spatial extent |
| 04_spatial_extent_and_so_properties | rev_extent_thresholds.py | extent recomputed under window-specific and recording-wide thresholds (per recording) | Methods: spatial extent; Supplementary Table 3 |
| 04_spatial_extent_and_so_properties | rev_extent_thresholds_stats.py | statistics for the two threshold scopes (Supplementary Table 3, part 'extent thresholds') | Results: spatial extent |
| 04_spatial_extent_and_so_properties | rev_coupling_rule_stats.py | primary interaction under the +-1 s co-occurrence rule (Supplementary Table 3, part 'coupling rule') | Sensitivity analyses: coupling rule |
| 04_spatial_extent_and_so_properties | alt_absolute_threshold.py | coupling within amplitude quintiles (Supplementary Figure 5) | Results: SO amplitude |
| 05_sleep_state_and_staging | rev_hypnogram_metrics.py | hypnogram metrics per recording and condition comparison | Methods: sleep staging |
| 05_sleep_state_and_staging | rev_staging_agreement.py | epoch-wise agreement between the two staging classifiers (kappa, per stage) | Methods: sleep staging |
| 05_sleep_state_and_staging | rev_redetect_yasaB.py | re-detection with every recording staged by YASA | Sensitivity analyses: sleep staging |
| 05_sleep_state_and_staging | rev_yasaB_stats.py | primary model with every recording staged by YASA (Supplementary Table 4, part 'sleep staging') | Sensitivity analyses: sleep staging |
| 05_sleep_state_and_staging | arousal_windows.py | wake/N1 fraction, artefact fraction, alpha and beta power, heart rate per window | Sensitivity analyses: sleep state and arousal |
| 05_sleep_state_and_staging | alt_explanations_tables.py | arousal-adjusted interaction and visit-order covariate | Sensitivity analyses: sleep state and arousal; visit order |
| 06_clinical_scales | questionnaires_final.py | first-to-last change on PSQI, ISI, BAI, BDI-II, ESS; ESS missing-code handling (Supplementary Table 11 part 1) | Methods and Results: clinical scales |
| 06_clinical_scales | severity_moderation.py | baseline-severity moderation (ISI >= 15), responders, coupling change vs symptom change (Figure 6; Supplementary Table 11 part 2) | Results: severity |
| 07_figures_and_manuscript | figures_fig1_split.py | Figures 1 and 2 | Figures |
| 07_figures_and_manuscript | figures_result4_v2.py | Figure 3 | Figures |
| 07_figures_and_manuscript | figures_fig3_solocked.py | SO-locked envelope panels (Figure 4g source; Supplementary Figure 3) | Figures |
| 07_figures_and_manuscript | figures_fig4_merged_envelope_ppc.py | Supplementary Figure 3 | Figures |
| 07_figures_and_manuscript | figures_fig6_spread_story.py | Figure 5 | Figures |
| 07_figures_and_manuscript | figures_fig7_clinical_merged_v3.py | Figure 6 | Figures |
| 07_figures_and_manuscript | figures_swa_supp.py | Supplementary Figure 4 | Figures |
| 07_figures_and_manuscript | figures_supp_from_fig6.py | Supplementary Figures 5 and 6 | Figures |
| 07_figures_and_manuscript | refig_19oct.py | re-runs the figure scripts with the final labels (Sham, 0-30 min window, grey for uncorrected P) into v19oct2026/ | Figures |
| 07_figures_and_manuscript | build_final_19oct.py | assembles the final manuscript and supplementary .docx, merges the 11 supplementary tables | Manuscript |
| 07_figures_and_manuscript | final_second_pass.py | text edits applied by build_final_19oct.py (called, not run directly) | Manuscript |