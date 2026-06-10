# Figure Documentation

Every figure in the manuscript, its generating script, required inputs, and interpretation.
All figures live in `figures/` (220 dpi PNG) and are regenerated with
`python src/build_all_figures.py`.

---

### Event-design registry — `inter_term_economic_event_design.png`
* **Script:** `src/regen_figure1.py`
* **Inputs:** the `EVENT_CATALOG` in `build_economic_shock_study.py`.
* **Shows:** every shock event placed by term and effective month, colored by family
  (red = core economic, orange = economic-governance, blue = mixed, gray = control).
* **Read it as:** the study's design map — which events are compared, and the pre/shock/post
  window convention (3 months before · the shock month · 3 months after) used everywhere else.

### Figure 2 — Inter-term economic latent axis (`figure_2_inter_term_economic_latent_axis.png`)
* **Script:** `src/step5_inter_term_economic_latent_axis.py`
* **Inputs:** pooled party-year-topic positions (`Inter_Term/CSVs/inter_term_global_economic_positions.csv`).
* **Shows:** each party's economic-discourse position per term on one common latent axis
  (range + central tendency), colors consistent across terms.
* **Read it as:** AKP stays durably positive and drifts further by Term 28; CHP moves to clearly
  negative positions by Terms 27–28 — inter-term drift on a shared scale.

### Figures 3–5 — Shock triptychs (`figure_3/4/5_term_{23,27,28}_shock_triptych.png`)
* **Script:** `src/step2_partisan_distance_triptychs.py`
* **Inputs:** per-term signed edges for Terms 23, 27, 28.
* **Shows:** the same party network side-by-side for pre / shock / post windows. Upper row =
  party–party alignment, lower row = party–topic structure; **blue = aligned (non-negative),
  red = conflictual (negative)**, thickness = magnitude.
* **Read it as:** shocks reorganize discourse geometry, but differently — Term 23 (global crisis)
  **widens** spread and raises negative-edge share; Term 27 (FX crisis) **compresses** toward a
  shared crisis vocabulary; Term 28 (local election) widens moderately with brokerage moving DSP→HDP.

### Figure 6 — Agenda ownership panel (`figure_6_agenda_ownership_panel.png`)
* **Script:** `src/step1_agenda_ownership_panel.py`
* **Inputs:** per-term event summaries + edges (Terms 23, 27, 28).
* **Shows:** (top) party–topic bipartite graphs; (bottom) economic salience by window,
  ownership-reallocation paths, and salience-jump magnitudes.
* **Read it as:** salience rises in both core shocks (H1), but ownership reallocates differently —
  CHP→AKP (Term 23) vs. CHP→IYI→CHP (Term 27) (H2).

### Figure 7 — Structural brokerage turnover (`figure_7_brokerage_turnover_panel.png`)
* **Script:** `src/step3_structural_brokerage_turnover.py`
* **Inputs:** speaker/party gatekeeper scores.
* **Shows:** party-level gatekeeper score (`GK`) pre vs. shock, the pre→shock top-broker switch,
  and net change by event.
* **Read it as:** brokerage is unstable — the top broker changes identity in every focal shock and
  persistence is false throughout (H4).

### Figure 8 — Shock heterogeneity & displacement (`figure_8_inter_term_shock_comparison.png`)
* **Script:** `src/step4_shock_heterogeneity_displacement.py`
* **Inputs:** `Inter_Term/CSVs/inter_term_economic_event_summary.csv`.
* **Shows:** competing-topic deltas, the displacement index `DI_e` per event, and a shock-family
  comparison. Events labeled Economic / Econ.-gov. / Mixed / Control.
* **Read it as:** core economic shocks keep **negative** displacement (economy stays central);
  mixed events (e.g. the 2007 e-memorandum) flip it **positive** — non-economic frames win (H5).

### Figure 9 — Robustness & null models (`figure_9_robustness_null_models.png`)
* **Script:** `src/step6_robustness_null_models.py`
* **Inputs:** per-term null tests (`term_*_economic_null_tests.csv`) and axis-stability diagnostics.
* **Shows:** placebo contrasts vs. real shock-window change, null-model p-values, and latent-axis
  stability under leave-one-anchor-out.
* **Read it as:** the headline shocks exceed routine monthly fluctuation (z = 2.89, 2.79) and the
  latent axis is stable (anchor r = 0.711; min leave-one-out correlation 0.972).
