# Codebase Documentation

All analysis code lives in `src/`. Scripts resolve the repository root from their own
location (`Path(__file__).resolve().parents[1]`), so nothing is hardcoded — clone anywhere
and run. Figure scripts read the committed processed CSVs and therefore run **without** the
large raw dataset.

## Dependency graph (figure pipeline)

```
step1 … step6, regen_figure1
        │  (import)
        ▼
build_economic_shock_study.py          event study, network construction, plotting
        │  (import)
        ▼
build_term_interval_polarization.py    repo paths, party palettes, signed-feature &
                                       graph primitives, gatekeeper scoring
```

`build_all_figures.py` orchestrates the seven figure scripts and syncs outputs into `figures/`.

---

## Core modules

### `build_term_interval_polarization.py`
* **Purpose:** Foundation layer. Defines `BASE` (repo root), `PARTY_COLORS`, `PARTY_SHORT`,
  `TERMS`, the signed agenda feature (`signed_agenda_feature`), sparse signed-graph
  construction (`build_sparse_signed_graph`), and speaker gatekeeper scoring
  (`compute_gatekeeper_scores`).
* **Inputs:** `Term_<t>/CSVs/term_<t>_*` processed edge/layer CSVs.
* **Outputs:** none directly — imported by everything downstream.

### `build_economic_shock_study.py`
* **Purpose:** Turns the signed network into an economy-centered **event study**. Holds
  `ECONOMIC_TOPICS`, the `EVENT_CATALOG` (per-term shock windows), data loaders
  (`read_term_edges`, `monthly_topic_panel`, `build_party_topic_matrix`), the pooled latent
  axis (`build_global_axis`, `build_axis_validation`), and plotting helpers (`draw_bipartite`,
  `plot_event_design`, `plot_global_interval_axis`, `plot_robustness`).
* **Inputs:** per-term CSVs via `read_term_edges`; macro series in `economy_data/`.
* **Outputs:** `Inter_Term/CSVs/`, `Inter_Term/Figures/`, `Inter_Term/Notes/`, and per-term files.

---

## Figure scripts (one figure each)

| Script | Produces | Reads |
|--------|----------|-------|
| `regen_figure1.py` | `inter_term_economic_event_design.png` | `EVENT_CATALOG` |
| `step1_agenda_ownership_panel.py` | `figure_6_agenda_ownership_panel.png` + `step1_*.csv` | per-term event summaries & edges |
| `step2_partisan_distance_triptychs.py` | `figure_3/4/5_*_shock_triptych.png` | per-term edges (Terms 23, 27, 28) |
| `step3_structural_brokerage_turnover.py` | `figure_7_brokerage_turnover_panel.png` + `step3_*.csv` | gatekeeper scores |
| `step4_shock_heterogeneity_displacement.py` | `figure_8_inter_term_shock_comparison.png` + `step4_*.csv` | inter-term event summary |
| `step5_inter_term_economic_latent_axis.py` | `figure_2_inter_term_economic_latent_axis.png` + `step5_*.csv` | pooled positions |
| `step6_robustness_null_models.py` | `figure_9_robustness_null_models.png` + `step6_*.csv` | per-term null tests |
| `build_all_figures.py` | all of the above, synced into `figures/` | — |

Each writes its figure to `Inter_Term/Figures/`; `build_all_figures.py` copies them into the
canonical `figures/` folder referenced by the manuscript. Per-step result notes are written to
`Inter_Term/Notes/`.

---

## Auxiliary / exploratory scripts (not part of the figure pipeline)

Included for completeness and transparency; not required to reproduce the paper figures.

| Script | Role |
|--------|------|
| `build_beme_validation_study.py` | BEME sentiment-labeling validation against the goldset. |
| `build_beme_layered_networks.py` | Layered (positive/negative/total) network visualizations. |
| `build_event_centered_shocks.py` | Earlier event-centered shock exploration. |
| `build_multiterm_analyses.py`, `build_multiterm_event_upgrades.py` | Multi-term aggregation / party-name upgrades. |
| `build_term_directed_party_analysis.py` | Directed party analysis. **Imports a local `flex` helper** that is not bundled, so it will not run as-is — kept for reference only. |
| `build_term_stage3_to_stage7.py` | Intermediate term-processing stages. |

---

## Conventions

* **Paths:** never hardcoded; always relative to `BASE` / `__file__`.
* **Determinism:** stochastic steps use a fixed seed (`np.random.default_rng(42)`).
* **Party names:** HDP/HEDEP/DEM Parti are unified to **HDP** in `PARTY_SHORT` / `PARTY_COLORS`.
