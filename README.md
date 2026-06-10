# Economic Shocks and the Reconfiguration of Parliamentary Discourse Networks

A reproducible network-science study of how external **economic shocks** reorganize
**parliamentary discourse** in the Turkish Grand National Assembly (TBMM), 2002–2026.
The pipeline turns plain-text plenary transcripts into **signed, event-centered
discourse networks** and measures how shocks shift *agenda ownership*, *partisan
distance*, and *structural brokerage* between government and opposition.

> **Paper:** [`paper/manuscript.pdf`](paper/manuscript.pdf) · **Source:** [`paper/manuscript.tex`](paper/manuscript.tex)
> **Course:** CS 414/514 Network Science, Sabancı University

---

## Abstract

Parliaments do not merely register economic crises — they reorganize their internal
discourse in response to them. Using TBMM transcripts from 2002 to 2026, we construct
signed, event-centered discourse networks from topic-targeted parliamentary speech and
combine them with monthly macroeconomic indicators. The empirical design centers on
pre-shock, shock, and post-shock windows, separating abrupt reorganization from ordinary
temporal drift. We show that economic shocks do not simply increase rhetorical intensity:
they redistribute ownership of the economic agenda, widen or compress partisan distance,
and reassign brokerage across political blocs. The effects are heterogeneous — the 2008
global crisis reaction and the 2022 FX crisis both elevate economic salience but produce
opposite geometric outcomes (one widens discourse distance, the other compresses it). Under
mixed constitutional-security shocks, non-economic frames can displace the economy entirely.
The result is a portable framework for studying parliamentary discourse under economic stress
where roll-call data are unavailable.

## Research Question

> **How do external economic shocks reconfigure parliamentary discourse networks over time** —
> through shifts in agenda ownership, partisan distance, and structural brokerage — rather
> than simply intensifying conflict?

The analysis is **event-centered**, not monotonic: it compares pre-shock / shock / post-shock
windows and allows different shocks to produce different effects.

## Main Contributions

1. **Signed discourse networks.** Parliamentary speech is modeled as a signed network whose
   party/speaker links are explicitly positive (aligned) or negative (conflictual), not just present/absent.
2. **Salience vs. ownership.** A formal separation of *how much* the chamber discusses the economy
   (salience) from *who controls* that conversation (agenda ownership).
3. **A portable design** for legislatures where speech is available but roll-call / cosponsorship
   data are sparse, incomplete, or incomparable — with an explicit robustness battery
   (placebo months, bootstrap, anchor permutation, leave-one-anchor-out).

## Dataset Description

| Layer | What it is | Location |
|-------|-----------|----------|
| **Raw network dataset** | Row-per-speech-turn TBMM corpus, Terms 22–28 (~919 MB ×2) | **Google Drive** (too large for GitHub — see [`docs/DATA.md`](docs/DATA.md)) |
| **MP roster** | Canonical MP↔party reference (`milletvekilleri.csv/json`) | [`Data/`](Data/) |
| **Macroeconomic indicators** | Monthly macro series used for event windows | [`economy_data/`](economy_data/) |
| **Processed per-term data** | BEME concept edges, event summaries, gatekeepers, null tests | `Term_22/CSVs … Term_28/CSVs` |
| **Inter-term outputs** | Pooled latent-axis positions, event summaries, metrics | [`Inter_Term/CSVs/`](Inter_Term/CSVs/) |

Full provenance, schema, cleaning steps, assumptions and limitations are documented in
**[`docs/DATA.md`](docs/DATA.md)**.

## Repository Structure

```
PoliEcoTRNetSci/
├── README.md                  ← you are here
├── LICENSE                    ← MIT (code) + CC BY 4.0 (data/figures/text)
├── requirements.txt           ← pinned Python dependencies
├── paper/
│   ├── manuscript.tex         ← single-column manuscript (canonical)
│   ├── manuscript.pdf         ← compiled paper
│   ├── manuscript_ieee_twocolumn.tex  ← original IEEE conference-format source
│   └── bibliography.bib       ← references (BibTeX)
├── figures/                   ← the 9 figures used in the paper
├── src/                       ← analysis & figure-generation pipeline (see docs/CODEBASE.md)
│   ├── build_term_interval_polarization.py   ← core: paths, palettes, signed features
│   ├── build_economic_shock_study.py         ← core: event study + network construction
│   ├── step1…step6_*.py, regen_figure1.py    ← one figure each
│   └── build_all_figures.py                  ← regenerate ALL figures in one command
├── docs/
│   ├── DATA.md                ← dataset provenance, schema, cleaning
│   ├── CODEBASE.md            ← script-by-script reference
│   └── FIGURES.md             ← each figure → generating script, inputs, interpretation
├── data/  (Data/)             ← small reference data; large raw data lives on Google Drive
├── economy_data/              ← monthly macroeconomic series
├── Inter_Term/                ← cross-term CSVs, Notes, regenerated Figures
├── Term_22 … Term_28/         ← processed per-term CSVs and result notes
├── Model_Comprasion/          ← BEME sentiment-labeling validation (goldset + notebooks)
└── Data_Creation_Codes/       ← scripts that build the raw corpus from transcripts
```

## Installation

```bash
git clone https://github.com/mesely/PoliEcoTRNetSci.git
cd PoliEcoTRNetSci
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

## Environment Setup

* **Python** 3.12 (tested on 3.12.9, macOS / Apple Silicon).
* **Dependencies** (pinned in `requirements.txt`): numpy, pandas, matplotlib, networkx, scipy, seaborn, scikit-learn.
* **LaTeX** (only to rebuild the PDF): a TeX Live distribution with `pdflatex` and the
  `lmodern`, `microtype`, `caption`, `booktabs`, `hyperref` packages.
* No paths are hardcoded — every script resolves the repository root from its own location,
  so the project runs from wherever you clone it.

## Reproducing Results

All figure scripts read the **committed processed CSVs**, so you can reproduce every figure
**without downloading the large raw dataset**. From the repository root:

```bash
python src/build_all_figures.py
```

This regenerates all nine figures into `Inter_Term/Figures/` and copies them into `figures/`.

### Reproducing Figures (individually)

| Figure | Command |
|--------|---------|
| Event-design registry | `python src/regen_figure1.py` |
| Fig. 2 — inter-term latent axis | `python src/step5_inter_term_economic_latent_axis.py` |
| Fig. 3–5 — shock triptychs | `python src/step2_partisan_distance_triptychs.py` |
| Fig. 6 — agenda ownership | `python src/step1_agenda_ownership_panel.py` |
| Fig. 7 — brokerage turnover | `python src/step3_structural_brokerage_turnover.py` |
| Fig. 8 — shock heterogeneity | `python src/step4_shock_heterogeneity_displacement.py` |
| Fig. 9 — robustness / null models | `python src/step6_robustness_null_models.py` |

See **[`docs/FIGURES.md`](docs/FIGURES.md)** for inputs and interpretation of each figure.

### Reproducing Statistical Tests

The robustness battery — within-term placebo months, bootstrap stability, anchor permutation,
and leave-one-anchor-out diagnostics — is computed by
`python src/step6_robustness_null_models.py`, which writes `Inter_Term/Notes/` summaries and
Figure 9. Per-term null tests are stored in `Term_*/CSVs/term_*_economic_null_tests.csv`.
The reported z-scores (2.89, 2.79), anchor correlation (r = 0.711), permutation p-value,
and bootstrap statistics are produced here.

### Reproducing Network Analyses

Signed party-party affinity networks, party-topic bipartite graphs, the common inter-term
latent axis, and speaker-level gatekeeper (brokerage) scores are all constructed in
`src/build_economic_shock_study.py` (which imports the signed-feature and graph primitives
from `src/build_term_interval_polarization.py`). The per-figure scripts call into these
modules; see `docs/CODEBASE.md`.

### Full reproduction from raw data (optional)

To rebuild the processed per-term CSVs from the original transcripts, download the raw
dataset from Google Drive (see `docs/DATA.md`), place the two CSVs in `Data/`, and run the
construction scripts in `Data_Creation_Codes/` followed by the term-level builders. This is
only needed if you want to regenerate the corpus itself rather than the figures.

## Outputs Generated

* **`figures/`** — nine publication figures (PNG, 220 dpi).
* **`Inter_Term/CSVs/`, `Term_*/CSVs/`** — processed metrics (salience, ownership, distance,
  brokerage, displacement, null tests).
* **`Inter_Term/Notes/`, `Term_*/Notes/`** — plain-text result summaries per analysis step.
* **`paper/manuscript.pdf`** — the compiled paper.

## File Descriptions

A full, script-by-script and dataset-by-dataset description lives in the `docs/` folder:

* **[`docs/DATA.md`](docs/DATA.md)** — every dataset: source, acquisition, format, schema, columns, preprocessing, assumptions, limitations.
* **[`docs/CODEBASE.md`](docs/CODEBASE.md)** — every major script: purpose, inputs, outputs, dependencies.
* **[`docs/FIGURES.md`](docs/FIGURES.md)** — every paper figure: generating script, required inputs, interpretation.

## Authors

**Mehmet Selman Yilmaz** — Sabancı University, Istanbul, Turkey
`selman.yilmaz@sabanciuniv.edu`

## Citation

```bibtex
@misc{yilmaz2026econshocks,
  author = {Yilmaz, Mehmet Selman},
  title  = {Economic Shocks and the Reconfiguration of Parliamentary Discourse Networks},
  year   = {2026},
  note   = {CS 414/514 Network Science, Sabanc\i\ University},
  howpublished = {\url{https://github.com/mesely/PoliEcoTRNetSci}}
}
```

## License

Code is released under the **MIT License**; processed data, figures, and manuscript text
under **CC BY 4.0**. See [`LICENSE`](LICENSE). The underlying TBMM transcripts are public
records of the Grand National Assembly of Turkey.

## Reproducibility Notes

* **No hardcoded paths.** Every script derives the repo root from `__file__`; clone anywhere and run.
* **Figures reproduce from committed data.** `src/build_all_figures.py` needs only the CSVs in the repo — not the 1.8 GB raw dataset.
* **Large raw data is on Google Drive** (GitHub's 100 MB/file limit); see `docs/DATA.md` for the link and placement.
* **Party-name convention.** HDP, HEDEP, and DEM Parti are the same movement under successive
  legally-forced names and are reported collectively as **HDP** throughout the code and paper.
* **Pinned environment.** Exact tested versions are in `requirements.txt` (Python 3.12.9).
* One auxiliary script (`src/build_term_directed_party_analysis.py`) imports a local helper
  `flex` and is **not** part of the figure pipeline; it is included for completeness only.
