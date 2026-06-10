# Data Documentation

This document describes every dataset used in the study: its source, acquisition,
format, schema, preprocessing, assumptions, and limitations.

---

## 1. Raw corpus — TBMM plenary transcripts

| | |
|---|---|
| **Source** | Plenary session transcripts (*tutanaklar*) of the **Grand National Assembly of Turkey (TBMM)**, published as public records by the parliament. |
| **Coverage** | Legislative **Terms 22–28** (*dönem*), spanning **2002 – March 2026**. |
| **Acquisition** | Collected as plain-text session records and parsed by the scripts in `Data_Creation_Codes/`. |
| **Format** | UTF-8 CSV (with BOM). |
| **Size** | ~919 MB per file → **hosted on Google Drive**, not GitHub (see §5). |

### Built dataset: `TBMM_Network_Dataset.csv` / `TBMM_Network_Dataset_partyfixed.csv`

One row per **speech turn**. `partyfixed` applies the party-name canonicalization
(notably the HDP/HEDEP/DEM unification, see §4).

| Column | Type | Description |
|--------|------|-------------|
| `transcript_id` | str | Identifier of the source session transcript |
| `donem` | int | Legislative term (22–28) |
| `date` | date | Session date (parsed from the Turkish-calendar header) |
| `speaker` | str | Canonicalized MP / speaker name |
| `city` | str | Electoral province of the speaker |
| `party` | str | Party affiliation at the time of the speech |
| `speech` | str | Cleaned speech-turn text |
| `word_count` | int | Token count of the speech turn |

**Scale:** 1,333,436 speech turns · 2,829 unique transcripts · 5,017 unique speakers ·
19 parties · 2,706 distinct dates.

### Cleaning / preprocessing (in `Data_Creation_Codes/`)
1. **Unicode NFKC normalization**; repair of curly quotes, dashes, and page-break fragments.
2. **Segmentation** of each transcript into speech turns; date extraction via a Turkish-calendar
   pattern (Ocak–Aralık).
3. **Speaker/party resolution** against a per-term roster (`Data/milletvekilleri.csv/json`) with
   fuzzy matching for spelling variants; non-MP roles (presiding officer, committee spokespersons)
   are tagged and excluded from party-level aggregation.

---

## 2. Reference & macro data (committed to the repo)

| File | Description |
|------|-------------|
| `Data/milletvekilleri.csv`, `Data/milletvekilleri.json` | Canonical MP roster (name ↔ party ↔ term) used for speaker resolution. |
| `Data/data.txt` | Auto-generated corpus summary (shape, unique counts, missingness). |
| `economy_data/` | Monthly macroeconomic indicator series used to define and order the event windows. |

---

## 3. Processed analytical data (committed to the repo)

These are the inputs the figure pipeline actually reads, so figures reproduce **without** the raw corpus.

### `Term_<22..28>/CSVs/term_<t>_beme_concept_edges.csv`
The signed party–topic edge layer (one row per topic mention in a speech).

| Column | Description |
|--------|-------------|
| `speech_id`, `transcript_id`, `date`, `year`, `month` | Speech identity & timing |
| `speaker`, `party` | Resolved speaker and party |
| `concept_slug`, `concept`, `concept_category` | Topic tag (e.g. `economy`, category `macro_economy`) |
| `mention_count` | Number of topic mentions in the window |
| `beme_positive_hits`, `beme_negative_hits` | Valence-pattern hit counts |
| `salience_weight` | Topic salience weight for the mention |
| `beme_score` | Signed balance score `(pos − neg)/(pos + neg + 1)` ∈ [−1, 1] |
| `beme_label` | `positive` / `negative` / `mixed` (thresholds ±0.20) |

### `Term_<t>/CSVs/term_<t>_economic_event_summary.csv`
One row per event with the full metric battery: salience (`econ_salience_*`), ownership
(`top_owner_party`, `owner_shift`, `owner_persistence`, `owner_concentration_hhi_*`),
distance (`spread_*`, `negative_edge_share_*`, `signed_modularity_*`), competing frames
(`constitutional_share_*`, `security_share_*`), displacement (`competing_gain_*`,
`displacement_index_*`), and null-model statistics (`*_null_mean`, `*_null_std`, `*_null_p`).

### `Inter_Term/CSVs/`
Cross-term aggregates, e.g. `inter_term_global_economic_positions.csv` (pooled latent-axis
positions), `inter_term_economic_event_summary.csv`, `inter_term_economic_null_tests.csv`,
`inter_term_economic_figure_index.csv`, and `step1…step6_*.csv` (per-figure tables).

---

## 4. Sentiment labeling — the BEME procedure

Positive / negative / mixed discourse is labeled by an automated **binary ensemble
market-equilibrium (BEME)** procedure — a market-simulation ensemble, *not* a raw keyword
count or a rule-based lexicon. For each topic mention, a local text window is scanned against
21 positive-valence and 24 negative-valence Turkish patterns; the hit counts feed the signed
balance score above. **Validation:** against a hand-annotated goldset of 87 binary-labeled
speech windows it reaches **Macro-F1 ≈ 0.70** (Cohen's κ ≈ 0.41), versus ≈ 0.44 for a
zero-shot lexicon-only baseline. The goldset and validation notebooks are in
`Model_Comprasion/`.

**Party-name convention.** Halkların Demokratik Partisi (HDP), Halkların Eşitlik ve Demokrasi
Partisi (HEDEP), and DEM Parti are the same political movement under successive legally-forced
names and are reported collectively as **HDP** everywhere.

---

## 5. Obtaining the large raw data (Google Drive)

The two ~919 MB corpus files exceed GitHub's 100 MB/file limit and are hosted on Google Drive:

> **https://drive.google.com/drive/folders/1Z1zCytImXABvIRBK7msO6vorx7DfUn2t**

Download `TBMM_Network_Dataset.csv` and `TBMM_Network_Dataset_partyfixed.csv` and place them in
`Data/`. They are required only for **full reproduction from raw transcripts**; the published
figures regenerate from the committed processed CSVs without them.

---

## 6. Assumptions & limitations

* **Discourse, not behavior.** Edges encode *what is said*, not roll-call votes or cosponsorship.
* **Theory-guided topics.** The four economy topics (economy, inflation, interest, IMF/external
  finance) are keyword-pattern families; comparative reuse needs translation into the target
  parliament's vocabulary.
* **Automated valence.** BEME labels are validated but imperfect (κ ≈ 0.41); mixed-valence windows
  are excluded from signed edges to limit noise.
* **Temporal asymmetry.** Some events (e.g. the Term 28 Şimşek return) have incomplete post-shock
  windows, so their triptych comparability is weaker than their ownership/brokerage evidence.
