"""
Create goldset_annotation.csv for Model_Comprasion project.
Combines 108 existing goldset rows + 625 new annotation candidates = 733 rows total.
Adds 5 annotator columns (ann_kisi1–ann_kisi5) and final_label column.

Sampling strategy for 625 new candidates (excluding existing goldset IDs):
  - 250 positive heuristic (heuristic_label = 'positive')
  - 250 negative heuristic (heuristic_label = 'negative')
  - 125 border        (heuristic_label = 'mixed', but pos_hits>0 or neg_hits>0)
"""

import pandas as pd
import numpy as np
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = Path('/Users/selmanyilmaz/PoliEcoTRNetworSci')
GOLDSET_CSV   = ROOT / 'Term_22/CSVs/term_22_beme_validation_goldset_final.csv'
CANDIDATES_CSV = ROOT / 'Term_22/CSVs/term_22_beme_validation_candidate_windows.csv'
OUT_DIR = ROOT / 'Model_Comprasion/data'
OUT_CSV = OUT_DIR / 'goldset_annotation.csv'

# ── Load sources ─────────────────────────────────────────────────────────────
goldset   = pd.read_csv(GOLDSET_CSV)
candidates = pd.read_csv(CANDIDATES_CSV)

print(f"Goldset  : {len(goldset)} rows  | labels: {goldset['manual_label'].value_counts().to_dict()}")
print(f"Candidates: {len(candidates)} rows | heuristic: {candidates['heuristic_label'].value_counts().to_dict()}")

# ── Exclude already-annotated IDs from candidate pool ────────────────────────
existing_ids = set(goldset['candidate_id'])
pool = candidates[~candidates['candidate_id'].isin(existing_ids)].copy()
print(f"Candidate pool after exclusion: {len(pool)} rows")

# ── Helper: binary label from heuristic ──────────────────────────────────────
def heuristic_binary(row):
    if row['heuristic_label'] == 'positive':
        return 'positive'
    elif row['heuristic_label'] == 'negative':
        return 'negative'
    else:  # mixed – break tie on score, default negative
        return 'positive' if row['heuristic_score'] > 0 else 'negative'

pool['heuristic_label_binary'] = pool.apply(heuristic_binary, axis=1)

# ── Sample 625 new candidates ─────────────────────────────────────────────────
pos_pool    = pool[pool['heuristic_label'] == 'positive']
neg_pool    = pool[pool['heuristic_label'] == 'negative']
border_pool = pool[(pool['heuristic_label'] == 'mixed') &
                   ((pool['heuristic_positive_hits'] > 0) | (pool['heuristic_negative_hits'] > 0))]

print(f"Pools → pos={len(pos_pool)}, neg={len(neg_pool)}, border={len(border_pool)}")

n_pos, n_neg, n_border = 250, 250, 125
assert len(pos_pool) >= n_pos,    f"Not enough positive: {len(pos_pool)}"
assert len(neg_pool) >= n_neg,    f"Not enough negative: {len(neg_pool)}"
assert len(border_pool) >= n_border, f"Not enough border: {len(border_pool)}"

sampled_pos    = pos_pool.sample(n=n_pos, random_state=SEED)
sampled_neg    = neg_pool.sample(n=n_neg, random_state=SEED)
sampled_border = border_pool.sample(n=n_border, random_state=SEED)

new_candidates = pd.concat([sampled_pos, sampled_neg, sampled_border], ignore_index=True)
print(f"New candidates sampled: {len(new_candidates)} | heuristic dist: {new_candidates['heuristic_label'].value_counts().to_dict()}")

# ── Build unified annotation DataFrame ───────────────────────────────────────
ANNOTATOR_COLS = ['ann_kisi1', 'ann_kisi2', 'ann_kisi3', 'ann_kisi4', 'ann_kisi5']
SHARED_COLS = ['candidate_id', 'window_text', 'source_party', 'target_party',
               'date', 'year', 'speaker',
               'heuristic_positive_hits', 'heuristic_negative_hits',
               'heuristic_score', 'heuristic_label']

# ── Section A: existing 108 goldset rows ─────────────────────────────────────
section_a = goldset[SHARED_COLS + ['manual_label', 'manual_notes']].copy()
section_a['heuristic_label_binary'] = section_a.apply(heuristic_binary, axis=1)
# ann_kisi1 = manual_label (the author's annotation), others blank
for col in ANNOTATOR_COLS:
    section_a[col] = ''
section_a['ann_kisi1']       = section_a['manual_label']   # existing human label
section_a['final_label']     = section_a['manual_label']   # authoritative
section_a['is_human_verified'] = True

# ── Section B: 625 new annotation candidates ──────────────────────────────────
section_b = new_candidates[SHARED_COLS + ['heuristic_label_binary']].copy()
section_b['manual_label']  = ''
section_b['manual_notes']  = ''
for col in ANNOTATOR_COLS:
    section_b[col] = ''
section_b['final_label']       = section_b['heuristic_label_binary']  # AI fallback
section_b['is_human_verified'] = False

# ── Merge and reorder columns ─────────────────────────────────────────────────
FINAL_COLS = [
    'candidate_id', 'manual_label', 'manual_notes',
    'window_text', 'source_party', 'target_party',
    'date', 'year', 'speaker',
    'heuristic_positive_hits', 'heuristic_negative_hits',
    'heuristic_score', 'heuristic_label', 'heuristic_label_binary',
    'ann_kisi1', 'ann_kisi2', 'ann_kisi3', 'ann_kisi4', 'ann_kisi5',
    'final_label', 'is_human_verified'
]

combined = pd.concat([section_a, section_b], ignore_index=True)
combined = combined[FINAL_COLS]

# ── Save ──────────────────────────────────────────────────────────────────────
combined.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
print(f"Total rows : {len(combined)}")
print(f"  Existing (human-verified): {combined['is_human_verified'].sum()}")
print(f"  New candidates           : {(~combined['is_human_verified']).sum()}")
print(f"\nfinal_label distribution:")
print(combined['final_label'].value_counts())
print(f"\nheuristic_label_binary distribution:")
print(combined['heuristic_label_binary'].value_counts())
