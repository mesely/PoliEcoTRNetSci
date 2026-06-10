"""
sample_annotation_candidates.py
───────────────────────────────
15K+ candidate havuzundan 200 çeşitli pencere örnekler.
Hedef: goldset v2 için annotation kuyruğu.

Örnekleme stratejisi
────────────────────
  100  heuristic_label='positive'  → beklenen true-positive yüksek
   80  heuristic_label='negative'  → beklenen true-negative yüksek
   20  heuristic 'mixed' ama |score|>=0.05  → sınır vakaları

Her katmanda yıl (2002-2007) ve parti çifti çeşitliliği korunur.
Çıktı: data/annotation_candidates_v2.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Yollar ──────────────────────────────────────────────────────────────────
CANDIDATE_CSV = Path(
    "/Users/selmanyilmaz/PoliEcoTRNetworSci/Term_22/CSVs/"
    "term_22_beme_validation_candidate_windows.csv"
)
GOLDSET_CSV   = Path(
    "/Users/selmanyilmaz/PoliEcoTRNetworSci/Term_22/CSVs/"
    "term_22_beme_validation_goldset_final.csv"
)
OUT_DIR       = Path(__file__).parent / "data"
OUT_CSV       = OUT_DIR / "annotation_candidates_v2.csv"

SEED          = 42
N_POSITIVE    = 100   # heuristic positive katmanı
N_NEGATIVE    = 80    # heuristic negative katmanı
N_BORDER      = 20    # mixed ama |score|>=0.05 → tartışmalı
MIN_LEN       = 30    # min kelime sayısı (çok kısa pencereleri atla)

rng = np.random.default_rng(SEED)


def stratified_sample(df: pd.DataFrame, n: int, strat_cols: list[str]) -> pd.DataFrame:
    """Katmanlara göre örnekle; kota dolmadıysa rastgele tamamla."""
    df = df.copy()
    df["_strat"] = df[strat_cols].astype(str).agg("|".join, axis=1)
    groups = df["_strat"].value_counts()
    per_group = max(1, n // len(groups))
    sampled = (
        df.groupby("_strat", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), per_group), random_state=SEED))
    )
    remaining = df[~df.index.isin(sampled.index)]
    need = n - len(sampled)
    if need > 0 and len(remaining):
        extra = remaining.sample(min(need, len(remaining)), random_state=SEED + 1)
        sampled = pd.concat([sampled, extra])
    return sampled.drop(columns="_strat").sample(frac=1, random_state=SEED).head(n)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cands = pd.read_csv(CANDIDATE_CSV)
    gold  = pd.read_csv(GOLDSET_CSV)

    # ── Ön filtreler ─────────────────────────────────────────────────────────
    already_labeled = set(gold["candidate_id"])
    pool = cands[~cands["candidate_id"].isin(already_labeled)].copy()

    # Kelime sayısı filtresi
    pool["_nwords"] = pool["window_text"].str.split().str.len()
    pool = pool[pool["_nwords"] >= MIN_LEN].drop(columns="_nwords")

    print(f"Toplam havuz (etiketlenmemiş, ≥{MIN_LEN} kelime): {len(pool):,}")
    print(f"  positive : {(pool['heuristic_label']=='positive').sum():,}")
    print(f"  negative : {(pool['heuristic_label']=='negative').sum():,}")
    print(f"  mixed    : {(pool['heuristic_label']=='mixed').sum():,}")

    # ── Katman 1: Heuristic POSITIVE ─────────────────────────────────────────
    pos_pool = pool[pool["heuristic_label"] == "positive"].copy()
    # Güven: score ne kadar yüksekse o kadar güvenilir positive
    pos_pool = pos_pool.sort_values("heuristic_score", ascending=False)
    sample_pos = stratified_sample(pos_pool, N_POSITIVE, strat_cols=["year", "source_party"])
    print(f"\nPositive katmanı örneklendi: {len(sample_pos)}")

    # ── Katman 2: Heuristic NEGATIVE ─────────────────────────────────────────
    neg_pool = pool[pool["heuristic_label"] == "negative"].copy()
    neg_pool = neg_pool.sort_values("heuristic_score", ascending=True)
    sample_neg = stratified_sample(neg_pool, N_NEGATIVE, strat_cols=["year", "source_party"])
    print(f"Negative katmanı örneklendi: {len(sample_neg)}")

    # ── Katman 3: Sınır vakaları (mixed ama |score| yüksek) ─────────────────
    border_pool = pool[
        (pool["heuristic_label"] == "mixed") & (pool["heuristic_score"].abs() >= 0.05)
    ].copy()
    # Eşit + ve - sınır örneği al
    n_each = N_BORDER // 2
    border_pos = border_pool[border_pool["heuristic_score"] >  0.05].sample(
        min(n_each, len(border_pool[border_pool["heuristic_score"] > 0.05])),
        random_state=SEED)
    border_neg = border_pool[border_pool["heuristic_score"] < -0.05].sample(
        min(n_each, len(border_pool[border_pool["heuristic_score"] < -0.05])),
        random_state=SEED)
    sample_brd = pd.concat([border_pos, border_neg])
    print(f"Sınır katmanı örneklendi  : {len(sample_brd)}")

    # ── Birleştir ve annotation sütunlarını ekle ─────────────────────────────
    result = pd.concat([sample_pos, sample_neg, sample_brd], ignore_index=True)
    result = result.sample(frac=1, random_state=SEED).reset_index(drop=True)

    result["manual_label"] = ""    # annotator dolduracak: negative / positive / mixed
    result["manual_notes"] = ""    # opsiyonel not

    # Sütun sırası: annotation için kolay okunabilir
    cols = [
        "candidate_id", "manual_label", "manual_notes",
        "transcript_id", "date", "year", "speaker",
        "source_party", "target_party",
        "heuristic_positive_hits", "heuristic_negative_hits",
        "heuristic_score", "heuristic_label",
        "window_text",
    ]
    result = result[[c for c in cols if c in result.columns]]

    result.to_csv(OUT_CSV, index=False)
    print(f"\n✅ {len(result)} aday kaydedildi → {OUT_CSV}")
    print()
    print("Yıl dağılımı:")
    print(result["year"].value_counts().sort_index().to_string())
    print()
    print("Heuristic etiket dağılımı:")
    print(result["heuristic_label"].value_counts().to_string())
    print()
    print("Sonraki adım: python3 annotate_candidates.py")


if __name__ == "__main__":
    main()
