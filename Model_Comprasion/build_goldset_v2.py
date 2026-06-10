"""
build_goldset_v2.py
───────────────────
Orijinal goldset (v1, 108 örnek) + yeni annotasyonlar (v2) birleştirir.
Binary filtreleme uygular: mixed örnekler çıkarılır.
Çıktı: data/goldset_binary_v2.csv

Çalıştır: python3 build_goldset_v2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Yollar ──────────────────────────────────────────────────────────────────
GOLDSET_V1_CSV    = Path(
    "/Users/selmanyilmaz/PoliEcoTRNetworSci/Term_22/CSVs/"
    "term_22_beme_validation_goldset_final.csv"
)
ANNOTATION_V2_CSV = Path(__file__).parent / "data" / "annotation_candidates_v2.csv"
OUT_DIR           = Path(__file__).parent / "data"
OUT_FULL_CSV      = OUT_DIR / "goldset_full_v2.csv"   # 3 sınıf (raw)
OUT_BINARY_CSV    = OUT_DIR / "goldset_binary_v2.csv" # binary (mixed çıkarılmış)

CLASS_ORDER = ["negative", "positive"]


def main():
    # ── v1: Orijinal goldset ─────────────────────────────────────────────────
    v1 = pd.read_csv(GOLDSET_V1_CSV)
    v1["manual_label"] = v1["manual_label"].astype(str).str.strip()
    v1["source"] = "v1_original"
    print(f"v1 goldset : {len(v1)} örnek")
    print(f"  Dağılım  : {v1['manual_label'].value_counts().to_dict()}")

    # ── v2: Yeni annotasyonlar ────────────────────────────────────────────────
    if not ANNOTATION_V2_CSV.exists():
        print(f"\n⚠️  {ANNOTATION_V2_CSV} bulunamadı.")
        print("   Önce çalıştır: python3 sample_annotation_candidates.py")
        print("   Sonra annotasyon yap: python3 annotate_candidates.py")
        print("\n   Sadece v1 goldset ile devam ediliyor...")
        combined = v1.copy()
        combined["source"] = "v1_original"
    else:
        v2_raw = pd.read_csv(ANNOTATION_V2_CSV)
        v2_raw["manual_label"] = v2_raw["manual_label"].astype(str).str.strip()

        # Sadece etiketlenmiş örnekleri al (skip ve boş dışla)
        v2 = v2_raw[
            v2_raw["manual_label"].isin(["negative", "positive", "mixed"])
        ].copy()
        v2["source"] = "v2_new"

        if len(v2) == 0:
            print("\n⚠️  v2'de etiketlenmiş örnek yok. Önce annotate_candidates.py çalıştır.")
            combined = v1.copy()
            combined["source"] = "v1_original"
        else:
            print(f"\nv2 annotasyon: {len(v2)} etiketlenmiş")
            print(f"  Dağılım    : {v2['manual_label'].value_counts().to_dict()}")
            print(f"  Bekleyen   : {(v2_raw['manual_label'].isin(['','skip'])).sum()} (etiketlenmemiş/atlandı)")

            # ── Birleştir ─────────────────────────────────────────────────────
            combined = pd.concat(
                [v1, v2[[c for c in v1.columns if c in v2.columns] + ["source"]]],
                ignore_index=True
            )

    # ── Tam goldset (3 sınıf) ─────────────────────────────────────────────────
    combined["heuristic_label_binary"] = combined["heuristic_score"].apply(
        lambda s: "positive" if float(s) >= 0 else "negative"
    )
    combined.to_csv(OUT_FULL_CSV, index=False)
    print(f"\n✅ Tam goldset (3 sınıf)  : {len(combined)} örnek → {OUT_FULL_CSV.name}")
    print(f"   {combined['manual_label'].value_counts().to_dict()}")

    # ── Binary goldset (mixed çıkar) ──────────────────────────────────────────
    binary = combined[combined["manual_label"].isin(CLASS_ORDER)].copy()
    binary = binary.reset_index(drop=True)
    binary.to_csv(OUT_BINARY_CSV, index=False)
    print(f"\n✅ Binary goldset (mixed çıkarıldı): {len(binary)} örnek → {OUT_BINARY_CSV.name}")
    print(f"   {binary['manual_label'].value_counts().reindex(CLASS_ORDER).to_dict()}")

    # ── Özet istatistikler ────────────────────────────────────────────────────
    neg_count = (binary["manual_label"] == "negative").sum()
    pos_count = (binary["manual_label"] == "positive").sum()
    print(f"\nDenge: {neg_count} negative / {pos_count} positive")
    print(f"Oran : {neg_count/(neg_count+pos_count):.1%} / {pos_count/(neg_count+pos_count):.1%}")

    if pos_count >= 30:
        print(f"\n{'='*55}")
        print(f"  {'✅' if pos_count >= 40 else '⚠️ '} BERT fine-tuning için",
              f"{'yeterli' if pos_count >= 40 else 'sınırda'}: {pos_count} positive")
        print(f"  Nested CV (5-fold): ~{pos_count*4//5} positive per train fold")
        if pos_count >= 40:
            print(f"  → fair_model_comparison.ipynb'i BERT fine-tuning ile çalıştır")
        else:
            print(f"  → Daha fazla annotation önerilir (hedef: 50+ positive)")
        print(f"{'='*55}")
    else:
        print(f"\n⚠️  {pos_count} positive — BERT fine-tuning için az.")
        print(f"   Hedef: 50+ positive. Daha fazla annotation devam et.")

    print(f"\n📄 Sonraki adım: fair_model_comparison.ipynb açıp")
    print(f"   GOLDSET_CSV = 'data/goldset_binary_v2.csv' satırını güncelle.")


if __name__ == "__main__":
    main()
