from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
DATA_DIR = PROJECT_ROOT / "Data"
DATA_PATH_CANDIDATES = [
    DATA_DIR / "TBMM_Network_Dataset_partyfixed.csv",
    DATA_DIR / "TBMM_Network_Dataset.csv",
    ROOT_DIR / "TBMM_Network_Dataset_partyfixed.csv",
    ROOT_DIR / "TBMM_Network_Dataset.csv",
]
DATA_PATH = next((path for path in DATA_PATH_CANDIDATES if path.exists()), DATA_PATH_CANDIDATES[0])
OUTPUT_PATH = DATA_DIR / "data.txt"
NON_MP_ROLES = {"MECLİS BAŞKANLIĞI", "KOMİSYON SÖZCÜSÜ", "Bilinmeyen", "BAKANLAR KURULU"}
VALID_DONEMLER = ["22", "23", "24", "25", "26", "27", "28"]


def fmt_int(value: int) -> str:
    return f"{int(value):,}".replace(",", ".")


def fmt_float(value: float) -> str:
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def section(title: str) -> str:
    return f"\n{title}\n" + "-" * len(title)


def build_overview(df: pd.DataFrame) -> list[str]:
    lines = [section("Genel Ozet")]
    rows, cols = df.shape
    lines.append(f"shape: ({rows}, {cols})")
    lines.append(f"kolonlar: {', '.join(df.columns)}")
    lines.append(f"benzersiz transcript_id: {fmt_int(df['transcript_id'].nunique())}")
    lines.append(f"benzersiz konusmaci: {fmt_int(df['speaker'].nunique())}")
    lines.append(f"benzersiz parti: {fmt_int(df['party'].nunique())}")
    lines.append(f"benzersiz tarih: {fmt_int(df['date'].nunique())}")
    lines.append(f"donemler: {', '.join(sorted(df['donem'].astype(str).unique()))}")
    return lines


def build_missingness(df: pd.DataFrame) -> list[str]:
    lines = [section("Eksik Veri")]
    missing = df.isna().sum().sort_values(ascending=False)
    for column, value in missing.items():
        ratio = (value / len(df)) * 100 if len(df) else 0
        lines.append(f"{column}: {fmt_int(value)} satir, %{fmt_float(ratio)}")
    return lines


def build_length_stats(df: pd.DataFrame) -> list[str]:
    lines = [section("Konusma Uzunlugu")]
    speech = df["speech"].fillna("").astype(str)
    word_count = speech.str.split().str.len()
    char_count = speech.str.len()
    lines.append(f"ortalama kelime: {fmt_float(word_count.mean())}")
    lines.append(f"medyan kelime: {fmt_float(word_count.median())}")
    lines.append(f"p90 kelime: {fmt_float(word_count.quantile(0.90))}")
    lines.append(f"p99 kelime: {fmt_float(word_count.quantile(0.99))}")
    lines.append(f"ortalama karakter: {fmt_float(char_count.mean())}")
    lines.append(f"medyan karakter: {fmt_float(char_count.median())}")
    return lines


def build_donem_summary(df: pd.DataFrame) -> list[str]:
    lines = [section("Donem Bazli Ozet")]
    grouped = (
        df.groupby("donem", dropna=False)
        .agg(
            konusma_sayisi=("speech", "size"),
            benzersiz_konusmaci=("speaker", "nunique"),
            benzersiz_parti=("party", "nunique"),
            benzersiz_transcript=("transcript_id", "nunique"),
            ort_kelime=("word_count", "mean"),
            medyan_kelime=("word_count", "median"),
        )
        .reindex(VALID_DONEMLER)
    )
    for donem, row in grouped.dropna(subset=["konusma_sayisi"]).iterrows():
        lines.append(
            " | ".join(
                [
                    f"donem {donem}",
                    f"konusma={fmt_int(row['konusma_sayisi'])}",
                    f"konusmaci={fmt_int(row['benzersiz_konusmaci'])}",
                    f"parti={fmt_int(row['benzersiz_parti'])}",
                    f"transcript={fmt_int(row['benzersiz_transcript'])}",
                    f"ort_kelime={fmt_float(row['ort_kelime'])}",
                    f"medyan_kelime={fmt_float(row['medyan_kelime'])}",
                ]
            )
        )
    return lines


def build_party_summary(df: pd.DataFrame) -> list[str]:
    lines = [section("En Yuksek Parti Hacmi")]
    top = (
        df.groupby("party")
        .agg(
            konusma_sayisi=("speech", "size"),
            benzersiz_konusmaci=("speaker", "nunique"),
            ort_kelime=("word_count", "mean"),
        )
        .sort_values("konusma_sayisi", ascending=False)
        .head(15)
    )
    for party, row in top.iterrows():
        lines.append(
            f"{party}: konusma={fmt_int(row['konusma_sayisi'])}, "
            f"konusmaci={fmt_int(row['benzersiz_konusmaci'])}, "
            f"ort_kelime={fmt_float(row['ort_kelime'])}"
        )
    return lines


def build_mp_summary(df: pd.DataFrame) -> list[str]:
    lines = [section("Milletvekili Odakli Ozet")]
    df_mps = df[~df["party"].isin(NON_MP_ROLES)].copy()
    lines.append(f"sadece vekil satirlari: {fmt_int(len(df_mps))}")
    lines.append(f"benzersiz vekil: {fmt_int(df_mps['speaker'].nunique())}")

    top_speakers = (
        df_mps.groupby(["speaker", "party"])
        .size()
        .sort_values(ascending=False)
        .head(20)
    )
    lines.append("en aktif 20 vekil:")
    for (speaker, party), count in top_speakers.items():
        lines.append(f"{speaker} [{party}]: {fmt_int(count)}")
    return lines


def build_quality_checks(df: pd.DataFrame) -> list[str]:
    lines = [section("Kalite Kontrolleri")]
    unknown_party = (df["party"] == "Bilinmeyen").sum()
    merkez_city = (df["city"] == "MERKEZ").sum()
    duplicate_rows = df.duplicated(subset=["transcript_id", "speaker", "speech"]).sum()
    lines.append(f"Bilinmeyen parti: {fmt_int(unknown_party)}")
    lines.append(f"MERKEZ city: {fmt_int(merkez_city)}")
    lines.append(f"aynı transcript-speaker-speech tekrarları: {fmt_int(duplicate_rows)}")
    lines.append(f"en kisa konusma kelime: {fmt_int(df['word_count'].min())}")
    lines.append(f"en uzun konusma kelime: {fmt_int(df['word_count'].max())}")
    return lines


def main() -> None:
    print("📂 Veri seti yukleniyor...")
    df = pd.read_csv(DATA_PATH)
    df["donem"] = df["donem"].astype(str).str.replace(".0", "", regex=False)
    df["speech"] = df["speech"].fillna("").astype(str)
    df["word_count"] = df["speech"].str.split().str.len()

    lines: list[str] = []
    lines.extend(build_overview(df))
    lines.extend(build_missingness(df))
    lines.extend(build_length_stats(df))
    lines.extend(build_donem_summary(df))
    lines.extend(build_party_summary(df))
    lines.extend(build_mp_summary(df))
    lines.extend(build_quality_checks(df))

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ Teknik EDA yazildi: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
