from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
DATA_DIR = PROJECT_ROOT / "Data"
DEFAULT_DATASET_CANDIDATES = [
    DATA_DIR / "TBMM_Network_Dataset_partyfixed.csv",
    ROOT_DIR / "TBMM_Network_Dataset_partyfixed.csv",
]
NON_MP_ROLES = {"MECLİS BAŞKANLIĞI", "KOMİSYON SÖZCÜSÜ", "Bilinmeyen", "BAKANLAR KURULU"}
VALID_PARTIES_BY_DONEM = {
    "22": {
        "Adalet ve Kalkınma Partisi",
        "Cumhuriyet Halk Partisi",
        "Anavatan Partisi",
        "Doğru Yol Partisi",
        "Bağımsız",
        "Genç Parti",
        "Saadet Partisi",
        "Demokratik Sol Parti",
    },
    "23": {
        "Adalet ve Kalkınma Partisi",
        "Cumhuriyet Halk Partisi",
        "Milliyetçi Hareket Partisi",
        "Bağımsız",
        "Demokratik Sol Parti",
        "Demokrat Parti",
        "Saadet Partisi",
    },
    "24": {
        "Adalet ve Kalkınma Partisi",
        "Cumhuriyet Halk Partisi",
        "Milliyetçi Hareket Partisi",
        "Halkların Demokratik Partisi",
        "Bağımsız",
    },
    "25": {
        "Adalet ve Kalkınma Partisi",
        "Cumhuriyet Halk Partisi",
        "Milliyetçi Hareket Partisi",
        "Halkların Demokratik Partisi",
        "Bağımsız",
    },
    "26": {
        "Adalet ve Kalkınma Partisi",
        "Cumhuriyet Halk Partisi",
        "Milliyetçi Hareket Partisi",
        "Halkların Demokratik Partisi",
        "İYİ Parti",
        "Saadet Partisi",
        "Bağımsız",
    },
    "27": {
        "Adalet ve Kalkınma Partisi",
        "Cumhuriyet Halk Partisi",
        "Milliyetçi Hareket Partisi",
        "Halkların Demokratik Partisi",
        "İYİ Parti",
        "Bağımsız",
        "Türkiye İşçi Partisi",
        "Demokrat Parti",
        "Büyük Birlik Partisi",
        "Saadet Partisi",
    },
    "28": {
        "Adalet ve Kalkınma Partisi",
        "Cumhuriyet Halk Partisi",
        "Milliyetçi Hareket Partisi",
        "İYİ Parti",
        "Halkların Eşitlik ve Demokrasi Partisi",
        "DEM Parti",
        "YENİ YOL Partisi",
        "Türkiye İşçi Partisi",
        "Yeniden Refah Partisi",
        "Saadet Partisi",
        "Demokratik Sol Parti",
        "Bağımsız",
    },
}


def resolve_existing_path(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{label} bulunamadı: {candidates}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TBMM dataset için hızlı konsol kontrolleri")
    parser.add_argument(
        "--input",
        type=Path,
        default=resolve_existing_path(DEFAULT_DATASET_CANDIDATES, "dataset"),
        help="Kontrol edilecek dataset yolu",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Her dönem için gösterilecek maksimum parti sayısı",
    )
    parser.add_argument(
        "--include-non-mp",
        action="store_true",
        help="Meclis Başkanlığı/Bilinmeyen gibi etiketleri dışlama",
    )
    parser.add_argument(
        "--show-overall",
        action="store_true",
        help="Dönem bazına ek olarak genel parti dağılımını da yazdır",
    )
    parser.add_argument(
        "--fix-invalid-parties",
        action="store_true",
        help="Dönem bazlı geçersiz parti etiketlerini Bilinmeyen yap ve çıktı yaz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Düzeltme modunda yazılacak çıktı yolu",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Düzeltme modunda girdinin üzerine yaz",
    )
    parser.add_argument(
        "--show-invalid",
        action="store_true",
        help="Dönem bazlı whitelist dışında kalan parti etiketlerini yazdır",
    )
    return parser.parse_args()


def normalize_invalid_parties(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    fixed = df.copy()
    invalid_mask = []

    for row in fixed[["donem", "party"]].itertuples(index=False):
        donem = str(row.donem)
        party = str(row.party)
        valid_parties = VALID_PARTIES_BY_DONEM.get(donem, set())
        invalid_mask.append(bool(valid_parties) and party not in valid_parties and party not in NON_MP_ROLES)

    mask = pd.Series(invalid_mask, index=fixed.index)
    changed = int(mask.sum())
    fixed.loc[mask, "party"] = "Bilinmeyen"
    return fixed, changed


def print_invalid_parties(df: pd.DataFrame) -> None:
    print("\nGeçersiz Parti Etiketleri")
    print("========================")
    found_any = False

    donemler = sorted(
        [d for d in df["donem"].dropna().astype(str).unique() if d != "nan"],
        key=lambda x: int(x) if str(x).isdigit() else 999,
    )
    for donem in donemler:
        block = df[df["donem"] == donem].copy()
        valid_parties = VALID_PARTIES_BY_DONEM.get(donem, set())
        invalid_counts = (
            block[~block["party"].isin(valid_parties.union(NON_MP_ROLES))]["party"]
            .value_counts()
        )
        if invalid_counts.empty:
            continue
        found_any = True
        print(f"\n--- {donem}. Dönem ---")
        for party, count in invalid_counts.items():
            print(f"{party} | {int(count)}")

    if not found_any:
        print("Geçersiz parti etiketi bulunmadı.")


def print_period_distributions(df: pd.DataFrame, top_k: int) -> None:
    donemler = sorted(
        [d for d in df["donem"].dropna().astype(str).unique() if d != "nan"],
        key=lambda x: int(x) if str(x).isdigit() else 999,
    )

    print("Dönem Bazlı Parti Dağılımları")
    print("============================")

    for donem in donemler:
        block = df[df["donem"] == donem].copy()
        party_counts = block["party"].fillna("NaN").value_counts().head(top_k)
        speaker_counts = block.groupby("party")["speaker"].nunique().sort_values(ascending=False).head(top_k)

        print(f"\n--- {donem}. Dönem ---")
        print(f"toplam satır: {len(block)}")
        print(f"benzersiz konuşmacı: {block['speaker'].nunique()}")
        print("parti | satır | benzersiz konuşmacı")
        for party, row_count in party_counts.items():
            uniq = int(speaker_counts.get(party, 0))
            print(f"{party} | {int(row_count)} | {uniq}")


def print_overall_distribution(df: pd.DataFrame, top_k: int) -> None:
    party_counts = df["party"].fillna("NaN").value_counts().head(top_k)
    speaker_counts = df.groupby("party")["speaker"].nunique().sort_values(ascending=False).head(top_k)

    print("\nGenel Parti Dağılımı")
    print("====================")
    print("parti | satır | benzersiz konuşmacı")
    for party, row_count in party_counts.items():
        uniq = int(speaker_counts.get(party, 0))
        print(f"{party} | {int(row_count)} | {uniq}")


def main() -> None:
    args = parse_args()
    print(f"📂 Dataset yükleniyor: {args.input}")
    df = pd.read_csv(args.input)
    df["donem"] = df["donem"].astype(str).str.replace(".0", "", regex=False)
    df["party"] = df["party"].fillna("Bilinmeyen").astype(str)
    df["speaker"] = df["speaker"].fillna("").astype(str)

    if args.fix_invalid_parties:
        output_path = args.input if args.inplace else (args.output or args.input.with_name(f"{args.input.stem}_partyfixed.csv"))
        df, changed = normalize_invalid_parties(df)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"🛠️ Bilinmeyen'e çekilen geçersiz party satırı: {changed}")
        print(f"💾 Düzeltme çıktısı: {output_path}")

    if not args.include_non_mp:
        df = df[~df["party"].isin(NON_MP_ROLES)].copy()

    print_period_distributions(df, args.top_k)

    if args.show_overall:
        print_overall_distribution(df, args.top_k)
    if args.show_invalid:
        print_invalid_parties(df)


if __name__ == "__main__":
    main()
