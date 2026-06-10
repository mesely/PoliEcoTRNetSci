from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from parser import (
    GLOBAL_MP_DICT,
    DONEM_DICT,
    NON_SPEAKER_TOKENS,
    MANUAL_OVERRIDES,
    build_mp_indices,
    extract_party_from_prefix,
    is_fuzzy_match,
    normalize_for_match,
)


DEFAULT_DATASET_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "Data" / "TBMM_Network_Dataset.csv",
    Path(__file__).resolve().parent / "TBMM_Network_Dataset.csv",
    Path("/content/drive/MyDrive/TBMM_Network_Dataset.csv"),
]
DEFAULT_MP_CSV_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "Data" / "milletvekilleri.csv",
    Path(__file__).resolve().parent / "milletvekilleri.csv",
    Path("/content/drive/MyDrive/milletvekilleri.csv"),
]
NON_MP_ROLES = {"MECLİS BAŞKANLIĞI", "KOMİSYON SÖZCÜSÜ", "Bilinmeyen", "BAKANLAR KURULU"}


def resolve_existing_path(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{label} bulunamadı: {candidates}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=resolve_existing_path(DEFAULT_DATASET_CANDIDATES, "dataset"))
    parser.add_argument("--mp-csv", type=Path, default=resolve_existing_path(DEFAULT_MP_CSV_CANDIDATES, "milletvekilleri.csv"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--inplace", action="store_true")
    return parser.parse_args()


def find_party_and_clean_name_from_dataset(speaker_raw: str, donem: str) -> tuple[str, str]:
    s_norm = normalize_for_match(speaker_raw)
    if s_norm in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[s_norm]
    if "baskan" in s_norm and "basbakan" not in s_norm:
        return "MECLİS BAŞKANLIĞI", "BAŞKAN"

    mps_in_donem = DONEM_DICT.get(donem, {})
    if s_norm in mps_in_donem:
        return mps_in_donem[s_norm]

    for mp_norm, data in mps_in_donem.items():
        if len(mp_norm) > 5 and mp_norm in s_norm:
            return data
        if is_fuzzy_match(s_norm, mp_norm):
            return data

    if s_norm in GLOBAL_MP_DICT:
        return GLOBAL_MP_DICT[s_norm]

    detected_party = extract_party_from_prefix(speaker_raw)
    return detected_party or "Bilinmeyen", str(speaker_raw).upper().strip()


def main() -> None:
    args = parse_args()
    output_path = args.input if args.inplace else (args.output or args.input.with_name(f"{args.input.stem}_fixed.csv"))

    print("📂 Milletvekili indeksleri hazırlanıyor...")
    donem_dict, global_mp_dict = build_mp_indices(args.mp_csv)
    DONEM_DICT.clear()
    DONEM_DICT.update(donem_dict)
    GLOBAL_MP_DICT.clear()
    GLOBAL_MP_DICT.update(global_mp_dict)

    print("📂 Mevcut dataset yükleniyor...")
    df = pd.read_csv(args.input)
    df["donem"] = df["donem"].astype(str).str.replace(".0", "", regex=False)
    df["speaker"] = df["speaker"].fillna("").astype(str)
    df["party"] = df["party"].fillna("").astype(str)

    old_party = df["party"].copy()
    fixed_parties: list[str] = []
    fixed_speakers: list[str] = []

    for row in df[["speaker", "donem", "party"]].itertuples(index=False):
        current_party = row.party.strip() if isinstance(row.party, str) else ""
        speaker = row.speaker.strip() if isinstance(row.speaker, str) else ""
        donem = row.donem

        if not speaker or speaker.upper() in NON_SPEAKER_TOKENS:
            fixed_parties.append(current_party or "Bilinmeyen")
            fixed_speakers.append(speaker)
            continue

        resolved_party, clean_name = find_party_and_clean_name_from_dataset(speaker, donem)

        if current_party in NON_MP_ROLES and resolved_party == "Bilinmeyen":
            fixed_parties.append(current_party)
        else:
            fixed_parties.append(resolved_party)
        fixed_speakers.append(clean_name if clean_name else speaker)

    df["party"] = fixed_parties
    df["speaker"] = fixed_speakers

    changed = int((old_party != df["party"]).sum())
    print(f"🛠️ Güncellenen party satırı: {changed}")
    print("💾 Düzeltilmiş dataset yazılıyor...")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Hazır: {output_path}")


if __name__ == "__main__":
    main()
