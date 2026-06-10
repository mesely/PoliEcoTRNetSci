from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, total=None):  # type: ignore[no-redef]
        return iterable

# --- PATHS ---
ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
DATA_DIR = PROJECT_ROOT / "Data"
DEFAULT_TXT_CANDIDATES = [
    DATA_DIR / "TBMM_Clean_Text",
    ROOT_DIR / "TBMM_Clean_Text",
    ROOT_DIR.parent / "TBMM_Clean_Text",
    Path("/content/drive/MyDrive/TBMM_Clean_Text"),
]
DEFAULT_MP_CSV_CANDIDATES = [
    DATA_DIR / "milletvekilleri.csv",
    ROOT_DIR / "milletvekilleri.csv",
    ROOT_DIR.parent / "milletvekilleri.csv",
    Path("/content/drive/MyDrive/milletvekilleri.csv"),
]
DEFAULT_OUTPUT_CANDIDATES = [
    DATA_DIR / "TBMM_Network_Dataset.csv",
    Path("/content/drive/MyDrive/TBMM_Network_Dataset.csv"),
]

# 1. MANUEL MÜDAHALELER
MANUAL_OVERRIDES = {
    "izzet celik": ("Adalet ve Kalkınma Partisi", "İZZET ÇETİN"),
    "reyhan balandi": ("Adalet ve Kalkınma Partisi", "REYHAN BALANDI"),
    "cicek": ("Adalet ve Kalkınma Partisi", "CEMİL ÇİÇEK"),
    "sidika aydogan": ("Cumhuriyet Halk Partisi", "SIDIKA AYDOĞAN"),
    "feridun fikrat baloglu": ("Cumhuriyet Halk Partisi", "FERİDUN FİKRET BALOĞLU"),
    "fevzi berdibek": ("Adalet ve Kalkınma Partisi", "FEVZİ BERDİBEK"),
    "idris naim sahih": ("Adalet ve Kalkınma Partisi", "İDRİS NAİM ŞAHİN"),
    "baskan": ("MECLİS BAŞKANLIĞI", "BAŞKAN"),
    "basan": ("MECLİS BAŞKANLIĞI", "BAŞKAN"),
}

GARBAGE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*İÇİNDEKİLER\s*$",
        r"^\s*TBMM\s*$",
        r"^\s*TÜRKİYE BÜYÜK MİLLET MECLİSİ",
        r"^\s*\d+\.\s*BİRLEŞİM",
        r"^\s*\d+\.\s*OTURUM",
        r"^\s*Açılma Saati",
        r"^\s*Kapanma Saati",
        r"^\s*Başkanlık Divanı",
        r"^\s*Sayfa\s+\d+",
        r"^\s*GEÇİCİ MADDE",
        r"^\s*EK MADDE",
        r"^\s*YASAMA YILI",
        r"^\s*CİLT\s*[:.]",
        r"^\s*TUTANAK DERGİSİ\s*$",
        r"^\s*DÖNEM\s*[:.]",
        r"^\s*BAŞKAN\s*:\s+",
        r"^\s*KÂTİP ÜYELER\s*:\s+",
        r"^\s*KATİP ÜYELER\s*:\s+",
    )
]
NON_SPEAKER_TOKENS = {
    "BİRİNCİ",
    "İKİNCİ",
    "ÜÇÜNCÜ",
    "DÖRDÜNCÜ",
    "BEŞİNCİ",
    "OTURUM",
    "BİRLEŞİM",
    "TUTANAK",
    "SAYFA",
    "MADDE",
    "KANUN",
    "TEKLİFİ",
    "ÖNERGE",
    "CETVELİ",
    "YASAMA",
    "YILI",
    "CİLT",
    "KÂTİP",
    "ÜYELER",
    "KATİP",
    "TBMM",
}
DATE_RE = re.compile(
    r"(\d{1,2}\s*(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık|\.)\s*(?:\.|[A-Za-z]*)\s*\d{4})"
)
INLINE_SPEAKER_RE = re.compile(
    r"^(?P<speaker>[A-ZÇĞİÖŞÜÂÎÛ][A-ZÇĞİÖŞÜÂÎÛ0-9\.'’/\-\s]{1,180}?)"
    r"(?:\((?P<meta>[^)]{1,120})\))?\s*(?:-|:)\s*(?P<speech>.+?)\s*$"
)
HEADER_ONLY_SPEAKER_RE = re.compile(
    r"^(?P<speaker>[A-ZÇĞİÖŞÜÂÎÛ][A-ZÇĞİÖŞÜÂÎÛ0-9\.'’/\-\s]{1,180}?)"
    r"(?:\((?P<meta>[^)]{1,120})\))?\s*-\s*$"
)
CONTINUATION_SEPARATOR_RE = re.compile(r"^\s*[-:]\s*(.*)$")
TRAILING_PAGE_NO_RE = re.compile(r"\s+\d+\s*$")
SESSION_START_RE = re.compile(r"(?:^|\n)\s*(?:Açılma Saati|Açılma Saati:|BİRİNCİ OTURUM|BİRİNCİ OTURUM\b)", re.IGNORECASE)
WORD_FRAGMENT_RE = re.compile(r"([A-Za-zÇĞİÖŞÜÂÎÛçğıöşüâîû]{1,12})\s*\n\s*\n\s*([A-Za-zÇĞİÖŞÜÂÎÛçğıöşüâîû'][A-Za-zÇĞİÖŞÜÂÎÛçğıöşüâîû']{0,12})")

DONEM_RANGES = [
    (8075, 19940, "22"),
    (19941, 20915, "23"),
    (20982, 22425, "24"),
    (22427, 22449, "25"),
    (22451, 23140, "26"),
    (23142, 23896, "27"),
    (23898, 24237, "28"),
]

TXT_FOLDER: Path | None = None
MP_CSV_PATH: Path | None = None
OUTPUT_FILE: Path | None = None
DONEM_DICT: dict[str, dict[str, tuple[str, str]]] = {}
GLOBAL_MP_DICT: dict[str, tuple[str, str]] = {}


def resolve_existing_path(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"{label} bulunamadı. Aranan yollar:\n{searched}")


def sanitize_content(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\x96": "-",
        "\x97": "-",
        "\xad": "",
        "\u00a0": " ",
        "\u2009": " ",
        "\u200b": "",
        "\u2026": "...",
        "\x85": "...",
        "\u2018": "'",
        "\u2019": "'",
        "\x91": "'",
        "\x92": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\x93": '"',
        "\x94": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    def merge_fragment(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        if right[0].islower():
            return left + right
        if left.isupper() and right.isupper() and len(left) + len(right) <= 24:
            return left + right
        if left[-1].islower() and len(left) <= 6:
            return left + right
        return match.group(0)

    for _ in range(8):
        new_text = WORD_FRAGMENT_RE.sub(merge_fragment, text)
        if new_text == text:
            break
        text = new_text

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def get_donem_from_id(file_id: str) -> str:
    try:
        fid = int(file_id)
    except (TypeError, ValueError):
        return "Hata"

    for start, end, donem in DONEM_RANGES:
        if start <= fid <= end:
            return donem
    return "Bilinmiyor"


def normalize_for_match(text: str) -> str:
    if not text:
        return ""

    t = str(text).upper()
    mapping = str.maketrans("ÇĞİÖŞÜÂÎÛI", "CGIOSUAIUI")
    t = t.translate(mapping)
    t = t.replace(".", " ").replace(",", " ").replace("'", " ").lower()
    t = t.replace("cuneyit", "cuneyt").replace("feramuz", "feramus")
    t = re.sub(r"\bmemet\b", "mehmet", t)
    t = re.sub(r"\bmehmed\b", "mehmet", t)
    t = re.sub(r"\batila\b", "atilla", t)
    prefixes = [
        r".*?grubu adina",
        r".*?şahsi adina",
        r".*?sahsi adina",
        r"hukumet adina",
        r"komisyon adina",
        r"sözcüsü",
        r"sozcusu",
    ]
    for prefix in prefixes:
        t = re.sub(prefix, "", t)
    return " ".join(t.split())


def is_fuzzy_match(t_norm: str, c_norm: str) -> bool:
    t_words, c_words = t_norm.split(), c_norm.split()
    if not t_words or not c_words:
        return False
    set_t, set_c = set(t_words), set(c_words)
    return len(set_t.intersection(set_c)) >= 2 and (set_t.issubset(set_c) or set_c.issubset(set_t))


def extract_party_from_prefix(raw_text: str) -> str | None:
    t = raw_text.upper()
    mapping = {
        "CHP": "Cumhuriyet Halk Partisi",
        "CUMHURİYET HALK": "Cumhuriyet Halk Partisi",
        "AK PARTİ": "Adalet ve Kalkınma Partisi",
        "ADALET VE KALKINMA": "Adalet ve Kalkınma Partisi",
        "MHP": "Milliyetçi Hareket Partisi",
        "MİLLİYETÇİ HAREKET": "Milliyetçi Hareket Partisi",
        "DEM PARTİ": "DEM Parti",
        "HALKLARIN EŞİTLİK": "DEM Parti",
        "İYİ PARTİ": "İYİ Parti",
        "HDP": "Halkların Demokratik Partisi",
        "SAADET": "Saadet Partisi",
        "YENİ YOL": "YENİ YOL Partisi",
    }
    for key, val in mapping.items():
        if key in t:
            return val
    return None


def normalize_party_label(raw_value: Any) -> str | None:
    if raw_value is None:
        return None

    text = str(raw_value).strip()
    if not text or text.lower() == "nan":
        return None

    lowered = text.lower()
    blocked_markers = ("istifa", "ölüm", "olum", "düşmesi", "dusmesi")
    if any(marker in lowered for marker in blocked_markers):
        return None

    if text.isdigit():
        return None

    direct_map = {
        "Adalet ve Kalkınma Partisi": "Adalet ve Kalkınma Partisi",
        "Cumhuriyet Halk Partisi": "Cumhuriyet Halk Partisi",
        "Milliyetçi Hareket Partisi": "Milliyetçi Hareket Partisi",
        "Halkların Demokratik Partisi": "Halkların Demokratik Partisi",
        "Halkların Eşitlik ve Demokrasi Partisi": "Halkların Eşitlik ve Demokrasi Partisi",
        "DEM Parti": "DEM Parti",
        "İYİ Parti": "İYİ Parti",
        "Doğru Yol Partisi": "Doğru Yol Partisi",
        "Anavatan Partisi": "Anavatan Partisi",
        "Demokrat Parti": "Demokrat Parti",
        "Demokratik Sol Parti": "Demokratik Sol Parti",
        "Saadet Partisi": "Saadet Partisi",
        "Yeni Yol": "YENİ YOL Partisi",
        "YENİ YOL": "YENİ YOL Partisi",
        "Türkiye İşçi Partisi": "Türkiye İşçi Partisi",
        "Hür Dava Partisi": "Hür Dava Partisi",
        "Büyük Birlik Partisi": "Büyük Birlik Partisi",
        "Bağımsız": "Bağımsız",
    }
    if text in direct_map:
        return direct_map[text]

    inferred = extract_party_from_prefix(text)
    if inferred:
        return inferred

    if " · " in text or len(text) > 80:
        return None

    return None


def build_mp_indices(mp_csv_path: Path) -> tuple[dict[str, dict[str, tuple[str, str]]], dict[str, tuple[str, str]]]:
    print("📂 Milletvekili listesi ve eşleştirme indeksleri hazırlanıyor...")
    donem_dict = {str(d): {} for d in range(22, 29)}
    global_mp_dict: dict[str, tuple[str, str]] = {}

    with mp_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            donem = str(row["donem"]).strip()
            isim_golge = normalize_for_match(row["isim"])
            raw_party = str(row["parti"]).strip()
            party = normalize_party_label(raw_party)
            if not party:
                if " · " in raw_party or raw_party.isdigit() or len(raw_party) > 80:
                    party = "Bilinmeyen"
                else:
                    party = raw_party

            updated_party = normalize_party_label(row.get("degisiklik"))
            if updated_party:
                party = updated_party

            value = (party, str(row["isim"]))
            if donem in donem_dict:
                donem_dict[donem][isim_golge] = value
            global_mp_dict[isim_golge] = value

    return donem_dict, global_mp_dict


def find_party_and_clean_name(speaker_raw: str, donem: str) -> tuple[str, str]:
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


def looks_like_garbage(line: str) -> bool:
    upper_line = line.upper()
    return any(pattern.search(upper_line) for pattern in GARBAGE_PATTERNS)


def speaker_confidence(candidate: str) -> float:
    letters = [char for char in candidate if char.isalpha()]
    if len(letters) < 3:
        return 0.0
    uppercase_letters = [char for char in letters if char.upper() == char]
    return len(uppercase_letters) / len(letters)


def looks_like_speaker_name(candidate: str) -> bool:
    candidate = TRAILING_PAGE_NO_RE.sub("", candidate.strip())
    if len(candidate) < 3 or len(candidate) > 180:
        return False
    if looks_like_garbage(candidate):
        return False
    bare_candidate = re.sub(r"\([^)]*\)", " ", candidate)
    if speaker_confidence(bare_candidate) < 0.78:
        return False

    cleaned = bare_candidate
    tokens = [token for token in re.split(r"[\s./-]+", cleaned) if token]
    if not tokens or len(tokens) > 18:
        return False
    if len(tokens) <= 3 and all(re.fullmatch(r"[IVXLCM]+", token) for token in tokens):
        return False
    if all(token.isdigit() for token in tokens):
        return False
    if sum(token.isdigit() for token in tokens) > 1:
        return False
    if set(tokens).intersection(NON_SPEAKER_TOKENS) and len(tokens) <= 3:
        return False
    return True


def extract_speaker_line(line: str) -> dict[str, str] | None:
    for regex, variant in ((INLINE_SPEAKER_RE, "inline"), (HEADER_ONLY_SPEAKER_RE, "header_only")):
        match = regex.match(line)
        if not match:
            continue

        speaker = TRAILING_PAGE_NO_RE.sub("", match.group("speaker").strip())
        if not looks_like_speaker_name(speaker):
            continue

        meta = (match.group("meta") or "").strip()
        speech = (match.groupdict().get("speech") or "").strip()
        return {"speaker": speaker, "meta": meta, "speech": speech, "variant": variant}
    return None


def merge_split_speaker_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    idx = 0
    while idx < len(lines):
        current = lines[idx].strip()
        if not current:
            merged.append("")
            idx += 1
            continue

        if idx + 1 < len(lines):
            next_line = lines[idx + 1].strip()
            if looks_like_speaker_name(current):
                continuation = CONTINUATION_SEPARATOR_RE.match(next_line)
                if continuation:
                    merged.append(f"{current} - {continuation.group(1).strip()}".strip())
                    idx += 2
                    continue

        merged.append(current)
        idx += 1
    return merged


def finalize_speech(
    speech_list: list[dict[str, str]],
    file_id: str,
    donem: str,
    tarih: str,
    current_speaker: str | None,
    current_city: str,
    current_speech: list[str],
) -> None:
    if not current_speaker:
        return

    speech_text = " ".join(part.strip() for part in current_speech if part and part.strip())
    speech_text = re.sub(r"\s+", " ", speech_text).strip()
    if len(speech_text) <= 15:
        return

    party, correct_name = find_party_and_clean_name(current_speaker, donem)
    speech_list.append(
        {
            "transcript_id": file_id,
            "donem": donem,
            "date": tarih,
            "speaker": correct_name,
            "city": current_city or "MERKEZ",
            "party": party,
            "speech": speech_text,
        }
    )


def parse_transcript(content: str, file_id: str) -> list[dict[str, str]]:
    content = sanitize_content(content)
    current_donem = get_donem_from_id(file_id)
    tarih_match = DATE_RE.search(content)
    tarih = tarih_match.group(1).replace(" ", "") if tarih_match else "Bilinmiyor"
    session_start = SESSION_START_RE.search(content)
    if session_start:
        content = content[session_start.start():]

    speech_list: list[dict[str, str]] = []
    lines = merge_split_speaker_lines(content.split("\n"))
    current_speaker: str | None = None
    current_city = ""
    current_speech: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or looks_like_garbage(line):
            continue

        speaker_data = extract_speaker_line(line)
        if speaker_data:
            finalize_speech(
                speech_list,
                file_id,
                current_donem,
                tarih,
                current_speaker,
                current_city,
                current_speech,
            )
            current_speaker = speaker_data["speaker"]
            current_city = speaker_data["meta"]
            current_speech = [speaker_data["speech"]] if speaker_data["speech"] else []
            continue

        if current_speaker:
            current_speech.append(line)

    finalize_speech(speech_list, file_id, current_donem, tarih, current_speaker, current_city, current_speech)
    return speech_list


def diagnose_transcript(content: str, file_id: str) -> dict[str, object]:
    content = sanitize_content(content)
    lines = merge_split_speaker_lines(content.split("\n"))
    stats = Counter()
    examples: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        speaker_data = extract_speaker_line(stripped)
        if speaker_data:
            stats[f"speaker_{speaker_data['variant']}"] += 1
            continue
        if looks_like_garbage(stripped):
            stats["garbage"] += 1
            continue
        if looks_like_speaker_name(stripped):
            stats["uppercase_like_unmatched"] += 1
            if len(examples) < 20:
                examples.append(stripped)
        else:
            stats["other"] += 1

    return {
        "transcript_id": file_id,
        "donem": get_donem_from_id(file_id),
        "stats": dict(stats),
        "examples": examples,
    }


def init_worker(txt_folder: str, donem_dict: dict[str, dict[str, tuple[str, str]]], global_mp_dict: dict[str, tuple[str, str]]) -> None:
    global TXT_FOLDER, DONEM_DICT, GLOBAL_MP_DICT
    TXT_FOLDER = Path(txt_folder)
    DONEM_DICT = donem_dict
    GLOBAL_MP_DICT = global_mp_dict


def process_single_file(filename: str) -> list[dict[str, str]]:
    assert TXT_FOLDER is not None
    file_id = Path(filename).stem
    path = TXT_FOLDER / filename
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return parse_transcript(handle.read(), file_id)


def process_for_diagnostics(filename: str) -> dict[str, object]:
    assert TXT_FOLDER is not None
    file_id = Path(filename).stem
    path = TXT_FOLDER / filename
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return diagnose_transcript(handle.read(), file_id)


def detect_chunksize(file_count: int, workers: int) -> int:
    if file_count <= 0:
        return 1
    return max(1, min(64, file_count // max(workers * 4, 1) or 1))


def run_parser(txt_folder: Path, output_file: Path, workers: int | None = None) -> list[dict[str, str]]:
    global TXT_FOLDER
    TXT_FOLDER = txt_folder

    all_files = sorted(file.name for file in txt_folder.iterdir() if file.suffix.lower() == ".txt")
    worker_count = workers or (os.cpu_count() or 2)
    chunksize = detect_chunksize(len(all_files), worker_count)
    print(
        f"🔄 {len(all_files)} adet tutanak işleniyor. "
        f"Workers={worker_count}, chunksize={chunksize}, klasör={txt_folder}"
    )

    final_data: list[dict[str, str]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=init_worker,
        initargs=(str(txt_folder), DONEM_DICT, GLOBAL_MP_DICT),
    ) as executor:
        results = executor.map(process_single_file, all_files, chunksize=chunksize)
        for speech_batch in tqdm(results, total=len(all_files)):
            if speech_batch:
                final_data.extend(speech_batch)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["transcript_id", "donem", "date", "speaker", "city", "party", "speech"],
        )
        writer.writeheader()
        writer.writerows(final_data)

    print("\n✅ İŞLEM TAMAM! DÖNEM DAĞILIMI:")
    donem_counts = Counter(row["donem"] for row in final_data)
    for donem in sorted(donem_counts):
        print(f"{donem}: {donem_counts[donem]}")
    return final_data


def run_diagnostics(txt_folder: Path, output_json: Path, sample_limit: int = 50, workers: int | None = None) -> None:
    global TXT_FOLDER
    TXT_FOLDER = txt_folder

    all_files = sorted(file.name for file in txt_folder.iterdir() if file.suffix.lower() == ".txt")
    worker_count = workers or (os.cpu_count() or 2)
    chunksize = detect_chunksize(len(all_files), worker_count)
    reports: list[dict[str, object]] = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=init_worker,
        initargs=(str(txt_folder), DONEM_DICT, GLOBAL_MP_DICT),
    ) as executor:
        results = executor.map(process_for_diagnostics, all_files, chunksize=chunksize)
        for report in tqdm(results, total=len(all_files)):
            reports.append(report)

    grouped: dict[str, dict[str, object]] = {}
    for report in reports:
        donem = str(report["donem"])
        bucket = grouped.setdefault(
            donem,
            {"files": 0, "speaker_inline": 0, "speaker_header_only": 0, "uppercase_like_unmatched": 0, "examples": []},
        )
        bucket["files"] += 1
        stats = report["stats"]
        bucket["speaker_inline"] += stats.get("speaker_inline", 0)
        bucket["speaker_header_only"] += stats.get("speaker_header_only", 0)
        bucket["uppercase_like_unmatched"] += stats.get("uppercase_like_unmatched", 0)
        if len(bucket["examples"]) < sample_limit:
            room = sample_limit - len(bucket["examples"])
            bucket["examples"].extend(report["examples"][:room])

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(grouped, handle, ensure_ascii=False, indent=2)

    print(f"🧪 Tanı raporu yazıldı: {output_json}")
    for donem in sorted(grouped):
        print(donem, grouped[donem])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt-folder", type=Path)
    parser.add_argument("--mp-csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--diagnostics-json", type=Path, default=None)
    parser.add_argument("--diagnostics-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    global MP_CSV_PATH, OUTPUT_FILE, DONEM_DICT, GLOBAL_MP_DICT
    args = parse_args()

    txt_folder = args.txt_folder or resolve_existing_path(DEFAULT_TXT_CANDIDATES, "TBMM_Clean_Text")
    MP_CSV_PATH = args.mp_csv or resolve_existing_path(DEFAULT_MP_CSV_CANDIDATES, "milletvekilleri.csv")
    OUTPUT_FILE = args.output or DEFAULT_OUTPUT_CANDIDATES[0]
    DONEM_DICT, GLOBAL_MP_DICT = build_mp_indices(MP_CSV_PATH)

    if args.diagnostics_only:
        diagnostics_path = args.diagnostics_json or (DATA_DIR / "diagnostics" / "transcript_diagnostics.json")
        run_diagnostics(txt_folder, diagnostics_path, workers=args.workers)
        return

    run_parser(txt_folder, OUTPUT_FILE, workers=args.workers)

    if args.diagnostics_json:
        run_diagnostics(txt_folder, args.diagnostics_json, workers=args.workers)


if __name__ == "__main__":
    main()
