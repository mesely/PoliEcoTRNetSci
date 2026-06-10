from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "Data").exists() or (candidate / "TBMM_Network_Dataset_partyfixed.csv").exists():
            return candidate
    return start


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
MPLCONFIGDIR = PROJECT_ROOT / "Other" / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = PROJECT_ROOT / "Other" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


DATA_PATH_CANDIDATES = [
    PROJECT_ROOT / "Data" / "TBMM_Network_Dataset_partyfixed.csv",
    PROJECT_ROOT / "Data" / "TBMM_Network_Dataset.csv",
    PROJECT_ROOT / "TBMM_Network_Dataset_partyfixed.csv",
    PROJECT_ROOT / "TBMM_Network_Dataset.csv",
]
DATA_PATH = next((path for path in DATA_PATH_CANDIDATES if path.exists()), DATA_PATH_CANDIDATES[0])

NON_MP_ROLES = {"meclis baskanligi", "komisyon sozcusu", "bilinmeyen", "bakanlar kurulu"}
MIN_SPEECH_WORDS = 6
MIN_SPEECH_CHARS = 40
LOCAL_WINDOW_RADIUS = 90
BEME_THRESHOLD = 0.20

TERM_WINDOWS = {
    "22": (pd.Timestamp("2002-11-01"), pd.Timestamp("2007-07-31")),
    "23": (pd.Timestamp("2007-08-01"), pd.Timestamp("2011-06-30")),
    "24": (pd.Timestamp("2011-07-01"), pd.Timestamp("2015-06-30")),
    "25": (pd.Timestamp("2015-07-01"), pd.Timestamp("2015-11-30")),
    "26": (pd.Timestamp("2015-12-01"), pd.Timestamp("2018-06-30")),
    "27": (pd.Timestamp("2018-07-01"), pd.Timestamp("2023-05-31")),
    "28": (pd.Timestamp("2023-06-01"), pd.Timestamp("2026-12-31")),
}

LAYER_STYLES = {
    "total": {"title": "Total", "edge_color": "#6b7280", "accent": "#334155"},
    "positive": {"title": "Positive / Constructive", "edge_color": "#1f8f72", "accent": "#166b57"},
    "negative": {"title": "Negative / Confrontational", "edge_color": "#c4493d", "accent": "#92342b"},
}

PARTY_BLOCS = {
    "22": {
        "adalet ve kalkinma partisi": "government",
    }
}

PARTY_COLORS = {
    "adalet ve kalkinma partisi": "#F4B400",
    "cumhuriyet halk partisi": "#D7263D",
    "milliyetci hareket partisi": "#A61E4D",
    "anavatan partisi": "#FFD23F",
    "dogru yol partisi": "#0B4F6C",
    "bagimsiz": "#6C757D",
    "halklarin demokratik partisi": "#6A4C93",
    "halklarin esitlik ve demokrasi partisi": "#5B8C5A",
    "dem parti": "#4C956C",
    "iyi parti": "#2E86AB",
    "saadet partisi": "#0E9F6E",
    "demokrat parti": "#4C6FA5",
    "demokratik sol partisi": "#3A86FF",
    "turkiye isci partisi": "#C0392B",
    "buyuk birlik partisi": "#9D2235",
    "yeniden refah partisi": "#2A9D8F",
    "yeni yol partisi": "#A23B72",
    "genc parti": "#2C7A7B",
}

PARTY_DISPLAY_LABELS = {
    "adalet ve kalkinma partisi": "Justice and Development Party",
    "cumhuriyet halk partisi": "Republican People's Party",
    "milliyetci hareket partisi": "Nationalist Movement Party",
    "anavatan partisi": "Motherland Party",
    "dogru yol partisi": "True Path Party",
    "bagimsiz": "Independent",
    "halklarin demokratik partisi": "Peoples' Democratic Party",
    "halklarin esitlik ve demokrasi partisi": "Equality and Democracy Party",
    "dem parti": "DEM Party",
    "iyi parti": "Good Party",
    "saadet partisi": "Felicity Party",
    "demokrat parti": "Democratic Party",
    "demokratik sol partisi": "Democratic Left Party",
    "turkiye isci partisi": "Workers' Party of Turkey",
    "buyuk birlik partisi": "Great Union Party",
    "yeniden refah partisi": "New Welfare Party",
    "yeni yol partisi": "New Path Party",
    "genc parti": "Young Party",
}

PARTY_SHORT_LABELS = {
    "adalet ve kalkinma partisi": "AK Party",
    "cumhuriyet halk partisi": "CHP",
    "milliyetci hareket partisi": "MHP",
    "anavatan partisi": "ANAP",
    "dogru yol partisi": "DYP",
    "bagimsiz": "Independent",
    "halklarin demokratik partisi": "HDP",
    "halklarin esitlik ve demokrasi partisi": "HEDEP",
    "dem parti": "DEM",
    "iyi parti": "Good Party",
    "saadet partisi": "Felicity",
    "demokrat parti": "DP",
    "demokratik sol partisi": "DSP",
    "turkiye isci partisi": "TIP",
    "buyuk birlik partisi": "BBP",
    "yeniden refah partisi": "New Welfare",
    "yeni yol partisi": "New Path",
    "genc parti": "Young Party",
}

NETWORK_SUMMARY_LABELS = {
    "party_projection_density": "Party network density",
    "party_bloc_assortativity": "Bloc assortativity",
    "avg_party_closeness": "Avg. party closeness",
    "avg_concept_closeness": "Avg. concept closeness",
}

CONCEPT_DISPLAY_LABELS = {
    "Demokrasi": "Democracy",
    "Reform": "Reform",
    "Avrupa Birligi": "European Union",
    "Hukuk Devleti": "Rule of Law",
    "Ekonomi": "Economy",
    "IMF": "IMF",
    "Enflasyon": "Inflation",
    "Faiz": "Interest Rates",
    "Guvenlik": "Security",
    "Egemenlik": "Sovereignty",
    "Anayasa": "Constitution",
    "Vesayet": "Tutelage",
}

CONCEPT_COLORS = {
    "Demokrasi": "#3A7CA5",
    "Reform": "#5AA469",
    "Avrupa Birligi": "#6BAED6",
    "Hukuk Devleti": "#4C6FA5",
    "Ekonomi": "#C78B36",
    "IMF": "#A8692A",
    "Enflasyon": "#B35C1E",
    "Faiz": "#8D4F2A",
    "Guvenlik": "#7B5EA7",
    "Egemenlik": "#8E6C8A",
    "Anayasa": "#B03A48",
    "Vesayet": "#7A2835",
}

TURKISH_MONTHS = {
    "ocak": 1,
    "subat": 2,
    "mart": 3,
    "nisan": 4,
    "mayis": 5,
    "haziran": 6,
    "temmuz": 7,
    "agustos": 8,
    "eylul": 9,
    "ekim": 10,
    "kasim": 11,
    "aralik": 12,
}

WORD_RE = re.compile(r"[a-z]+")
MONTH_YEAR_RE = re.compile(r"^(?P<month>\d{1,2})\.(?P<year>\d{4})$")
DAY_MONTH_YEAR_RE = re.compile(r"^(?P<day>\d{1,2})(?P<month>[a-z]+)(?P<year>\d{4})$")
MONTH_NAME_YEAR_RE = re.compile(r"^(?P<month>[a-z]+)(?P<year>\d{4})$")


@dataclass(frozen=True)
class ConceptDefinition:
    slug: str
    label: str
    category: str
    patterns: tuple[str, ...]


CONCEPTS: tuple[ConceptDefinition, ...] = (
    ConceptDefinition("democracy", "Demokrasi", "democratic_reform", (r"\bdemokra\w*", r"\bdemokratikles\w*")),
    ConceptDefinition("reform", "Reform", "democratic_reform", (r"\breform\w*", r"\byeniden yapilan\w*", r"\byapisal donus\w*")),
    ConceptDefinition("european_union", "Avrupa Birligi", "democratic_reform", (r"\bavrupa birligi\b", r"\bab\b", r"\bmuzakere sureci\b", r"\bkatilim ortakligi\b")),
    ConceptDefinition("law_state", "Hukuk Devleti", "democratic_reform", (r"\bhukuk\w*", r"\bhukuk devleti\b", r"\bhukukun ustunlugu\b", r"\byargi\w*")),
    ConceptDefinition("economy", "Ekonomi", "macro_economy", (r"\bekonomi\w*", r"\bekonomik\w*", r"\bkalkinma\w*", r"\bbuyume\w*", r"\bistikrar\w*")),
    ConceptDefinition("imf", "IMF", "macro_economy", (r"\bimf\b", r"\bstand[- ]?by\b", r"\buluslararasi para fonu\b")),
    ConceptDefinition("inflation", "Enflasyon", "macro_economy", (r"\benflasyon\w*", r"\bfiyat istikrar\w*")),
    ConceptDefinition("interest", "Faiz", "macro_economy", (r"\bfaiz\w*",)),
    ConceptDefinition("security", "Guvenlik", "security_conflict", (r"\bguvenlik\w*", r"\bteror\w*", r"\btehdit\w*")),
    ConceptDefinition("sovereignty", "Egemenlik", "security_conflict", (r"\begemenlik\w*", r"\bmilli egemenlik\b", r"\bulusal egemenlik\b", r"\bmilli irade\b")),
    ConceptDefinition("constitution", "Anayasa", "constitutional_conflict", (r"\banayasa\w*", r"\banayasal\w*", r"\bcumhurbaskani secimi\b", r"\b367\b")),
    ConceptDefinition("tutelage", "Vesayet", "constitutional_conflict", (r"\bvesayet\w*", r"\bmuhtira\w*", r"\be[- ]?muhtira\b")),
)

POSITIVE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bdestek\w*",
        r"\bsavun\w*",
        r"\bguclen\w*",
        r"\biyiles\w*",
        r"\bolumlu\w*",
        r"\buyum\w*",
        r"\bcozum\w*",
        r"\bkatil\w*",
        r"\buyelik\w*",
        r"\bilerle\w*",
        r"\bgerekli\w*",
        r"\bonemli\w*",
        r"\bbagimsiz\w*",
        r"\bozgurluk\w*",
        r"\brefah\w*",
        r"\bbasari\w*",
        r"\bkoru\w*",
        r"\byatirim\w*",
        r"\bguvence\w*",
        r"\bsaglam\w*",
        r"\bistikrar\w*",
    )
)

NEGATIVE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bkriz\w*",
        r"\bsorun\w*",
        r"\btehdit\w*",
        r"\byanlis\w*",
        r"\byetersiz\w*",
        r"\bbasarisiz\w*",
        r"\bhukuksuz\w*",
        r"\bantidemokrat\w*",
        r"\bmudahale\w*",
        r"\bbaski\w*",
        r"\bkarsi\w*",
        r"\bredd\w*",
        r"\byolsuzluk\w*",
        r"\brisk\w*",
        r"\bsikinti\w*",
        r"\bcokus\w*",
        r"\bissiz\w*",
        r"\byoksulluk\w*",
        r"\bzarar\w*",
        r"\bengel\w*",
        r"\bdayatma\w*",
        r"\bvesayet\w*",
        r"\bmuhtira\w*",
    )
)

MASTER_CONCEPT_RE = re.compile(
    "|".join(
        (
            r"demokra",
            r"reform",
            r"avrupa birligi",
            r"\bab\b",
            r"hukuk",
            r"yargi",
            r"ekonomi",
            r"ekonomik",
            r"kalkinma",
            r"buyume",
            r"istikrar",
            r"imf",
            r"enflasyon",
            r"faiz",
            r"guvenlik",
            r"teror",
            r"tehdit",
            r"egemenlik",
            r"milli irade",
            r"anayasa",
            r"vesayet",
            r"muhtira",
        )
    )
)


def get_output_dirs(term: str) -> tuple[Path, Path, Path]:
    root = PROJECT_ROOT / f"Term_{term}"
    return root / "CSVs", root / "Figures", root / "Notes"


def reset_output_dirs(term: str) -> tuple[Path, Path, Path]:
    csvs_dir, figures_dir, notes_dir = get_output_dirs(term)
    root = csvs_dir.parent
    if root.exists():
        shutil.rmtree(root)
    csvs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)
    return csvs_dir, figures_dir, notes_dir


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    replacements = {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "i̇": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "â": "a",
        "î": "i",
        "û": "u",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^\w\s\-.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_party_key(party: str) -> str:
    return normalize_text(party)


def normalize_date_key(raw_date: object) -> str:
    text = unicodedata.normalize("NFKC", str(raw_date)).strip().lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", "", text)
    text = normalize_text(text)
    text = text.replace(" ", "")
    return text


def parse_tbmm_date(raw_date: object) -> pd.Timestamp | pd.NaT:
    text = normalize_date_key(raw_date)
    if not text or text == "nan":
        return pd.NaT

    direct = MONTH_YEAR_RE.match(text)
    if direct:
        month = int(direct.group("month"))
        year = int(direct.group("year"))
        if 1 <= month <= 12:
            return pd.Timestamp(year=year, month=month, day=1)

    if re.fullmatch(r"^\d{1,2}\.\d{1,2}\.\d{4}$", text):
        parsed = pd.to_datetime(text, format="%d.%m.%Y", errors="coerce")
        return parsed if pd.notna(parsed) else pd.NaT

    compact = text.replace(".", "")
    day_match = DAY_MONTH_YEAR_RE.match(compact)
    if day_match:
        day = int(day_match.group("day"))
        month_name = day_match.group("month")
        year = int(day_match.group("year"))
        month = TURKISH_MONTHS.get(month_name)
        if month is not None:
            try:
                return pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                return pd.NaT

    month_match = MONTH_NAME_YEAR_RE.match(compact)
    if month_match:
        month_name = month_match.group("month")
        year = int(month_match.group("year"))
        month = TURKISH_MONTHS.get(month_name)
        if month is not None:
            return pd.Timestamp(year=year, month=month, day=1)

    return pd.NaT


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def count_pattern_hits(text: str, patterns: Iterable[re.Pattern[str]]) -> int:
    return sum(len(pattern.findall(text)) for pattern in patterns)


def dominant_party_color(party: str) -> str:
    return PARTY_COLORS.get(normalize_party_key(party), "#6C757D")


def display_party_label(party: str, short: bool = True) -> str:
    normalized = normalize_party_key(party)
    if short:
        return PARTY_SHORT_LABELS.get(normalized, PARTY_DISPLAY_LABELS.get(normalized, party))
    return PARTY_DISPLAY_LABELS.get(normalized, party)


def display_concept_label(label: str) -> str:
    return CONCEPT_DISPLAY_LABELS.get(str(label), str(label))


def wrap_display_label(label: str, width: int = 20) -> str:
    return fill(display_concept_label(str(label)), width=width)


def party_bloc(term: str, party: str) -> str:
    return PARTY_BLOCS.get(term, {}).get(normalize_party_key(party), "opposition")


def clipped_edge_strength(score: float, layer: str) -> float:
    if layer == "positive":
        return max(score, 0.0)
    if layer == "negative":
        return abs(min(score, 0.0))
    return abs(score)


def beme_label(score: float) -> str:
    if score >= BEME_THRESHOLD:
        return "positive"
    if score <= -BEME_THRESHOLD:
        return "negative"
    return "mixed"


def build_compiled_concepts() -> dict[str, dict[str, object]]:
    compiled: dict[str, dict[str, object]] = {}
    for concept in CONCEPTS:
        compiled[concept.slug] = {
            "definition": concept,
            "patterns": tuple(re.compile(pattern) for pattern in concept.patterns),
        }
    return compiled


COMPILED_CONCEPTS = build_compiled_concepts()


def score_speech_edges(
    speech_id: str,
    transcript_id: str,
    parsed_date: pd.Timestamp,
    speaker: str,
    party: str,
    normalized_speech: str,
) -> list[dict[str, object]]:
    if not MASTER_CONCEPT_RE.search(normalized_speech):
        return []

    rows: list[dict[str, object]] = []
    for payload in COMPILED_CONCEPTS.values():
        definition: ConceptDefinition = payload["definition"]  # type: ignore[assignment]
        patterns: tuple[re.Pattern[str], ...] = payload["patterns"]  # type: ignore[assignment]
        windows: list[str] = []
        mention_count = 0
        for pattern in patterns:
            for match in pattern.finditer(normalized_speech):
                mention_count += 1
                start = max(0, match.start() - LOCAL_WINDOW_RADIUS)
                end = min(len(normalized_speech), match.end() + LOCAL_WINDOW_RADIUS)
                windows.append(normalized_speech[start:end])

        if mention_count == 0:
            continue

        positive_hits = sum(count_pattern_hits(window, POSITIVE_PATTERNS) for window in windows)
        negative_hits = sum(count_pattern_hits(window, NEGATIVE_PATTERNS) for window in windows)
        salience_weight = mention_count + (0.35 * (positive_hits + negative_hits))
        score = (positive_hits - negative_hits) / (positive_hits + negative_hits + 1)
        rows.append(
            {
                "speech_id": speech_id,
                "transcript_id": transcript_id,
                "date": parsed_date.date().isoformat(),
                "year": parsed_date.year,
                "month": parsed_date.month,
                "speaker": speaker,
                "party": party,
                "concept_slug": definition.slug,
                "concept": definition.label,
                "concept_category": definition.category,
                "mention_count": mention_count,
                "beme_positive_hits": positive_hits,
                "beme_negative_hits": negative_hits,
                "salience_weight": round(salience_weight, 4),
                "beme_score": round(score, 4),
                "beme_label": beme_label(score),
            }
        )
    return rows


def load_term_beme_edges(term: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    usecols = ["transcript_id", "donem", "date", "speaker", "party", "speech"]
    start_date, end_date = TERM_WINDOWS[term]
    speech_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    speech_counter = 0
    total_rows_seen = 0
    filtered_out_of_window = 0
    filtered_non_mp = 0
    filtered_short = 0

    for chunk in pd.read_csv(DATA_PATH, usecols=usecols, chunksize=100000, low_memory=False):
        total_rows_seen += int(len(chunk))
        chunk["donem"] = chunk["donem"].astype(str).str.replace(".0", "", regex=False)
        chunk = chunk[chunk["donem"] == term].copy()
        if chunk.empty:
            continue

        chunk["speaker"] = chunk["speaker"].fillna("").astype(str).str.strip()
        chunk["party"] = chunk["party"].fillna("").astype(str).str.strip()
        chunk["speech"] = chunk["speech"].fillna("").astype(str)
        chunk["party_key"] = chunk["party"].map(normalize_party_key)
        mask_non_mp = chunk["party_key"].isin(NON_MP_ROLES)
        filtered_non_mp += int(mask_non_mp.sum())
        chunk = chunk[~mask_non_mp].copy()
        if chunk.empty:
            continue

        chunk["normalized_speech"] = chunk["speech"].map(normalize_text)
        chunk["parsed_date"] = chunk["date"].map(parse_tbmm_date)
        mask_outside = chunk["parsed_date"].isna() | (chunk["parsed_date"] < start_date) | (chunk["parsed_date"] > end_date)
        filtered_out_of_window += int(mask_outside.sum())
        chunk = chunk[~mask_outside].copy()
        if chunk.empty:
            continue

        chunk["word_count"] = chunk["normalized_speech"].map(count_words)
        mask_short = (chunk["word_count"] < MIN_SPEECH_WORDS) | (chunk["normalized_speech"].str.len() < MIN_SPEECH_CHARS)
        filtered_short += int(mask_short.sum())
        chunk = chunk[~mask_short].copy()
        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):
            speech_id = f"term_{term}_speech_{speech_counter:06d}"
            speech_counter += 1
            concept_edges = score_speech_edges(
                speech_id=speech_id,
                transcript_id=str(row.transcript_id),
                parsed_date=row.parsed_date,
                speaker=row.speaker,
                party=row.party,
                normalized_speech=row.normalized_speech,
            )
            if not concept_edges:
                continue

            speech_positive_hits = int(sum(edge["beme_positive_hits"] for edge in concept_edges))
            speech_negative_hits = int(sum(edge["beme_negative_hits"] for edge in concept_edges))
            speech_score = (speech_positive_hits - speech_negative_hits) / (speech_positive_hits + speech_negative_hits + 1)
            speech_rows.append(
                {
                    "speech_id": speech_id,
                    "transcript_id": str(row.transcript_id),
                    "date": row.parsed_date.date().isoformat(),
                    "year": row.parsed_date.year,
                    "month": row.parsed_date.month,
                    "speaker": row.speaker,
                    "party": row.party,
                    "party_bloc": party_bloc(term, row.party),
                    "word_count": int(row.word_count),
                    "concept_hit_count": int(len(concept_edges)),
                    "beme_positive_hits": speech_positive_hits,
                    "beme_negative_hits": speech_negative_hits,
                    "beme_score": round(speech_score, 4),
                    "beme_label": beme_label(speech_score),
                }
            )
            edge_rows.extend(concept_edges)

    speech_df = pd.DataFrame(speech_rows)
    edge_df = pd.DataFrame(edge_rows)
    metadata = {
        "term": term,
        "window_start": start_date.date().isoformat(),
        "window_end": end_date.date().isoformat(),
        "raw_rows_seen": total_rows_seen,
        "speeches_with_concepts": int(len(speech_df)),
        "edge_rows": int(len(edge_df)),
        "filtered_non_mp": filtered_non_mp,
        "filtered_out_of_window": filtered_out_of_window,
        "filtered_short": filtered_short,
        "available_years": sorted(edge_df["year"].dropna().astype(int).unique().tolist()) if not edge_df.empty else [],
    }
    return speech_df, edge_df, metadata


def aggregate_layer_edges(edge_df: pd.DataFrame, layer: str) -> pd.DataFrame:
    if layer == "total":
        layer_df = edge_df.copy()
    elif layer == "positive":
        layer_df = edge_df[edge_df["beme_label"] == "positive"].copy()
    elif layer == "negative":
        layer_df = edge_df[edge_df["beme_label"] == "negative"].copy()
    else:
        raise ValueError(f"Unknown layer: {layer}")

    if layer_df.empty:
        return pd.DataFrame(
            columns=[
                "layer",
                "party",
                "concept_slug",
                "concept",
                "concept_category",
                "speech_count",
                "mention_count",
                "beme_positive_hits",
                "beme_negative_hits",
                "salience_weight",
                "layer_weight",
                "beme_score",
            ]
        )

    layer_df["layer_weight_component"] = layer_df.apply(
        lambda row: row["salience_weight"] * clipped_edge_strength(float(row["beme_score"]), layer),
        axis=1,
    )
    if layer == "total":
        layer_df["layer_weight_component"] = layer_df["salience_weight"]

    grouped = (
        layer_df.groupby(["party", "concept_slug", "concept", "concept_category"], dropna=False)
        .agg(
            speech_count=("speech_id", "nunique"),
            mention_count=("mention_count", "sum"),
            beme_positive_hits=("beme_positive_hits", "sum"),
            beme_negative_hits=("beme_negative_hits", "sum"),
            salience_weight=("salience_weight", "sum"),
            layer_weight=("layer_weight_component", "sum"),
        )
        .reset_index()
    )
    grouped["beme_score"] = (
        (grouped["beme_positive_hits"] - grouped["beme_negative_hits"])
        / (grouped["beme_positive_hits"] + grouped["beme_negative_hits"] + 1)
    )
    grouped["layer"] = layer
    return grouped.sort_values(["layer_weight", "salience_weight"], ascending=False).reset_index(drop=True)


def aggregate_yearly_layer_edges(edge_df: pd.DataFrame, layer: str) -> pd.DataFrame:
    if layer == "total":
        layer_df = edge_df.copy()
    elif layer == "positive":
        layer_df = edge_df[edge_df["beme_label"] == "positive"].copy()
    elif layer == "negative":
        layer_df = edge_df[edge_df["beme_label"] == "negative"].copy()
    else:
        raise ValueError(f"Unknown layer: {layer}")

    if layer_df.empty:
        return pd.DataFrame()

    layer_df["layer_weight_component"] = layer_df.apply(
        lambda row: row["salience_weight"] * clipped_edge_strength(float(row["beme_score"]), layer),
        axis=1,
    )
    if layer == "total":
        layer_df["layer_weight_component"] = layer_df["salience_weight"]

    grouped = (
        layer_df.groupby(["year", "party", "concept_slug", "concept", "concept_category"], dropna=False)
        .agg(
            speech_count=("speech_id", "nunique"),
            mention_count=("mention_count", "sum"),
            beme_positive_hits=("beme_positive_hits", "sum"),
            beme_negative_hits=("beme_negative_hits", "sum"),
            salience_weight=("salience_weight", "sum"),
            layer_weight=("layer_weight_component", "sum"),
        )
        .reset_index()
    )
    grouped["beme_score"] = (
        (grouped["beme_positive_hits"] - grouped["beme_negative_hits"])
        / (grouped["beme_positive_hits"] + grouped["beme_negative_hits"] + 1)
    )
    grouped["layer"] = layer
    return grouped.sort_values(["year", "layer_weight"], ascending=[True, False]).reset_index(drop=True)


def build_projection_matrix(edge_df: pd.DataFrame) -> pd.DataFrame:
    if edge_df.empty:
        return pd.DataFrame()
    matrix = (
        edge_df.pivot_table(
            index="party",
            columns="concept",
            values="layer_weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
        .sort_index(axis=1)
    )
    return matrix


def cosine_similarity_adjacency(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame(index=matrix.index, columns=matrix.index)
    values = matrix.to_numpy(dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    adjacency = normalized @ normalized.T
    np.fill_diagonal(adjacency, 0.0)
    return pd.DataFrame(adjacency, index=matrix.index, columns=matrix.index)


def prune_similarity_adjacency(adjacency: pd.DataFrame, keep_ratio: float) -> pd.DataFrame:
    if adjacency.empty:
        return adjacency.copy()
    if len(adjacency.index) <= 2:
        return adjacency.copy()

    working = adjacency.copy().astype(float)
    labels = working.index.tolist()
    all_edges: list[tuple[str, str, float]] = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            weight = float(working.iat[left, right])
            if weight > 0.0:
                all_edges.append((labels[left], labels[right], weight))

    if not all_edges:
        return working * 0.0

    helper = nx.Graph()
    helper.add_weighted_edges_from(all_edges, weight="weight")
    selected_edges = {
        tuple(sorted((left, right)))
        for left, right in nx.maximum_spanning_tree(helper, weight="weight").edges()
    }

    target_edges = max(len(labels) - 1, math.ceil(len(all_edges) * keep_ratio))
    for left, right, _ in sorted(all_edges, key=lambda item: item[2], reverse=True):
        selected_edges.add(tuple(sorted((left, right))))
        if len(selected_edges) >= target_edges:
            break

    pruned = pd.DataFrame(0.0, index=working.index, columns=working.columns)
    for left, right, weight in all_edges:
        key = tuple(sorted((left, right)))
        if key not in selected_edges:
            continue
        pruned.loc[left, right] = weight
        pruned.loc[right, left] = weight
    return pruned


def build_weighted_graph(adjacency: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    if adjacency.empty:
        return graph

    for node in adjacency.index:
        graph.add_node(node)

    values = adjacency.to_numpy(dtype=float)
    labels = adjacency.index.tolist()
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            weight = float(values[left, right])
            if weight <= 0.0:
                continue
            graph.add_edge(labels[left], labels[right], weight=weight, distance=(1.0 / weight))
    return graph


def safe_eigenvector_centrality(graph: nx.Graph) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_edges() == 0:
        return {node: 0.0 for node in graph.nodes}
    try:
        return nx.eigenvector_centrality(graph, weight="weight", max_iter=2000)
    except Exception:
        return nx.eigenvector_centrality_numpy(graph, weight="weight")


def metric_series(graph: nx.Graph, metric: str) -> pd.Series:
    if graph.number_of_nodes() == 0:
        return pd.Series(dtype=float)
    if metric == "closeness":
        values = nx.closeness_centrality(graph, distance="distance", wf_improved=True)
    elif metric == "betweenness":
        values = nx.betweenness_centrality(graph, weight="distance", normalized=True)
    elif metric == "eigenvector":
        values = safe_eigenvector_centrality(graph)
    else:
        raise ValueError(metric)
    return pd.Series(values, dtype=float)


def build_party_metrics(term: str, edge_df: pd.DataFrame, layer: str) -> pd.DataFrame:
    matrix = build_projection_matrix(edge_df)
    if matrix.empty:
        return pd.DataFrame(columns=["layer", "party", "party_bloc", "weighted_degree", "closeness_centrality", "betweenness_centrality", "eigenvector_centrality"])
    adjacency = prune_similarity_adjacency(cosine_similarity_adjacency(matrix), keep_ratio=0.50)
    graph = build_weighted_graph(adjacency)
    closeness = metric_series(graph, "closeness").rename("closeness_centrality")
    betweenness = metric_series(graph, "betweenness").rename("betweenness_centrality")
    eigenvector = metric_series(graph, "eigenvector").rename("eigenvector_centrality")
    degree = matrix.sum(axis=1).rename("weighted_degree")
    metrics = pd.concat([degree, closeness, betweenness, eigenvector], axis=1).reset_index()
    metrics = metrics.rename(columns={"index": "party"})
    metrics["party_bloc"] = metrics["party"].map(lambda party: party_bloc(term, party))
    metrics["layer"] = layer
    return metrics.fillna(0.0).sort_values(["weighted_degree", "closeness_centrality"], ascending=False).reset_index(drop=True)


def build_concept_metrics(edge_df: pd.DataFrame, layer: str) -> pd.DataFrame:
    matrix = build_projection_matrix(edge_df)
    if matrix.empty:
        return pd.DataFrame(columns=["layer", "concept", "concept_category", "weighted_degree", "closeness_centrality", "betweenness_centrality", "eigenvector_centrality"])
    concept_matrix = matrix.T
    adjacency = prune_similarity_adjacency(cosine_similarity_adjacency(concept_matrix), keep_ratio=0.35)
    graph = build_weighted_graph(adjacency)
    closeness = metric_series(graph, "closeness").rename("closeness_centrality")
    betweenness = metric_series(graph, "betweenness").rename("betweenness_centrality")
    eigenvector = metric_series(graph, "eigenvector").rename("eigenvector_centrality")
    degree = concept_matrix.sum(axis=1).rename("weighted_degree")
    category_map = {concept.label: concept.category for concept in CONCEPTS}
    metrics = pd.concat([degree, closeness, betweenness, eigenvector], axis=1).reset_index()
    metrics = metrics.rename(columns={"index": "concept"})
    metrics["concept_category"] = metrics["concept"].map(category_map)
    metrics["layer"] = layer
    return metrics.fillna(0.0).sort_values(["weighted_degree", "closeness_centrality"], ascending=False).reset_index(drop=True)


def weighted_average(values: pd.Series) -> float:
    return float(values.mean()) if not values.empty else 0.0


def graph_density(graph: nx.Graph) -> float:
    if graph.number_of_nodes() <= 1:
        return 0.0
    return float(nx.density(graph))


def average_shortest_path_largest_component(graph: nx.Graph) -> float:
    if graph.number_of_nodes() <= 1 or graph.number_of_edges() == 0:
        return 0.0
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    if not components:
        return 0.0
    largest = graph.subgraph(components[0]).copy()
    if largest.number_of_nodes() <= 1:
        return 0.0
    try:
        return float(nx.average_shortest_path_length(largest, weight="distance"))
    except Exception:
        return 0.0


def bloc_assortativity(term: str, graph: nx.Graph) -> float:
    if graph.number_of_edges() == 0:
        return 0.0
    for node in graph.nodes:
        graph.nodes[node]["bloc"] = party_bloc(term, node)
    try:
        return float(nx.attribute_assortativity_coefficient(graph, "bloc"))
    except Exception:
        return 0.0


def build_network_summary(term: str, edge_df: pd.DataFrame, layer: str, party_metrics: pd.DataFrame, concept_metrics: pd.DataFrame) -> pd.DataFrame:
    matrix = build_projection_matrix(edge_df)
    if matrix.empty:
        return pd.DataFrame(
            [
                {
                    "layer": layer,
                    "party_nodes": 0,
                    "concept_nodes": 0,
                    "bipartite_edges": 0,
                    "party_projection_edges": 0,
                    "concept_projection_edges": 0,
                    "party_projection_density": 0.0,
                    "concept_projection_density": 0.0,
                    "party_bloc_assortativity": 0.0,
                    "avg_party_closeness": 0.0,
                    "avg_party_betweenness": 0.0,
                    "avg_concept_closeness": 0.0,
                    "avg_concept_betweenness": 0.0,
                    "party_avg_path_lcc": 0.0,
                    "concept_avg_path_lcc": 0.0,
                }
            ]
        )

    party_adjacency = prune_similarity_adjacency(cosine_similarity_adjacency(matrix), keep_ratio=0.50)
    concept_adjacency = prune_similarity_adjacency(cosine_similarity_adjacency(matrix.T), keep_ratio=0.35)
    party_graph = build_weighted_graph(party_adjacency)
    concept_graph = build_weighted_graph(concept_adjacency)
    return pd.DataFrame(
        [
            {
                "layer": layer,
                "party_nodes": int(matrix.shape[0]),
                "concept_nodes": int(matrix.shape[1]),
                "bipartite_edges": int(len(edge_df)),
                "party_projection_edges": int(party_graph.number_of_edges()),
                "concept_projection_edges": int(concept_graph.number_of_edges()),
                "party_projection_density": graph_density(party_graph),
                "concept_projection_density": graph_density(concept_graph),
                "party_bloc_assortativity": bloc_assortativity(term, party_graph),
                "avg_party_closeness": weighted_average(party_metrics["closeness_centrality"]),
                "avg_party_betweenness": weighted_average(party_metrics["betweenness_centrality"]),
                "avg_concept_closeness": weighted_average(concept_metrics["closeness_centrality"]),
                "avg_concept_betweenness": weighted_average(concept_metrics["betweenness_centrality"]),
                "party_avg_path_lcc": average_shortest_path_largest_component(party_graph),
                "concept_avg_path_lcc": average_shortest_path_largest_component(concept_graph),
            }
        ]
    )


def select_display_edges(edge_df: pd.DataFrame, top_per_concept: int = 3, max_edges: int = 30) -> pd.DataFrame:
    if edge_df.empty:
        return edge_df.copy()
    by_weight = edge_df.sort_values("layer_weight", ascending=False)
    display = (
        by_weight.groupby("concept", group_keys=False)
        .head(top_per_concept)
        .drop_duplicates(subset=["party", "concept"])
        .sort_values("layer_weight", ascending=False)
        .head(max_edges)
        .reset_index(drop=True)
    )
    return display


def draw_bipartite_panel(
    ax: plt.Axes,
    edge_df: pd.DataFrame,
    layer: str,
    subtitle: str,
    *,
    node_scale: float = 1.0,
    label_scale: float = 1.0,
) -> None:
    ax.set_facecolor("#f7f5ef")
    if edge_df.empty:
        ax.axis("off")
        ax.set_title(f"{subtitle}\nNo data", fontsize=13, fontweight="bold")
        return

    display = select_display_edges(edge_df)
    parties = display.groupby("party")["layer_weight"].sum().sort_values(ascending=False).index.tolist()
    concepts = display.groupby("concept")["layer_weight"].sum().sort_values(ascending=False).index.tolist()
    if not parties or not concepts:
        ax.axis("off")
        ax.set_title(f"{subtitle}\nNo data", fontsize=13, fontweight="bold")
        return

    party_y = np.linspace(0.95, 0.05, len(parties))
    concept_y = np.linspace(0.95, 0.05, len(concepts))
    positions: dict[str, tuple[float, float]] = {}
    for idx, concept in enumerate(concepts):
        positions[f"concept::{concept}"] = (0.20, float(concept_y[idx]))
    for idx, party in enumerate(parties):
        positions[f"party::{party}"] = (0.80, float(party_y[idx]))

    max_weight = float(display["layer_weight"].max()) if not display.empty else 1.0
    cmap = plt.cm.RdYlGn

    for row in display.itertuples(index=False):
        source = positions[f"concept::{row.concept}"]
        target = positions[f"party::{row.party}"]
        width = 0.8 + (5.0 * float(row.layer_weight) / max_weight)
        alpha = 0.18 + 0.55 * float(row.layer_weight) / max_weight
        if layer == "total":
            color = cmap((float(row.beme_score) + 1.0) / 2.0)
        else:
            color = LAYER_STYLES[layer]["edge_color"]
        ax.plot([source[0], target[0]], [source[1], target[1]], color=color, linewidth=width, alpha=alpha, zorder=1)

    concept_sizes = display.groupby("concept")["layer_weight"].sum()
    party_sizes = display.groupby("party")["layer_weight"].sum()
    max_concept_total = float(concept_sizes.max()) if not concept_sizes.empty else 1.0
    max_party_total = float(party_sizes.max()) if not party_sizes.empty else 1.0
    concept_density_scale = max(0.66, 8.0 / max(len(concepts), 8))
    party_density_scale = max(0.72, 6.0 / max(len(parties), 6))
    concept_fontsize = (8.9 if len(concepts) >= 10 else 9.8) * label_scale
    party_fontsize = (9.8 if len(parties) >= 6 else 10.4) * label_scale

    for concept in concepts:
        size = (((380 + 1320 * float(concept_sizes[concept]) / max_concept_total) * concept_density_scale) / 1.5) * node_scale
        ax.scatter(*positions[f"concept::{concept}"], s=size, color=CONCEPT_COLORS.get(concept, "#64748b"), edgecolor="#ffffff", linewidth=1.6, zorder=3)
        ax.text(0.09, positions[f"concept::{concept}"][1], wrap_display_label(concept, width=16), ha="right", va="center", fontsize=concept_fontsize, fontweight="bold", color="#1f2937")

    for party in parties:
        size = (((405 + 1450 * float(party_sizes[party]) / max_party_total) * party_density_scale) / 1.5) * node_scale
        ax.scatter(*positions[f"party::{party}"], s=size, color=dominant_party_color(party), edgecolor="#ffffff", linewidth=1.6, zorder=3)
        ax.text(0.89, positions[f"party::{party}"][1], display_party_label(party, short=True), ha="left", va="center", fontsize=party_fontsize, fontweight="bold", color="#1f2937")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title(subtitle, fontsize=13, fontweight="bold", color=LAYER_STYLES[layer]["accent"])


def plot_beme_distribution(speech_df: pd.DataFrame, output_path: Path, term: str) -> None:
    if speech_df.empty:
        return

    counts = speech_df["beme_label"].value_counts().reindex(["positive", "mixed", "negative"], fill_value=0).reset_index()
    counts.columns = ["beme_label", "speech_count"]
    colors = {"positive": "#1f8f72", "mixed": "#8a8f98", "negative": "#c4493d"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#f7f5ef")

    sns.barplot(data=counts, x="beme_label", y="speech_count", hue="beme_label", palette=colors, legend=False, ax=axes[0])
    axes[0].set_title("BEME Label Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Speech count")
    axes[0].grid(axis="y", alpha=0.2)

    sns.histplot(data=speech_df, x="beme_score", bins=35, color="#334155", ax=axes[1])
    axes[1].axvline(BEME_THRESHOLD, color="#1f8f72", linestyle="--", linewidth=1.8)
    axes[1].axvline(-BEME_THRESHOLD, color="#c4493d", linestyle="--", linewidth=1.8)
    axes[1].set_title("BEME Score Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("BEME score")
    axes[1].set_ylabel("Speech count")
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle(f"Term {term} | Stage 1 BEME Breakdown", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_layered_bipartite_networks(layer_edges: pd.DataFrame, output_path: Path, term: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(23, 8))
    fig.patch.set_facecolor("#f7f5ef")
    for ax, layer in zip(axes, ["total", "positive", "negative"]):
        subset = layer_edges[layer_edges["layer"] == layer].copy()
        draw_bipartite_panel(ax, subset, layer, LAYER_STYLES[layer]["title"])
    fig.suptitle(f"Term {term} | Stage 2 Layered Party-Concept Networks", fontsize=19, fontweight="bold", y=0.98)
    fig.text(0.5, 0.03, "Edge width shows BEME-weighted salience intensity.", ha="center", fontsize=11, color="#475569")
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.95])
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def top_n_per_layer(metric_df: pd.DataFrame, label_col: str, value_col: str, n: int = 5) -> pd.DataFrame:
    if metric_df.empty:
        return metric_df.copy()
    return (
        metric_df.sort_values(["layer", value_col], ascending=[True, False])
        .groupby("layer", group_keys=False)
        .head(n)
        .copy()
    )


def plot_layered_metric_dots(
    ax: plt.Axes,
    metric_df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    explanation: str,
    xlabel: str,
    label_formatter,
) -> None:
    if metric_df.empty:
        ax.axis("off")
        return

    plot_df = metric_df.copy()
    plot_df["display_label"] = plot_df[label_col].map(label_formatter)
    ordering = (
        plot_df.groupby("display_label")[value_col]
        .max()
        .sort_values(ascending=True)
        .index
        .tolist()
    )
    y_positions = {label: idx for idx, label in enumerate(ordering)}
    layer_offsets = {"positive": -0.18, "total": 0.0, "negative": 0.18}

    for layer in ["total", "positive", "negative"]:
        subset = plot_df[plot_df["layer"] == layer].copy()
        if subset.empty:
            continue
        y_values = [y_positions[label] + layer_offsets[layer] for label in subset["display_label"]]
        ax.hlines(
            y=y_values,
            xmin=0.0,
            xmax=subset[value_col],
            color=LAYER_STYLES[layer]["edge_color"],
            alpha=0.22,
            linewidth=2.0,
        )
        ax.scatter(
            subset[value_col],
            y_values,
            s=95,
            color=LAYER_STYLES[layer]["edge_color"],
            edgecolor="#ffffff",
            linewidth=0.8,
            label=LAYER_STYLES[layer]["title"],
            zorder=3,
        )

    ax.set_yticks(range(len(ordering)))
    ax.set_yticklabels(ordering)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=14)
    ax.text(
        0.01,
        0.98,
        explanation,
        transform=ax.transAxes,
        fontsize=8.8,
        color="#475569",
        va="top",
        bbox={"facecolor": "#f7f5ef", "edgecolor": "none", "pad": 0.2, "alpha": 0.9},
    )
    ax.set_xlim(-0.02, max(0.05, float(plot_df[value_col].max()) * 1.18))
    ax.grid(axis="x", alpha=0.2)


def plot_metric_dashboard(
    party_metrics: pd.DataFrame,
    concept_metrics: pd.DataFrame,
    network_summary: pd.DataFrame,
    output_path: Path,
    term: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor("#f7f5ef")

    layer_palette = {layer: style["edge_color"] for layer, style in LAYER_STYLES.items()}

    plot_layered_metric_dots(
        axes[0, 0],
        party_metrics,
        "party",
        "closeness_centrality",
        "Party reachability (Closeness)",
        "Higher values mean the party reaches the rest of the network through shorter paths.",
        "0-1 scale | higher = easier access",
        lambda label: display_party_label(label, short=True),
    )

    plot_layered_metric_dots(
        axes[0, 1],
        concept_metrics,
        "concept",
        "closeness_centrality",
        "Concept reachability (Closeness)",
        "Higher values mean the concept connects more easily to other concepts and parties.",
        "0-1 scale | higher = easier access",
        lambda label: display_concept_label(label),
    )

    plot_layered_metric_dots(
        axes[1, 0],
        party_metrics,
        "party",
        "betweenness_centrality",
        "Party bridge role (Betweenness)",
        "Higher values indicate the party sits on shortest paths between different clusters.",
        "0-1 scale | 0 = not a required bridge in that layer",
        lambda label: display_party_label(label, short=True),
    )

    summary_long = network_summary.melt(
        id_vars="layer",
        value_vars=["party_projection_density", "party_bloc_assortativity", "avg_party_closeness", "avg_concept_closeness"],
        var_name="metric",
        value_name="value",
    )
    summary_long["metric"] = summary_long["metric"].map(lambda value: NETWORK_SUMMARY_LABELS.get(value, value))
    sns.barplot(data=summary_long, x="value", y="metric", hue="layer", palette=layer_palette, ax=axes[1, 1])
    axes[1, 1].set_title("Layer summary", fontsize=14, fontweight="bold", loc="left", pad=14)
    axes[1, 1].text(
        0.01,
        0.98,
        "Density shows how tightly the network is connected, while assortativity summarizes bloc separation.",
        transform=axes[1, 1].transAxes,
        fontsize=8.8,
        color="#475569",
        va="top",
        bbox={"facecolor": "#f7f5ef", "edgecolor": "none", "pad": 0.2, "alpha": 0.9},
    )
    axes[1, 1].set_xlabel("Metric value")
    axes[1, 1].set_ylabel("")
    axes[1, 1].grid(axis="x", alpha=0.2)

    for ax in axes.flatten():
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("Layer")

    fig.suptitle(f"Term {term} | Closeness, Betweenness, and Network Summaries", fontsize=18, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.02,
        "Note: Closeness captures how quickly a node reaches the network; betweenness captures how much it bridges separate regions.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.95])
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_yearly_layer_networks(yearly_edges: pd.DataFrame, layer: str, output_path: Path, term: str) -> None:
    subset = yearly_edges[yearly_edges["layer"] == layer].copy()
    years = sorted(subset["year"].dropna().astype(int).unique().tolist())
    if not years:
        return

    cols = 1 if len(years) == 1 else 2
    rows = math.ceil(len(years) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8.8 * cols, 6.4 * rows))
    fig.patch.set_facecolor("#f7f5ef")
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, year in zip(axes_flat, years):
        year_df = subset[subset["year"] == year].copy()
        draw_bipartite_panel(ax, year_df, layer, str(year), node_scale=0.74, label_scale=0.94)

    for ax in axes_flat[len(years):]:
        ax.axis("off")

    fig.suptitle(f"Term {term} | {LAYER_STYLES[layer]['title']} Yearly Subnetworks", fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95], h_pad=2.0, w_pad=1.8)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_summary_note(
    metadata: dict[str, object],
    speech_df: pd.DataFrame,
    layer_summary: pd.DataFrame,
    notes_dir: Path,
) -> None:
    years = ", ".join(str(year) for year in metadata["available_years"]) if metadata["available_years"] else "yok"
    lines = [
        f"Term {metadata['term']} | Asama 1-2 ozet",
        "",
        f"Analiz penceresi: {metadata['window_start']} -> {metadata['window_end']}",
        f"Kavram iceren konusma sayisi: {metadata['speeches_with_concepts']}",
        f"Konusma-kavram kenar sayisi: {metadata['edge_rows']}",
        f"Kapsanan yillar: {years}",
        "",
        "BEME etiket dagilimi:",
    ]
    if not speech_df.empty:
        label_counts = speech_df["beme_label"].value_counts().reindex(["positive", "mixed", "negative"], fill_value=0)
        for label, count in label_counts.items():
            lines.append(f"- {label}: {int(count)}")
    lines.extend(
        [
            "",
            "Filtreler:",
            f"- non-mp / bilinmeyen satirlar: {metadata['filtered_non_mp']}",
            f"- donem penceresi disina dusen / parse edilemeyen tarih: {metadata['filtered_out_of_window']}",
            f"- cok kisa konusma: {metadata['filtered_short']}",
            "",
            "Katman ozetleri:",
        ]
    )
    if not layer_summary.empty:
        for row in layer_summary.itertuples(index=False):
            lines.append(
                f"- {row.layer}: party_edges={int(row.party_projection_edges)}, "
                f"concept_edges={int(row.concept_projection_edges)}, "
                f"party_density={row.party_projection_density:.3f}, "
                f"avg_party_closeness={row.avg_party_closeness:.3f}, "
                f"assortativity={row.party_bloc_assortativity:.3f}"
            )
    lines.extend(
        [
            "",
            "Uretilen ana dosyalar:",
            "- CSVs/term_22_beme_speech_labels.csv",
            "- CSVs/term_22_beme_concept_edges.csv",
            "- CSVs/term_22_layer_party_concept_edges.csv",
            "- CSVs/term_22_layer_party_metrics.csv",
            "- CSVs/term_22_layer_concept_metrics.csv",
            "- CSVs/term_22_layer_network_summary.csv",
            "- CSVs/term_22_yearly_layer_party_concept_edges.csv",
            "- CSVs/term_22_yearly_layer_party_metrics.csv",
            "- CSVs/term_22_yearly_layer_concept_metrics.csv",
            "- CSVs/term_22_yearly_layer_network_summary.csv",
            "- Figures/term_22_beme_label_distribution.png",
            "- Figures/term_22_layered_bipartite_networks.png",
            "- Figures/term_22_layered_metric_dashboard.png",
            "- Figures/term_22_total_yearly_networks.png",
            "- Figures/term_22_positive_yearly_networks.png",
            "- Figures/term_22_negative_yearly_networks.png",
        ]
    )
    (notes_dir / "term_22_stage1_stage2_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    (notes_dir / "term_22_output_index.txt").write_text("\n".join(lines[-16:]), encoding="utf-8")


def write_outputs(
    term: str,
    speech_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    layer_edges: pd.DataFrame,
    layer_party_metrics: pd.DataFrame,
    layer_concept_metrics: pd.DataFrame,
    layer_summary: pd.DataFrame,
    yearly_edges: pd.DataFrame,
    yearly_party_metrics: pd.DataFrame,
    yearly_concept_metrics: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    csvs_dir, figures_dir, notes_dir = reset_output_dirs(term)

    speech_df.to_csv(csvs_dir / f"term_{term}_beme_speech_labels.csv", index=False, encoding="utf-8-sig")
    edge_df.to_csv(csvs_dir / f"term_{term}_beme_concept_edges.csv", index=False, encoding="utf-8-sig")
    layer_edges.to_csv(csvs_dir / f"term_{term}_layer_party_concept_edges.csv", index=False, encoding="utf-8-sig")
    layer_party_metrics.to_csv(csvs_dir / f"term_{term}_layer_party_metrics.csv", index=False, encoding="utf-8-sig")
    layer_concept_metrics.to_csv(csvs_dir / f"term_{term}_layer_concept_metrics.csv", index=False, encoding="utf-8-sig")
    layer_summary.to_csv(csvs_dir / f"term_{term}_layer_network_summary.csv", index=False, encoding="utf-8-sig")
    yearly_edges.to_csv(csvs_dir / f"term_{term}_yearly_layer_party_concept_edges.csv", index=False, encoding="utf-8-sig")
    yearly_party_metrics.to_csv(csvs_dir / f"term_{term}_yearly_layer_party_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_concept_metrics.to_csv(csvs_dir / f"term_{term}_yearly_layer_concept_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_summary.to_csv(csvs_dir / f"term_{term}_yearly_layer_network_summary.csv", index=False, encoding="utf-8-sig")

    plot_beme_distribution(speech_df, figures_dir / f"term_{term}_beme_label_distribution.png", term)
    plot_layered_bipartite_networks(layer_edges, figures_dir / f"term_{term}_layered_bipartite_networks.png", term)
    plot_metric_dashboard(layer_party_metrics, layer_concept_metrics, layer_summary, figures_dir / f"term_{term}_layered_metric_dashboard.png", term)
    for layer in ["total", "positive", "negative"]:
        plot_yearly_layer_networks(yearly_edges, layer, figures_dir / f"term_{term}_{layer}_yearly_networks.png", term)

    write_summary_note(metadata, speech_df, layer_summary, notes_dir)


def build_all_outputs(term: str) -> None:
    speech_df, edge_df, metadata = load_term_beme_edges(term)

    layer_edge_frames: list[pd.DataFrame] = []
    layer_party_metric_frames: list[pd.DataFrame] = []
    layer_concept_metric_frames: list[pd.DataFrame] = []
    layer_summary_frames: list[pd.DataFrame] = []

    yearly_edge_frames: list[pd.DataFrame] = []
    yearly_party_metric_frames: list[pd.DataFrame] = []
    yearly_concept_metric_frames: list[pd.DataFrame] = []
    yearly_summary_frames: list[pd.DataFrame] = []

    for layer in ["total", "positive", "negative"]:
        term_layer_edges = aggregate_layer_edges(edge_df, layer)
        layer_edge_frames.append(term_layer_edges)

        party_metrics = build_party_metrics(term, term_layer_edges, layer)
        concept_metrics = build_concept_metrics(term_layer_edges, layer)
        layer_party_metric_frames.append(party_metrics)
        layer_concept_metric_frames.append(concept_metrics)
        layer_summary_frames.append(build_network_summary(term, term_layer_edges, layer, party_metrics, concept_metrics))

        year_layer_edges = aggregate_yearly_layer_edges(edge_df, layer)
        yearly_edge_frames.append(year_layer_edges)
        if not year_layer_edges.empty:
            for year in sorted(year_layer_edges["year"].dropna().astype(int).unique()):
                year_subset = year_layer_edges[year_layer_edges["year"] == year].copy()
                year_party_metrics = build_party_metrics(term, year_subset, layer)
                year_concept_metrics = build_concept_metrics(year_subset, layer)
                year_summary = build_network_summary(term, year_subset, layer, year_party_metrics, year_concept_metrics)
                year_party_metrics["year"] = year
                year_concept_metrics["year"] = year
                year_summary["year"] = year
                yearly_party_metric_frames.append(year_party_metrics)
                yearly_concept_metric_frames.append(year_concept_metrics)
                yearly_summary_frames.append(year_summary)

    write_outputs(
        term=term,
        speech_df=speech_df,
        edge_df=edge_df,
        layer_edges=pd.concat(layer_edge_frames, ignore_index=True) if layer_edge_frames else pd.DataFrame(),
        layer_party_metrics=pd.concat(layer_party_metric_frames, ignore_index=True) if layer_party_metric_frames else pd.DataFrame(),
        layer_concept_metrics=pd.concat(layer_concept_metric_frames, ignore_index=True) if layer_concept_metric_frames else pd.DataFrame(),
        layer_summary=pd.concat(layer_summary_frames, ignore_index=True) if layer_summary_frames else pd.DataFrame(),
        yearly_edges=pd.concat(yearly_edge_frames, ignore_index=True) if yearly_edge_frames else pd.DataFrame(),
        yearly_party_metrics=pd.concat(yearly_party_metric_frames, ignore_index=True) if yearly_party_metric_frames else pd.DataFrame(),
        yearly_concept_metrics=pd.concat(yearly_concept_metric_frames, ignore_index=True) if yearly_concept_metric_frames else pd.DataFrame(),
        yearly_summary=pd.concat(yearly_summary_frames, ignore_index=True) if yearly_summary_frames else pd.DataFrame(),
        metadata=metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BEME-based layered networks for a TBMM term.")
    parser.add_argument("--term", default="22", help="TBMM term number to process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_all_outputs(str(args.term))


if __name__ == "__main__":
    main()
