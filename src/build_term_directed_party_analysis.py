from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = PROJECT_ROOT / "Other" / ".mplconfig"
CACHE_DIR = PROJECT_ROOT / "Other" / ".cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

import build_beme_layered_networks as base

sys.path.insert(0, str(PROJECT_ROOT / "Data_Creation_Codes"))
from flex import VALID_PARTIES_BY_DONEM  # type: ignore[import-not-found]


DIRECTED_PREFIX = "directed_party"
TARGET_WINDOW_RADIUS = 120
MIN_MENTION_TEXT_LENGTH = 40
MIN_SPEECH_WORDS = 6
DEFAULT_CHUNK_SIZE = 100000
GRAPH_FACE = "#f7f5ef"
FOCUS_SIZE_BY_TERM = {
    "22": 4,
    "23": 4,
    "24": 4,
    "25": 4,
    "26": 5,
    "27": 6,
    "28": 6,
}
OUTPUT_STEM_TO_LABEL = {
    "mention_rows": "Mention rows",
    "period_edges": "Period edge table",
    "yearly_edges": "Yearly edge table",
    "monthly_panel": "Monthly directed panel",
    "party_metrics": "Party metric table",
    "network_summary": "Network summary",
    "hypothesis_coefficients": "Directed hypothesis coefficients",
    "hypothesis_summary": "Directed hypothesis summary",
    "granger_results": "Directed Granger tests",
    "main_networks": "Directed flow networks",
    "yearly_negative_networks": "Yearly negative flow networks",
    "yearly_positive_networks": "Yearly positive flow networks",
    "interaction_heatmaps": "Interaction heatmaps",
    "role_dashboard": "Role dashboard",
    "monthly_trends": "Monthly directed trends",
    "hypothesis_coefficients_figure": "Directed hypothesis figure",
}
LAYER_STYLES = {
    "total": {"title": "All Directed Attention", "edge_color": "#6b7280", "cmap": "Greys"},
    "positive": {"title": "Positive Support Flows", "edge_color": "#1f8f72", "cmap": "Greens"},
    "negative": {"title": "Negative Attack Flows", "edge_color": "#c4493d", "cmap": "Reds"},
}
GOVERNMENT_PARTIES_BY_TERM = {
    "22": {"Adalet ve Kalkınma Partisi"},
    "23": {"Adalet ve Kalkınma Partisi"},
    "24": {"Adalet ve Kalkınma Partisi"},
    "25": {"Adalet ve Kalkınma Partisi"},
    "26": {"Adalet ve Kalkınma Partisi"},
    "27": {"Adalet ve Kalkınma Partisi", "Milliyetçi Hareket Partisi", "Büyük Birlik Partisi"},
    "28": {"Adalet ve Kalkınma Partisi", "Milliyetçi Hareket Partisi", "Büyük Birlik Partisi", "Yeniden Refah Partisi"},
}
PARTY_METADATA = {
    "Adalet ve Kalkınma Partisi": {
        "short": "AK Party",
        "color": "#F4B400",
        "aliases": ("adalet ve kalkinma partisi", "ak parti", "ak party", "akp"),
    },
    "Cumhuriyet Halk Partisi": {
        "short": "CHP",
        "color": "#D7263D",
        "aliases": ("cumhuriyet halk partisi", "chp"),
    },
    "Milliyetçi Hareket Partisi": {
        "short": "MHP",
        "color": "#8F1D3F",
        "aliases": ("milliyetci hareket partisi", "mhp"),
    },
    "Anavatan Partisi": {
        "short": "ANAP",
        "color": "#FFD23F",
        "aliases": ("anavatan partisi", "anap"),
    },
    "Doğru Yol Partisi": {
        "short": "DYP",
        "color": "#0B4F6C",
        "aliases": ("dogru yol partisi", "dyp"),
    },
    "Bağımsız": {
        "short": "Independent",
        "color": "#6C757D",
        "aliases": ("bagimsiz", "bagimsizlar"),
    },
    "Halkların Demokratik Partisi": {
        "short": "HDP",
        "color": "#6A4C93",
        "aliases": ("halklarin demokratik partisi", "hdp"),
    },
    "Halkların Eşitlik ve Demokrasi Partisi": {
        "short": "HEDEP",
        "color": "#487A4C",
        "aliases": ("halklarin esitlik ve demokrasi partisi", "hedep"),
    },
    "DEM Parti": {
        "short": "DEM",
        "color": "#3A7D44",
        "aliases": ("dem parti",),
    },
    "İYİ Parti": {
        "short": "Good Party",
        "color": "#2E86AB",
        "aliases": ("iyi parti",),
    },
    "Saadet Partisi": {
        "short": "Felicity",
        "color": "#009966",
        "aliases": ("saadet partisi", "saadet"),
    },
    "Demokrat Parti": {
        "short": "DP",
        "color": "#4A6FA5",
        "aliases": ("demokrat parti",),
    },
    "Demokratik Sol Parti": {
        "short": "DSP",
        "color": "#3B82F6",
        "aliases": ("demokratik sol parti", "dsp"),
    },
    "Türkiye İşçi Partisi": {
        "short": "TIP",
        "color": "#C0392B",
        "aliases": ("turkiye isci partisi", "tip"),
    },
    "Büyük Birlik Partisi": {
        "short": "BBP",
        "color": "#9D2235",
        "aliases": ("buyuk birlik partisi", "bbp"),
    },
    "Yeniden Refah Partisi": {
        "short": "YRP",
        "color": "#3D9970",
        "aliases": ("yeniden refah partisi", "yrp"),
    },
    "YENİ YOL Partisi": {
        "short": "New Path",
        "color": "#A23B72",
        "aliases": ("yeni yol partisi", "yeni yol"),
    },
    "Genç Parti": {
        "short": "Young Party",
        "color": "#2C7A7B",
        "aliases": ("genc parti",),
    },
}


@dataclass
class DirectedArtifacts:
    term: str
    focus_parties: list[str]
    mentions: pd.DataFrame
    period_edges: pd.DataFrame
    yearly_edges: pd.DataFrame
    monthly_panel: pd.DataFrame
    party_metrics: pd.DataFrame
    network_summary: pd.DataFrame
    hypothesis_coefficients: pd.DataFrame
    hypothesis_summary: pd.DataFrame
    granger_results: pd.DataFrame
    metadata: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build readable directed inter-party network outputs.")
    parser.add_argument(
        "--terms",
        nargs="+",
        default=["22"],
        help="One or more parliamentary terms to process",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing directed CSV outputs and rebuild from the raw dataset",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size for raw dataset scanning",
    )
    return parser.parse_args()


def normalize_party_name(value: object) -> str:
    return base.normalize_text(str(value))


def slugify(value: str) -> str:
    normalized = base.normalize_text(value).replace(" ", "_").replace("-", "_").replace("/", "_")
    return "_".join(part for part in normalized.split("_") if part)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return float("nan")
    return float(numerator) / float(denominator)


def wrap_label(value: str, width: int = 18) -> str:
    return fill(value, width=width)


def canonicalize_dataset_party(value: object) -> str:
    raw = str(value).strip()
    normalized = normalize_party_name(raw)
    for canonical in PARTY_METADATA:
        if normalize_party_name(canonical) == normalized:
            return canonical
    return raw


def party_short_label(party: str) -> str:
    return PARTY_METADATA.get(party, {}).get("short", party)  # type: ignore[return-value]


def party_color(party: str) -> str:
    return PARTY_METADATA.get(party, {}).get("color", "#6C757D")  # type: ignore[return-value]


def party_bloc(term: str, party: str) -> str:
    return "government" if party in GOVERNMENT_PARTIES_BY_TERM.get(term, set()) else "opposition"


def ensure_output_dirs(term: str) -> tuple[Path, Path, Path]:
    csvs_dir, figures_dir, notes_dir = base.get_output_dirs(term)
    csvs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)
    return csvs_dir, figures_dir, notes_dir


def load_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def output_path(term: str, stem: str, suffix: str) -> Path:
    csvs_dir, figures_dir, notes_dir = ensure_output_dirs(term)
    if suffix == "csv":
        return csvs_dir / f"term_{term}_{DIRECTED_PREFIX}_{stem}.csv"
    if suffix == "png":
        return figures_dir / f"term_{term}_{DIRECTED_PREFIX}_{stem}.png"
    if suffix == "txt":
        return notes_dir / f"term_{term}_{DIRECTED_PREFIX}_{stem}.txt"
    raise ValueError(f"Unsupported suffix: {suffix}")


def build_alias_patterns(parties: list[str]) -> dict[str, tuple[re.Pattern[str], ...]]:
    compiled: dict[str, tuple[re.Pattern[str], ...]] = {}
    for party in parties:
        alias_specs = PARTY_METADATA.get(party, {}).get("aliases", (party,))
        patterns: list[re.Pattern[str]] = []
        for alias in alias_specs:
            normalized_alias = base.normalize_text(str(alias))
            escaped = re.escape(normalized_alias).replace(r"\ ", r"\s+")
            patterns.append(re.compile(rf"\b{escaped}\b"))
        compiled[party] = tuple(patterns)
    return compiled


def load_term_speeches(term: str, chunk_size: int) -> pd.DataFrame:
    start_date, end_date = base.TERM_WINDOWS[term]
    speech_rows: list[dict[str, object]] = []
    speech_counter = 0

    usecols = ["transcript_id", "donem", "date", "speaker", "party", "speech"]
    for chunk in pd.read_csv(base.DATA_PATH, usecols=usecols, chunksize=chunk_size, low_memory=False):
        chunk["donem"] = chunk["donem"].astype(str).str.replace(".0", "", regex=False)
        chunk = chunk[chunk["donem"] == term].copy()
        if chunk.empty:
            continue

        chunk["speaker"] = chunk["speaker"].fillna("").astype(str).str.strip()
        chunk["party"] = chunk["party"].map(canonicalize_dataset_party)
        chunk["speech"] = chunk["speech"].fillna("").astype(str)
        chunk["party_key"] = chunk["party"].map(normalize_party_name)
        chunk = chunk[~chunk["party_key"].isin(base.NON_MP_ROLES)].copy()
        if chunk.empty:
            continue

        chunk["parsed_date"] = chunk["date"].map(base.parse_tbmm_date)
        chunk = chunk[chunk["parsed_date"].notna()].copy()
        chunk = chunk[(chunk["parsed_date"] >= start_date) & (chunk["parsed_date"] <= end_date)].copy()
        if chunk.empty:
            continue

        chunk["normalized_speech"] = chunk["speech"].map(base.normalize_text)
        chunk["word_count"] = chunk["normalized_speech"].map(base.count_words)
        chunk = chunk[
            (chunk["word_count"] >= MIN_SPEECH_WORDS)
            & (chunk["normalized_speech"].str.len() >= MIN_MENTION_TEXT_LENGTH)
        ].copy()
        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):
            speech_rows.append(
                {
                    "speech_id": f"term_{term}_speech_{speech_counter:06d}",
                    "transcript_id": row.transcript_id,
                    "date": row.parsed_date.date().isoformat(),
                    "year": row.parsed_date.year,
                    "month": row.parsed_date.month,
                    "speaker": row.speaker,
                    "party": row.party,
                    "party_bloc": party_bloc(term, row.party),
                    "speech": row.speech,
                    "normalized_speech": row.normalized_speech,
                    "word_count": int(row.word_count),
                }
            )
            speech_counter += 1

    if not speech_rows:
        return pd.DataFrame(
            columns=[
                "speech_id",
                "transcript_id",
                "date",
                "year",
                "month",
                "speaker",
                "party",
                "party_bloc",
                "speech",
                "normalized_speech",
                "word_count",
            ]
        )
    return pd.DataFrame(speech_rows)


def select_focus_parties(term: str, speech: pd.DataFrame) -> list[str]:
    valid_parties = list(VALID_PARTIES_BY_DONEM.get(term, set()))
    counts = speech["party"].value_counts()
    ranked_named = [party for party in counts.index if party in valid_parties and party != "Bağımsız"]
    ranked_other = [party for party in counts.index if party in valid_parties and party == "Bağımsız"]
    focus_size = FOCUS_SIZE_BY_TERM.get(term, 5)
    selected = (ranked_named + ranked_other)[:focus_size]

    government_parties = [party for party in GOVERNMENT_PARTIES_BY_TERM.get(term, set()) if party in counts.index]
    for party in government_parties:
        if party not in selected:
            if len(selected) < focus_size:
                selected.append(party)
            elif selected:
                selected[-1] = party

    if "Cumhuriyet Halk Partisi" in counts.index and "Cumhuriyet Halk Partisi" not in selected:
        if len(selected) < focus_size:
            selected.append("Cumhuriyet Halk Partisi")
        elif selected:
            selected[-1] = "Cumhuriyet Halk Partisi"

    ordered = []
    for party in counts.index:
        if party in selected and party not in ordered:
            ordered.append(party)
    return ordered


def score_topic_hits(window_text: str) -> tuple[str | None, str | None, int]:
    topic_hits: Counter[str] = Counter()
    for concept_slug, payload in base.COMPILED_CONCEPTS.items():
        patterns: tuple[re.Pattern[str], ...] = payload["patterns"]  # type: ignore[assignment]
        hits = sum(len(pattern.findall(window_text)) for pattern in patterns)
        if hits > 0:
            topic_hits[concept_slug] += hits
    if not topic_hits:
        return None, None, 0
    topic_slug, topic_hit_count = topic_hits.most_common(1)[0]
    definition = base.COMPILED_CONCEPTS[topic_slug]["definition"]
    topic_label = definition.label
    return topic_slug, topic_label, int(topic_hit_count)


def build_directed_mentions(term: str, speech: pd.DataFrame, focus_parties: list[str]) -> pd.DataFrame:
    if speech.empty or not focus_parties:
        return pd.DataFrame()

    alias_patterns = build_alias_patterns(focus_parties)
    mention_rows: list[dict[str, object]] = []

    focus_set = set(focus_parties)
    for row in speech.itertuples(index=False):
        source_party = row.party
        if source_party not in focus_set:
            continue

        text = row.normalized_speech
        per_target: dict[str, dict[str, object]] = {}
        for target_party in focus_parties:
            if target_party == source_party:
                continue
            matches: list[re.Match[str]] = []
            for pattern in alias_patterns[target_party]:
                matches.extend(pattern.finditer(text))
            if not matches:
                continue

            matches = sorted(matches, key=lambda match: match.start())
            positive_hits = 0
            negative_hits = 0
            topic_counter: Counter[str] = Counter()
            topic_label_counter: Counter[str] = Counter()
            total_window_hits = 0

            for match in matches:
                start = max(0, match.start() - TARGET_WINDOW_RADIUS)
                end = min(len(text), match.end() + TARGET_WINDOW_RADIUS)
                window = text[start:end]
                positive_hits += base.count_pattern_hits(window, base.POSITIVE_PATTERNS)
                negative_hits += base.count_pattern_hits(window, base.NEGATIVE_PATTERNS)
                topic_slug, topic_label, topic_hit_count = score_topic_hits(window)
                if topic_slug is not None:
                    topic_counter[topic_slug] += topic_hit_count
                    topic_label_counter[topic_label or topic_slug] += topic_hit_count
                    total_window_hits += topic_hit_count

            mention_count = len(matches)
            tone_hits = positive_hits + negative_hits
            emphasis_bonus = min(total_window_hits, 5)
            total_weight = mention_count + (0.35 * tone_hits) + (0.20 * emphasis_bonus)
            frame_score = (positive_hits - negative_hits) / (tone_hits + 1)
            frame_label_local = base.beme_label(frame_score)
            # BINARY: mixed windows contribute zero signed weight — excluded from network edges
            if frame_label_local == "mixed":
                positive_weight = 0.0
                negative_weight = 0.0
            else:
                positive_weight = total_weight * max(frame_score, 0.0)
                negative_weight = total_weight * abs(min(frame_score, 0.0))
            dominant_topic_slug = topic_counter.most_common(1)[0][0] if topic_counter else None
            dominant_topic = topic_label_counter.most_common(1)[0][0] if topic_label_counter else None

            per_target[target_party] = {
                "speech_id": row.speech_id,
                "transcript_id": row.transcript_id,
                "date": row.date,
                "year": row.year,
                "month": row.month,
                "speaker": row.speaker,
                "source_party": source_party,
                "target_party": target_party,
                "source_bloc": party_bloc(term, source_party),
                "target_bloc": party_bloc(term, target_party),
                "mention_count": mention_count,
                "positive_hits": positive_hits,
                "negative_hits": negative_hits,
                "tone_hits": tone_hits,
                "topic_slug": dominant_topic_slug,
                "topic": dominant_topic,
                "topic_hit_count": total_window_hits,
                "total_weight": round(total_weight, 6),
                "positive_weight": round(positive_weight, 6),
                "negative_weight": round(negative_weight, 6),
                "net_weight": round(positive_weight - negative_weight, 6),
                "frame_score": round(frame_score, 6),
                "frame_label": base.beme_label(frame_score),
            }

        mention_rows.extend(per_target.values())

    if not mention_rows:
        return pd.DataFrame()

    mentions = pd.DataFrame(mention_rows)
    mentions["date"] = pd.to_datetime(mentions["date"])
    mentions["month_date"] = mentions["date"].dt.to_period("M").dt.to_timestamp()
    return mentions.sort_values(["date", "source_party", "target_party"]).reset_index(drop=True)


def aggregate_edges(mentions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if mentions.empty:
        return pd.DataFrame()

    aggregated = (
        mentions.groupby(group_cols, dropna=False)
        .agg(
            speech_count=("speech_id", "nunique"),
            mention_count=("mention_count", "sum"),
            positive_hits=("positive_hits", "sum"),
            negative_hits=("negative_hits", "sum"),
            total_weight=("total_weight", "sum"),
            positive_weight=("positive_weight", "sum"),
            negative_weight=("negative_weight", "sum"),
            net_weight=("net_weight", "sum"),
            avg_score=("frame_score", "mean"),
        )
        .reset_index()
    )
    return aggregated.sort_values(["source_party", "target_party"]).reset_index(drop=True)


def weighted_reciprocity(edge_df: pd.DataFrame, weight_col: str) -> float:
    positive_edges = edge_df[edge_df[weight_col] > 0].copy()
    if positive_edges.empty:
        return float("nan")

    lookup: dict[tuple[str, str], float] = {
        (row.source_party, row.target_party): float(getattr(row, weight_col))
        for row in positive_edges.itertuples(index=False)
    }
    numerator = 0.0
    denominator = float(positive_edges[weight_col].sum())
    seen_pairs: set[tuple[str, str]] = set()
    for source_party, target_party in lookup:
        ordered = tuple(sorted((source_party, target_party)))
        if ordered in seen_pairs or source_party == target_party:
            continue
        seen_pairs.add(ordered)
        numerator += min(lookup.get((source_party, target_party), 0.0), lookup.get((target_party, source_party), 0.0))
    return safe_divide(numerator, denominator)


def build_graph_from_edges(edge_df: pd.DataFrame, parties: list[str], weight_col: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(parties)
    for row in edge_df.itertuples(index=False):
        weight = float(getattr(row, weight_col, 0.0))
        if row.source_party == row.target_party or weight <= 0:
            continue
        graph.add_edge(row.source_party, row.target_party, weight=weight, distance=1.0 / weight)
    return graph


def compute_party_metrics(period_edges: pd.DataFrame, parties: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for layer, weight_col in (("total", "total_weight"), ("positive", "positive_weight"), ("negative", "negative_weight")):
        layer_graph = build_graph_from_edges(period_edges, parties, weight_col)
        if layer_graph.number_of_edges() > 0:
            betweenness = nx.betweenness_centrality(layer_graph, weight="distance", normalized=True)
            closeness = nx.closeness_centrality(layer_graph.reverse(copy=True), distance="distance")
            pagerank = nx.pagerank(layer_graph, weight="weight")
        else:
            betweenness = {party: 0.0 for party in parties}
            closeness = {party: 0.0 for party in parties}
            pagerank = {party: 0.0 for party in parties}

        for party in parties:
            rows.append(
                {
                    "party": party,
                    "party_short": party_short_label(party),
                    "layer": layer,
                    "weighted_out_degree": float(layer_graph.out_degree(party, weight="weight")),
                    "weighted_in_degree": float(layer_graph.in_degree(party, weight="weight")),
                    "betweenness_centrality": float(betweenness.get(party, 0.0)),
                    "closeness_centrality": float(closeness.get(party, 0.0)),
                    "pagerank": float(pagerank.get(party, 0.0)),
                }
            )
    return pd.DataFrame(rows)


def compute_network_summary(period_edges: pd.DataFrame, parties: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for layer, weight_col in (("total", "total_weight"), ("positive", "positive_weight"), ("negative", "negative_weight")):
        graph = build_graph_from_edges(period_edges, parties, weight_col)
        density = nx.density(graph) if graph.number_of_nodes() > 1 else float("nan")
        binary_reciprocity = nx.reciprocity(graph) if graph.number_of_edges() > 0 else float("nan")
        rows.append(
            {
                "layer": layer,
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "density": density,
                "binary_reciprocity": binary_reciprocity,
                "weighted_reciprocity": weighted_reciprocity(period_edges, weight_col),
            }
        )
    return pd.DataFrame(rows)


def build_monthly_panel(term: str, mentions: pd.DataFrame, focus_parties: list[str]) -> pd.DataFrame:
    start_date, end_date = base.TERM_WINDOWS[term]
    month_index = pd.date_range(start_date.to_period("M").to_timestamp(), end_date.to_period("M").to_timestamp(), freq="MS")
    panel = pd.DataFrame({"date": month_index})
    panel["term"] = term
    panel["term_year"] = panel["date"].dt.year

    if mentions.empty:
        return panel

    monthly = (
        mentions.groupby("month_date", dropna=False)
        .agg(
            directed_total_weight=("total_weight", "sum"),
            directed_positive_weight=("positive_weight", "sum"),
            directed_negative_weight=("negative_weight", "sum"),
            targeted_speeches=("speech_id", "nunique"),
            mention_rows=("speech_id", "size"),
        )
        .reset_index()
        .rename(columns={"month_date": "date"})
    )
    panel = panel.merge(monthly, on="date", how="left")
    panel["directed_negative_share"] = panel["directed_negative_weight"] / panel["directed_total_weight"].replace({0: np.nan})

    monthly_cross = (
        mentions.assign(
            negative_cross_weight=np.where(
                (mentions["source_bloc"] != mentions["target_bloc"]),
                mentions["negative_weight"],
                0.0,
            ),
            positive_cross_weight=np.where(
                (mentions["source_bloc"] != mentions["target_bloc"]),
                mentions["positive_weight"],
                0.0,
            ),
            economy_targeted_negative=np.where(mentions["topic_slug"] == "economy", mentions["negative_weight"], 0.0),
            constitution_targeted_negative=np.where(mentions["topic_slug"] == "constitution", mentions["negative_weight"], 0.0),
        )
        .groupby("month_date", dropna=False)
        .agg(
            negative_cross_weight=("negative_cross_weight", "sum"),
            positive_cross_weight=("positive_cross_weight", "sum"),
            economy_targeted_negative_weight=("economy_targeted_negative", "sum"),
            constitution_targeted_negative_weight=("constitution_targeted_negative", "sum"),
        )
        .reset_index()
        .rename(columns={"month_date": "date"})
    )
    panel = panel.merge(monthly_cross, on="date", how="left")
    panel["negative_cross_bloc_share"] = panel["negative_cross_weight"] / panel["directed_negative_weight"].replace({0: np.nan})
    panel["positive_cross_bloc_share"] = panel["positive_cross_weight"] / panel["directed_positive_weight"].replace({0: np.nan})
    panel["economy_targeted_negative_share"] = panel["economy_targeted_negative_weight"] / panel["directed_negative_weight"].replace({0: np.nan})
    panel["constitution_targeted_negative_share"] = panel["constitution_targeted_negative_weight"] / panel["directed_negative_weight"].replace({0: np.nan})

    negative_monthly_edges = aggregate_edges(
        mentions[mentions["negative_weight"] > 0].assign(month_key=mentions["month_date"]),
        ["month_key", "source_party", "target_party", "source_bloc", "target_bloc"],
    ).rename(columns={"month_key": "date"})
    positive_monthly_edges = aggregate_edges(
        mentions[mentions["positive_weight"] > 0].assign(month_key=mentions["month_date"]),
        ["month_key", "source_party", "target_party", "source_bloc", "target_bloc"],
    ).rename(columns={"month_key": "date"})

    negative_reciprocity_rows: list[dict[str, object]] = []
    positive_reciprocity_rows: list[dict[str, object]] = []
    for month_date in month_index:
        neg_edges = negative_monthly_edges[negative_monthly_edges["date"] == month_date].copy()
        pos_edges = positive_monthly_edges[positive_monthly_edges["date"] == month_date].copy()
        neg_graph = build_graph_from_edges(neg_edges, focus_parties, "negative_weight")
        pos_graph = build_graph_from_edges(pos_edges, focus_parties, "positive_weight")
        negative_reciprocity_rows.append(
            {
                "date": month_date,
                "negative_reciprocity": weighted_reciprocity(neg_edges, "negative_weight"),
                "negative_density": nx.density(neg_graph) if neg_graph.number_of_nodes() > 1 else float("nan"),
            }
        )
        positive_reciprocity_rows.append(
            {
                "date": month_date,
                "positive_reciprocity": weighted_reciprocity(pos_edges, "positive_weight"),
                "positive_density": nx.density(pos_graph) if pos_graph.number_of_nodes() > 1 else float("nan"),
            }
        )

    panel = panel.merge(pd.DataFrame(negative_reciprocity_rows), on="date", how="left")
    panel = panel.merge(pd.DataFrame(positive_reciprocity_rows), on="date", how="left")

    if {"Adalet ve Kalkınma Partisi", "Cumhuriyet Halk Partisi"}.issubset(set(focus_parties)):
        akp_chp = (
            mentions[
                (
                    ((mentions["source_party"] == "Adalet ve Kalkınma Partisi") & (mentions["target_party"] == "Cumhuriyet Halk Partisi"))
                    | ((mentions["source_party"] == "Cumhuriyet Halk Partisi") & (mentions["target_party"] == "Adalet ve Kalkınma Partisi"))
                )
            ]
            .groupby("month_date", dropna=False)
            .agg(akp_chp_negative_weight=("negative_weight", "sum"))
            .reset_index()
            .rename(columns={"month_date": "date"})
        )
        panel = panel.merge(akp_chp, on="date", how="left")
        panel["akp_chp_negative_share"] = panel["akp_chp_negative_weight"] / panel["directed_negative_weight"].replace({0: np.nan})

        akp_in = (
            mentions[mentions["target_party"] == "Adalet ve Kalkınma Partisi"]
            .groupby("month_date", dropna=False)
            .agg(akp_negative_in_weight=("negative_weight", "sum"))
            .reset_index()
            .rename(columns={"month_date": "date"})
        )
        chp_in = (
            mentions[mentions["target_party"] == "Cumhuriyet Halk Partisi"]
            .groupby("month_date", dropna=False)
            .agg(chp_negative_in_weight=("negative_weight", "sum"))
            .reset_index()
            .rename(columns={"month_date": "date"})
        )
        panel = panel.merge(akp_in, on="date", how="left")
        panel = panel.merge(chp_in, on="date", how="left")
        panel["akp_negative_in_share"] = panel["akp_negative_in_weight"] / panel["directed_negative_weight"].replace({0: np.nan})
        panel["chp_negative_in_share"] = panel["chp_negative_in_weight"] / panel["directed_negative_weight"].replace({0: np.nan})

    macro_panel_path = PROJECT_ROOT / f"Term_{term}" / "CSVs" / f"term_{term}_stage3_stage7_monthly_network_macro_panel.csv"
    if macro_panel_path.exists():
        macro_panel = pd.read_csv(macro_panel_path)
        macro_panel["date"] = pd.to_datetime(macro_panel["date"])
        panel = macro_panel.merge(panel, on="date", how="left")

    for column in ["negative_cross_bloc_share", "positive_cross_bloc_share", "akp_chp_negative_share"]:
        if column in panel.columns:
            panel[f"{column}_lag1"] = panel[column].shift(1)
    return panel.sort_values("date").reset_index(drop=True)


def build_party_order(period_edges: pd.DataFrame, parties: list[str]) -> tuple[list[str], list[str], list[str]]:
    if period_edges.empty:
        return parties, parties, parties

    total_out = period_edges.groupby("source_party")["total_weight"].sum().sort_values(ascending=False)
    total_in = period_edges.groupby("target_party")["total_weight"].sum().sort_values(ascending=False)
    total_strength = (
        period_edges.groupby("source_party")["total_weight"].sum().add(
            period_edges.groupby("target_party")["total_weight"].sum(),
            fill_value=0.0,
        )
        .sort_values(ascending=False)
    )

    left_order = [party for party in total_out.index if party in parties]
    right_order = [party for party in total_in.index if party in parties]
    heatmap_order = [party for party in total_strength.index if party in parties]
    for party in parties:
        if party not in left_order:
            left_order.append(party)
        if party not in right_order:
            right_order.append(party)
        if party not in heatmap_order:
            heatmap_order.append(party)
    return left_order, right_order, heatmap_order


def select_plot_edges(edge_df: pd.DataFrame, weight_col: str, max_edges: int) -> pd.DataFrame:
    edge_df = edge_df[edge_df[weight_col] > 0].copy()
    if edge_df.empty:
        return edge_df
    edge_df = edge_df.sort_values(weight_col, ascending=False).reset_index(drop=True)
    threshold = max(float(edge_df[weight_col].max()) * 0.18, float(edge_df[weight_col].quantile(0.65)))
    filtered = edge_df[edge_df[weight_col] >= threshold].copy()
    if filtered.shape[0] < min(max_edges, edge_df.shape[0]):
        filtered = edge_df.head(min(max_edges, edge_df.shape[0])).copy()
    return filtered


def draw_flow_panel(
    ax: plt.Axes,
    edge_df: pd.DataFrame,
    parties: list[str],
    weight_col: str,
    title: str,
    edge_color: str,
    left_order: list[str],
    right_order: list[str],
    compact: bool = False,
) -> None:
    ax.set_facecolor(GRAPH_FACE)
    ax.axis("off")

    font_size = 11 if compact else 13
    note_size = 8 if compact else 10
    node_edge = 1.0 if compact else 1.5
    plot_edges = select_plot_edges(edge_df, weight_col, max_edges=10 if compact else 14)

    if plot_edges.empty:
        ax.text(0.5, 0.52, "No readable flow above threshold", ha="center", va="center", fontsize=font_size, color="#475569")
        ax.text(0.5, 0.43, "The directed layer exists, but weights are too small for display.", ha="center", va="center", fontsize=note_size, color="#64748b")
        ax.set_title(title, fontsize=font_size + 2, fontweight="bold", color="#1f2937", pad=10)
        return

    out_weight = edge_df.groupby("source_party")[weight_col].sum()
    in_weight = edge_df.groupby("target_party")[weight_col].sum()
    max_out = max(float(out_weight.max()), 1.0) if not out_weight.empty else 1.0
    max_in = max(float(in_weight.max()), 1.0) if not in_weight.empty else 1.0
    max_edge = max(float(plot_edges[weight_col].max()), 1.0)

    y_left = np.linspace(0.90, 0.10, len(left_order)) if left_order else np.array([])
    y_right = np.linspace(0.90, 0.10, len(right_order)) if right_order else np.array([])
    left_positions = {party: (0.20, y_left[idx]) for idx, party in enumerate(left_order)}
    right_positions = {party: (0.80, y_right[idx]) for idx, party in enumerate(right_order)}

    for row in plot_edges.itertuples(index=False):
        source_x, source_y = left_positions[row.source_party]
        target_x, target_y = right_positions[row.target_party]
        weight = float(getattr(row, weight_col))
        width = 1.2 + (5.5 * math.sqrt(weight / max_edge))
        alpha = 0.22 + (0.55 * math.sqrt(weight / max_edge))
        arrow = FancyArrowPatch(
            (source_x + 0.04, source_y),
            (target_x - 0.04, target_y),
            arrowstyle="-|>",
            mutation_scale=10 if compact else 12,
            linewidth=width,
            color=edge_color,
            alpha=min(alpha, 0.90),
            connectionstyle="arc3,rad=0.0",
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=1,
        )
        ax.add_patch(arrow)

    for party in left_order:
        x, y = left_positions[party]
        node_size = 900 + (2400 * math.sqrt(float(out_weight.get(party, 0.0)) / max_out))
        ax.scatter(x, y, s=node_size, color=party_color(party), edgecolor="white", linewidth=node_edge, zorder=3)
        ax.text(x - 0.07, y, party_short_label(party), ha="right", va="center", fontsize=font_size, fontweight="bold", color="#111827")

    for party in right_order:
        x, y = right_positions[party]
        node_size = 900 + (2400 * math.sqrt(float(in_weight.get(party, 0.0)) / max_in))
        ax.scatter(x, y, s=node_size, color=party_color(party), edgecolor="white", linewidth=node_edge, zorder=3)
        ax.text(x + 0.07, y, party_short_label(party), ha="left", va="center", fontsize=font_size, fontweight="bold", color="#111827")

    ax.text(0.20, 0.98, "Sources", ha="center", va="bottom", fontsize=note_size, color="#475569", transform=ax.transAxes)
    ax.text(0.80, 0.98, "Targets", ha="center", va="bottom", fontsize=note_size, color="#475569", transform=ax.transAxes)
    ax.text(0.5, 0.02, "Line width = targeting weight", ha="center", va="bottom", fontsize=note_size, color="#64748b", transform=ax.transAxes)
    ax.set_title(title, fontsize=font_size + 3, fontweight="bold", color="#1f2937", pad=12)


def plot_main_networks(term: str, period_edges: pd.DataFrame, parties: list[str], figures_dir: Path) -> Path:
    left_order, right_order, _ = build_party_order(period_edges, parties)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=GRAPH_FACE)
    positive_edges = period_edges[period_edges["positive_weight"] > 0].copy()
    negative_edges = period_edges[period_edges["negative_weight"] > 0].copy()

    draw_flow_panel(
        axes[0],
        positive_edges,
        parties,
        "positive_weight",
        "Positive Support Flows",
        LAYER_STYLES["positive"]["edge_color"],
        left_order,
        right_order,
    )
    draw_flow_panel(
        axes[1],
        negative_edges,
        parties,
        "negative_weight",
        "Negative Attack Flows",
        LAYER_STYLES["negative"]["edge_color"],
        left_order,
        right_order,
    )
    fig.suptitle(f"Term {term} Directed Inter-Party Flows", fontsize=18, fontweight="bold", color="#0f172a", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output = figures_dir / f"term_{term}_{DIRECTED_PREFIX}_main_networks.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=GRAPH_FACE)
    plt.close(fig)
    return output


def plot_yearly_flow_networks(
    term: str,
    yearly_edges: pd.DataFrame,
    period_edges: pd.DataFrame,
    parties: list[str],
    figures_dir: Path,
    layer: str,
    weight_col: str,
    output_stem: str,
    title: str,
) -> Path:
    yearly_subset = yearly_edges[yearly_edges[weight_col] > 0].copy()
    years = sorted(int(year) for year in yearly_subset["year"].dropna().unique()) if not yearly_subset.empty else []
    if not years:
        years = list(range(base.TERM_WINDOWS[term][0].year, base.TERM_WINDOWS[term][1].year + 1))

    overall_layer = period_edges[period_edges[weight_col] > 0].copy()
    left_order = (
        overall_layer.groupby("source_party")[weight_col].sum().sort_values(ascending=False).index.tolist()
        if not overall_layer.empty
        else parties[:]
    )
    right_order = (
        overall_layer.groupby("target_party")[weight_col].sum().sort_values(ascending=False).index.tolist()
        if not overall_layer.empty
        else parties[:]
    )
    for party in parties:
        if party not in left_order:
            left_order.append(party)
        if party not in right_order:
            right_order.append(party)

    ncols = 2
    nrows = math.ceil(len(years) / ncols) if years else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5.2 * nrows), facecolor=GRAPH_FACE)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)

    for axis in axes_array.ravel():
        axis.set_facecolor(GRAPH_FACE)

    for axis, year in zip(axes_array.ravel(), years):
        year_slice = yearly_subset[yearly_subset["year"] == year].copy()
        draw_flow_panel(
            axis,
            year_slice,
            parties,
            weight_col,
            str(year),
            LAYER_STYLES[layer]["edge_color"],
            left_order,
            right_order,
            compact=True,
        )

    for axis in axes_array.ravel()[len(years):]:
        axis.axis("off")

    fig.suptitle(title, fontsize=18, fontweight="bold", color="#0f172a", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output = figures_dir / f"term_{term}_{DIRECTED_PREFIX}_{output_stem}.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=GRAPH_FACE)
    plt.close(fig)
    return output


def interaction_matrix(period_edges: pd.DataFrame, parties: list[str], weight_col: str) -> pd.DataFrame:
    matrix = (
        period_edges.pivot_table(
            index="source_party",
            columns="target_party",
            values=weight_col,
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=parties, columns=parties, fill_value=0.0)
    )
    return matrix


def plot_interaction_heatmaps(term: str, period_edges: pd.DataFrame, parties: list[str], figures_dir: Path) -> Path:
    _, _, heatmap_order = build_party_order(period_edges, parties)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), facecolor=GRAPH_FACE)
    layers = [("total", "total_weight"), ("positive", "positive_weight"), ("negative", "negative_weight")]

    for axis, (layer, weight_col) in zip(axes, layers):
        matrix = interaction_matrix(period_edges, heatmap_order, weight_col)
        sns.heatmap(
            matrix.rename(index=party_short_label, columns=party_short_label),
            ax=axis,
            cmap=LAYER_STYLES[layer]["cmap"],
            annot=True,
            fmt=".0f",
            linewidths=0.6,
            linecolor="#ffffff",
            cbar_kws={"shrink": 0.8},
        )
        axis.set_title(LAYER_STYLES[layer]["title"], fontsize=13, fontweight="bold")
        axis.set_xlabel("Target party")
        axis.set_ylabel("Source party")
        axis.tick_params(axis="x", rotation=35)
        axis.tick_params(axis="y", rotation=0)

    fig.suptitle(f"Term {term} Directed Interaction Matrices", fontsize=18, fontweight="bold", color="#0f172a", y=1.01)
    fig.tight_layout()
    output = figures_dir / f"term_{term}_{DIRECTED_PREFIX}_interaction_heatmaps.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=GRAPH_FACE)
    plt.close(fig)
    return output


def plot_role_dashboard(term: str, party_metrics: pd.DataFrame, parties: list[str], figures_dir: Path) -> Path:
    total = party_metrics[party_metrics["layer"] == "total"].copy()
    negative = party_metrics[party_metrics["layer"] == "negative"].copy()
    positive = party_metrics[party_metrics["layer"] == "positive"].copy()

    total = total.set_index("party").reindex(parties).reset_index()
    negative = negative.set_index("party").reindex(parties).reset_index()
    positive = positive.set_index("party").reindex(parties).reset_index()

    labels = [party_short_label(party) for party in parties]
    colors = [party_color(party) for party in parties]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor=GRAPH_FACE)
    for axis in axes.ravel():
        axis.set_facecolor(GRAPH_FACE)

    axes[0, 0].barh(labels, total["weighted_out_degree"].fillna(0.0), color=colors)
    axes[0, 0].set_title("Weighted Outgoing Targeting", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Outgoing weight")

    axes[0, 1].barh(labels, total["weighted_in_degree"].fillna(0.0), color=colors)
    axes[0, 1].set_title("Weighted Incoming Targeting", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Incoming weight")

    axes[1, 0].barh(labels, negative["weighted_in_degree"].fillna(0.0), color="#c4493d")
    axes[1, 0].set_title("Negative Incoming Pressure", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Negative incoming weight")

    x = np.arange(len(labels))
    axes[1, 1].bar(x - 0.18, total["betweenness_centrality"].fillna(0.0), width=0.36, color="#64748b", label="Betweenness")
    axes[1, 1].bar(x + 0.18, total["pagerank"].fillna(0.0), width=0.36, color="#1f8f72", label="PageRank")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1, 1].set_title("Brokerage and Centrality", fontsize=13, fontweight="bold")
    axes[1, 1].legend(frameon=False, loc="upper right")

    for axis in axes.ravel():
        axis.grid(axis="x", alpha=0.15, linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    support_order = positive["weighted_out_degree"].fillna(0.0).sort_values(ascending=False).index
    del support_order  # keep the dashboard focused and silence lints

    fig.suptitle(f"Term {term} Directed Party Roles", fontsize=18, fontweight="bold", color="#0f172a", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output = figures_dir / f"term_{term}_{DIRECTED_PREFIX}_role_dashboard.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=GRAPH_FACE)
    plt.close(fig)
    return output


def plot_hypothesis_placeholder(term: str, figures_dir: Path, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=GRAPH_FACE)
    ax.set_facecolor(GRAPH_FACE)
    ax.axis("off")
    ax.text(0.5, 0.60, "Directed hypothesis outputs are unavailable", ha="center", va="center", fontsize=16, fontweight="bold", color="#0f172a")
    ax.text(0.5, 0.42, wrap_label(message, width=48), ha="center", va="center", fontsize=11, color="#475569")
    output = figures_dir / f"term_{term}_{DIRECTED_PREFIX}_hypothesis_coefficients.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=GRAPH_FACE)
    plt.close(fig)
    return output


def strongest_negative_statement(period_edges: pd.DataFrame) -> tuple[str, float]:
    negative = period_edges[period_edges["negative_weight"] > 0].copy()
    if negative.empty:
        return "No negative edge exceeds the readability threshold.", float("nan")
    top_row = negative.sort_values("negative_weight", ascending=False).iloc[0]
    total_negative = float(negative["negative_weight"].sum())
    share = safe_divide(float(top_row["negative_weight"]), total_negative)
    sentence = (
        f"Strongest negative edge: {party_short_label(str(top_row['source_party']))} -> "
        f"{party_short_label(str(top_row['target_party']))} "
        f"({float(top_row['negative_weight']):.1f} weight, {share * 100.0:.1f}% of all negative weight)."
    )
    return sentence, share


def write_summary_note(
    term: str,
    artifacts: DirectedArtifacts,
    notes_dir: Path,
    created_files: list[Path],
) -> Path:
    targeted_speeches = int(artifacts.mentions["speech_id"].nunique()) if not artifacts.mentions.empty else 0
    mention_rows = int(len(artifacts.mentions))
    strongest_line, _ = strongest_negative_statement(artifacts.period_edges)

    lines = [
        f"Term {term} | Directed inter-party network summary",
        "",
        f"Focus parties: {', '.join(party_short_label(party) for party in artifacts.focus_parties)}",
        f"Directed mention rows: {mention_rows}",
        f"Speeches with explicit party targeting: {targeted_speeches}",
        strongest_line,
    ]

    if not artifacts.network_summary.empty:
        negative_summary = artifacts.network_summary[artifacts.network_summary["layer"] == "negative"]
        if not negative_summary.empty:
            row = negative_summary.iloc[0]
            lines.append(
                "Negative network summary: "
                f"density={float(row['density']):.3f}, "
                f"binary reciprocity={float(row['binary_reciprocity']):.3f}, "
                f"weighted reciprocity={float(row['weighted_reciprocity']):.3f}"
            )

    if not artifacts.party_metrics.empty:
        total_metrics = artifacts.party_metrics[artifacts.party_metrics["layer"] == "total"].copy()
        if not total_metrics.empty:
            top_target = total_metrics.sort_values("weighted_in_degree", ascending=False).iloc[0]
            top_source = total_metrics.sort_values("weighted_out_degree", ascending=False).iloc[0]
            lines.append(f"Main target node: {top_target['party_short']}")
            lines.append(f"Main outgoing node: {top_source['party_short']}")

    if not artifacts.hypothesis_summary.empty:
        lines.append("")
        lines.append("Hypothesis results:")
        for row in artifacts.hypothesis_summary.itertuples(index=False):
            lines.append(
                f"- {row.hypothesis_id}: beta={float(row.primary_beta):.4f} | p={float(row.primary_p_value):.4f} | "
                f"supported_10pct={bool(row.supported_at_10pct)} | {row.label}"
            )
    else:
        lines.append("")
        lines.append("Hypothesis results:")
        lines.append("- No directed hypothesis outputs were generated for this term.")

    if not artifacts.granger_results.empty:
        lines.append("")
        lines.append("Directed Granger signals:")
        for row in artifacts.granger_results.sort_values(["cause", "effect", "lag"]).itertuples(index=False):
            lines.append(
                f"- {row.cause} -> {row.effect} | lag={int(row.lag)} | p={float(row.p_value):.4f}"
            )

    lines.append("")
    for file_path in created_files:
        relative = file_path.relative_to(PROJECT_ROOT / f"Term_{term}")
        lines.append(f"- {relative}")

    summary_path = notes_dir / f"term_{term}_{DIRECTED_PREFIX}_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def write_output_index(term: str, notes_dir: Path, created_files: list[Path]) -> Path:
    lines = [f"Term {term} | Directed output index", ""]
    for file_path in created_files:
        relative = file_path.relative_to(PROJECT_ROOT / f"Term_{term}")
        stem = file_path.stem.replace(f"term_{term}_{DIRECTED_PREFIX}_", "")
        label = OUTPUT_STEM_TO_LABEL.get(stem, stem.replace("_", " ").title())
        lines.append(f"{label}: {relative}")
    index_path = notes_dir / f"term_{term}_{DIRECTED_PREFIX}_output_index.txt"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def save_table_if_nonempty(frame: pd.DataFrame, path: Path) -> Path | None:
    if frame.empty:
        return None
    frame.to_csv(path, index=False)
    return path


def load_existing_artifacts(term: str) -> DirectedArtifacts | None:
    period_edges_path = output_path(term, "period_edges", "csv")
    yearly_edges_path = output_path(term, "yearly_edges", "csv")
    party_metrics_path = output_path(term, "party_metrics", "csv")
    network_summary_path = output_path(term, "network_summary", "csv")
    if not all(path.exists() for path in (period_edges_path, yearly_edges_path, party_metrics_path, network_summary_path)):
        return None

    mentions = load_optional_csv(output_path(term, "mention_rows", "csv"))
    if not mentions.empty and "date" in mentions.columns:
        mentions["date"] = pd.to_datetime(mentions["date"])
        if "month_date" in mentions.columns:
            mentions["month_date"] = pd.to_datetime(mentions["month_date"])

    period_edges = pd.read_csv(period_edges_path)
    yearly_edges = pd.read_csv(yearly_edges_path)
    monthly_panel = load_optional_csv(output_path(term, "monthly_panel", "csv"))
    if not monthly_panel.empty and "date" in monthly_panel.columns:
        monthly_panel["date"] = pd.to_datetime(monthly_panel["date"])
    party_metrics = pd.read_csv(party_metrics_path)
    network_summary = pd.read_csv(network_summary_path)
    hypothesis_coefficients = load_optional_csv(output_path(term, "hypothesis_coefficients", "csv"))
    hypothesis_summary = load_optional_csv(output_path(term, "hypothesis_summary", "csv"))
    granger_results = load_optional_csv(output_path(term, "granger_results", "csv"))

    left_order = (
        period_edges.groupby("source_party")["total_weight"].sum().sort_values(ascending=False).index.tolist()
        if not period_edges.empty
        else []
    )
    parties = []
    for party in left_order + period_edges.get("target_party", pd.Series(dtype=str)).tolist():
        if party not in parties:
            parties.append(party)

    metadata = {
        "used_existing_outputs": True,
        "macro_panel_available": bool((PROJECT_ROOT / f"Term_{term}" / "CSVs" / f"term_{term}_stage3_stage7_monthly_network_macro_panel.csv").exists()),
    }
    return DirectedArtifacts(
        term=term,
        focus_parties=parties,
        mentions=mentions,
        period_edges=period_edges,
        yearly_edges=yearly_edges,
        monthly_panel=monthly_panel,
        party_metrics=party_metrics,
        network_summary=network_summary,
        hypothesis_coefficients=hypothesis_coefficients,
        hypothesis_summary=hypothesis_summary,
        granger_results=granger_results,
        metadata=metadata,
    )


def compute_artifacts(term: str, chunk_size: int) -> DirectedArtifacts:
    speech = load_term_speeches(term, chunk_size)
    focus_parties = select_focus_parties(term, speech)
    mentions = build_directed_mentions(term, speech, focus_parties)

    period_edges = aggregate_edges(
        mentions,
        ["source_party", "target_party", "source_bloc", "target_bloc"],
    )
    yearly_edges = aggregate_edges(
        mentions,
        ["year", "source_party", "target_party", "source_bloc", "target_bloc"],
    )
    party_metrics = compute_party_metrics(period_edges, focus_parties)
    network_summary = compute_network_summary(period_edges, focus_parties)
    monthly_panel = build_monthly_panel(term, mentions, focus_parties)

    macro_panel_available = bool((PROJECT_ROOT / f"Term_{term}" / "CSVs" / f"term_{term}_stage3_stage7_monthly_network_macro_panel.csv").exists())
    return DirectedArtifacts(
        term=term,
        focus_parties=focus_parties,
        mentions=mentions,
        period_edges=period_edges,
        yearly_edges=yearly_edges,
        monthly_panel=monthly_panel,
        party_metrics=party_metrics,
        network_summary=network_summary,
        hypothesis_coefficients=pd.DataFrame(),
        hypothesis_summary=pd.DataFrame(),
        granger_results=pd.DataFrame(),
        metadata={
            "used_existing_outputs": False,
            "speech_rows": int(len(speech)),
            "macro_panel_available": macro_panel_available,
        },
    )


def persist_artifacts(term: str, artifacts: DirectedArtifacts) -> list[Path]:
    csvs_dir, figures_dir, notes_dir = ensure_output_dirs(term)
    created_files: list[Path] = []

    csv_map = {
        "mention_rows": artifacts.mentions,
        "period_edges": artifacts.period_edges,
        "yearly_edges": artifacts.yearly_edges,
        "monthly_panel": artifacts.monthly_panel,
        "party_metrics": artifacts.party_metrics,
        "network_summary": artifacts.network_summary,
        "hypothesis_coefficients": artifacts.hypothesis_coefficients,
        "hypothesis_summary": artifacts.hypothesis_summary,
        "granger_results": artifacts.granger_results,
    }
    for stem, frame in csv_map.items():
        path = csvs_dir / f"term_{term}_{DIRECTED_PREFIX}_{stem}.csv"
        saved = save_table_if_nonempty(frame, path)
        if saved is not None:
            created_files.append(saved)

    main_network_path = plot_main_networks(term, artifacts.period_edges, artifacts.focus_parties, figures_dir)
    yearly_negative_path = plot_yearly_flow_networks(
        term,
        artifacts.yearly_edges,
        artifacts.period_edges,
        artifacts.focus_parties,
        figures_dir,
        "negative",
        "negative_weight",
        "yearly_negative_networks",
        f"Term {term} Yearly Negative Targeting Flows",
    )
    yearly_positive_path = plot_yearly_flow_networks(
        term,
        artifacts.yearly_edges,
        artifacts.period_edges,
        artifacts.focus_parties,
        figures_dir,
        "positive",
        "positive_weight",
        "yearly_positive_networks",
        f"Term {term} Yearly Positive Support Flows",
    )
    heatmap_path = plot_interaction_heatmaps(term, artifacts.period_edges, artifacts.focus_parties, figures_dir)
    role_path = plot_role_dashboard(term, artifacts.party_metrics, artifacts.focus_parties, figures_dir)
    created_files.extend([main_network_path, yearly_negative_path, yearly_positive_path, heatmap_path, role_path])

    hypothesis_figure = figures_dir / f"term_{term}_{DIRECTED_PREFIX}_hypothesis_coefficients.png"
    if artifacts.hypothesis_coefficients.empty:
        placeholder_message = (
            "The readable network figures were regenerated for this term, "
            "but no reusable directed hypothesis table was available."
        )
        created_files.append(plot_hypothesis_placeholder(term, figures_dir, placeholder_message))
    elif hypothesis_figure.exists():
        created_files.append(hypothesis_figure)

    summary_path = write_summary_note(term, artifacts, notes_dir, created_files)
    created_files.append(summary_path)
    output_index = write_output_index(term, notes_dir, created_files)
    created_files.append(output_index)
    return created_files


def main() -> None:
    args = parse_args()
    for raw_term in args.terms:
        term = str(raw_term)
        existing = None if args.force_recompute else load_existing_artifacts(term)
        artifacts = existing if existing is not None else compute_artifacts(term, args.chunk_size)
        persist_artifacts(term, artifacts)
        print(
            f"[directed] term={term} "
            f"focus={','.join(party_short_label(party) for party in artifacts.focus_parties)} "
            f"used_existing={artifacts.metadata.get('used_existing_outputs', False)}"
        )


if __name__ == "__main__":
    main()
