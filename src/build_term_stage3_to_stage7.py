from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd
from scipy import stats

import build_beme_layered_networks as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ECONOMY_DIR = PROJECT_ROOT / "economy_data" / "preprocessed"
MPLCONFIGDIR = PROJECT_ROOT / "Other" / ".mplconfig"
CACHE_DIR = PROJECT_ROOT / "Other" / ".cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import seaborn as sns

STAGE3_STAGE7_PREFIX = "stage3_stage7"
TERM_KEY_PARTIES = {
    "akp": "Adalet ve Kalkınma Partisi",
    "chp": "Cumhuriyet Halk Partisi",
    "anap": "Anavatan Partisi",
    "dyp": "Doğru Yol Partisi",
    "bagimsiz": "Bağımsız",
}
TERM_KEY_CONCEPTS = {
    "economy": "Ekonomi",
    "democracy": "Demokrasi",
    "european_union": "Avrupa Birligi",
    "law_state": "Hukuk Devleti",
    "constitution": "Anayasa",
    "security": "Guvenlik",
    "inflation": "Enflasyon",
    "interest": "Faiz",
}
CONCEPT_TO_SLUG = {label: slug for slug, label in TERM_KEY_CONCEPTS.items()}
VARIABLE_DISPLAY_LABELS = {
    "opposition_speeches_per_active_speaker": "Opposition speech intensity",
    "economy_negative_share": "Negative economy framing share",
    "inflation_negative_share": "Negative inflation framing share",
    "reform_positive_share": "Positive reform framing share",
    "crisis_negative_share": "Crisis/conflict framing share",
    "gov_opp_cosine_distance_total": "Government-opposition semantic distance",
    "gov_opp_cosine_distance_negative": "Negative semantic distance",
    "total_party_bloc_assortativity": "Party bloc assortativity",
    "democracy_positive_betweenness_centrality": "Bridge role of democracy node",
    "constitution_negative_betweenness_centrality": "Bridge role of constitution node",
    "financial_stress_index": "Financial stress index",
    "macro_volatility_index": "Macro volatility index",
    "macro_improvement_index": "Macro improvement index",
    "usd_return": "Monthly USD/TRY return",
    "bist_return": "Monthly BIST 100 return",
    "inflation_yoy": "Year-over-year inflation",
    "lead1_financial_stress_index": "Next-month financial stress",
    "lead1_bist_return": "Next-month BIST return",
    "european_union_positive_closeness_centrality": "EU node closeness",
    "crisis_month": "Crisis-month indicator",
}
DASHBOARD_PAIR_SPECS = (
    ("economy_negative_share", "macro_volatility_index", "Negative economy framing and macro volatility"),
    ("reform_positive_share", "macro_improvement_index", "Positive reform framing and macro improvement"),
    ("crisis_negative_share", "gov_opp_cosine_distance_total", "Crisis framing and semantic divergence"),
    ("opposition_speeches_per_active_speaker", "financial_stress_index", "Opposition intensity and financial stress"),
    ("democracy_positive_betweenness_centrality", "european_union_positive_closeness_centrality", "Democracy bridges and EU proximity"),
    ("constitution_negative_betweenness_centrality", "usd_return", "Constitutional tension and USD/TRY movement"),
)
SELECTED_MONTHLY_SUMMARY_METRICS = [
    "party_projection_density",
    "concept_projection_density",
    "party_bloc_assortativity",
    "avg_party_closeness",
    "avg_party_betweenness",
    "avg_concept_closeness",
    "avg_concept_betweenness",
    "party_avg_path_lcc",
    "concept_avg_path_lcc",
]
SELECTED_NODE_METRICS = [
    "weighted_degree",
    "closeness_centrality",
    "betweenness_centrality",
    "eigenvector_centrality",
]
MONTHLY_EVENTS = (
    {"event_id": "tezkere_1mart", "label": "March 1 Motion", "date": pd.Timestamp("2003-03-01"), "pre_months": 4, "post_months": 4},
    {"event_id": "ab_muzakereleri", "label": "EU Accession Talks", "date": pd.Timestamp("2005-10-01"), "pre_months": 6, "post_months": 6},
    {"event_id": "367_e_muhtira", "label": "367 Crisis / e-Memorandum", "date": pd.Timestamp("2007-04-01"), "pre_months": 6, "post_months": 4},
)
MARKET_EVENTS = (
    {"event_id": "election_2002", "label": "2002 General Election", "date": pd.Timestamp("2002-11-03"), "window_days": 15},
    {"event_id": "tezkere_1mart", "label": "March 1 Motion", "date": pd.Timestamp("2003-03-01"), "window_days": 15},
    {"event_id": "ab_muzakereleri", "label": "EU Accession Talks", "date": pd.Timestamp("2005-10-03"), "window_days": 15},
    {"event_id": "367_e_muhtira", "label": "367 Crisis / e-Memorandum", "date": pd.Timestamp("2007-04-27"), "window_days": 15},
)
HYPOTHESIS_MODELS = (
    {
        "hypothesis_id": "H1",
        "label": "Kriz aylarinda muhalefet konusma yogunlugu artar",
        "family": "speech_intensity",
        "outcome": "opposition_speeches_per_active_speaker",
        "predictors": ["crisis_month"],
        "controls": ["opposition_speeches_per_active_speaker_lag1"],
        "primary_predictor": "crisis_month",
        "expected_sign": "+",
    },
    {
        "hypothesis_id": "H2",
        "label": "Ekonomi ve enflasyon salience'i makro oynakligi yukari iter",
        "family": "macro_volatility",
        "outcome": "macro_volatility_index",
        "predictors": ["economy_negative_share", "inflation_negative_share"],
        "controls": ["macro_volatility_index_lag1"],
        "primary_predictor": "economy_negative_share",
        "expected_sign": "+",
    },
    {
        "hypothesis_id": "H3",
        "label": "Kriz salience'i iktidar-muhalefet semantik mesafesini artirir",
        "family": "polarization",
        "outcome": "gov_opp_cosine_distance_total",
        "predictors": ["crisis_negative_share"],
        "controls": ["gov_opp_cosine_distance_total_lag1"],
        "primary_predictor": "crisis_negative_share",
        "expected_sign": "+",
    },
    {
        "hypothesis_id": "H4a",
        "label": "Negatif ekonomi soylemi gelecek ay finansal stresi ondeleyebilir",
        "family": "lead_macro",
        "outcome": "lead1_financial_stress_index",
        "predictors": ["economy_negative_share", "gov_opp_cosine_distance_negative"],
        "controls": ["financial_stress_index"],
        "primary_predictor": "economy_negative_share",
        "expected_sign": "+",
    },
    {
        "hypothesis_id": "H4b",
        "label": "Pozitif reform soylemi gelecek ay BIST getirisiyle iliskilidir",
        "family": "lead_macro",
        "outcome": "lead1_bist_return",
        "predictors": ["reform_positive_share", "democracy_positive_betweenness_centrality"],
        "controls": ["bist_return"],
        "primary_predictor": "reform_positive_share",
        "expected_sign": "+",
    },
)
EVENT_ID_TO_LABEL = {event["event_id"]: event["label"] for event in (*MONTHLY_EVENTS, *MARKET_EVENTS)}
HYPOTHESIS_ID_TO_LABEL = {item["hypothesis_id"]: item["label"] for item in HYPOTHESIS_MODELS}


@dataclass(frozen=True)
class LoadedInputs:
    speech: pd.DataFrame
    edges: pd.DataFrame
    macro_monthly: pd.DataFrame
    market_daily: pd.DataFrame
    month_index: pd.DatetimeIndex
    start_date: pd.Timestamp
    end_date: pd.Timestamp


def slugify(value: str) -> str:
    normalized = base.normalize_text(str(value))
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace("/", "_")
    normalized = normalized.replace("-", "_")
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized


def safe_divide(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series | float:
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        num = numerator if isinstance(numerator, pd.Series) else pd.Series(numerator, index=denominator.index)  # type: ignore[arg-type]
        den = denominator if isinstance(denominator, pd.Series) else pd.Series(denominator, index=numerator.index)  # type: ignore[arg-type]
        return num.divide(den.replace({0: np.nan}))
    if denominator in (0, 0.0) or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def ensure_output_dirs(term: str) -> tuple[Path, Path, Path]:
    csvs_dir, figures_dir, notes_dir = base.get_output_dirs(term)
    csvs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)
    return csvs_dir, figures_dir, notes_dir


def monthly_floor(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=numeric.index, dtype=float)
    return (numeric - numeric.mean()) / std


def row_mean_of_standardized(data: pd.DataFrame, signed_columns: list[tuple[str, float]]) -> pd.Series:
    frame = pd.DataFrame(index=data.index)
    for column, sign in signed_columns:
        frame[column] = zscore(pd.to_numeric(data[column], errors="coerce")) * sign
    return frame.mean(axis=1, skipna=True)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return np.nan
    cosine_sim = float(np.dot(left, right) / (left_norm * right_norm))
    return 1.0 - cosine_sim


def month_diff(date: pd.Timestamp, reference: pd.Timestamp) -> int:
    return (date.year - reference.year) * 12 + (date.month - reference.month)


def numeric_columns(frame: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [column for column in frame.columns if column not in exclude and pd.api.types.is_numeric_dtype(frame[column])]


def display_label(raw_name: object) -> str:
    name = str(raw_name)
    if name in VARIABLE_DISPLAY_LABELS:
        return VARIABLE_DISPLAY_LABELS[name]
    if name in EVENT_ID_TO_LABEL:
        return EVENT_ID_TO_LABEL[name]
    if name.endswith("_lag1"):
        return f"{display_label(name[:-5])} (lagged 1 month)"
    if name.startswith("lead1_"):
        return f"Next-month {display_label(name[6:]).lower()}"
    if name.startswith("lead2_"):
        return f"{display_label(name[6:])} two months ahead"
    if name.startswith("lead3_"):
        return f"{display_label(name[6:])} three months ahead"
    return name.replace("_", " ")


def wrapped_label(raw_name: object, width: int = 24) -> str:
    return fill(display_label(raw_name), width=width)


def load_inputs(term: str) -> LoadedInputs:
    term_start, term_end = base.TERM_WINDOWS[term]
    speech = pd.read_csv(PROJECT_ROOT / f"Term_{term}" / "CSVs" / f"term_{term}_beme_speech_labels.csv")
    edges = pd.read_csv(PROJECT_ROOT / f"Term_{term}" / "CSVs" / f"term_{term}_beme_concept_edges.csv")
    macro_monthly = pd.read_csv(ECONOMY_DIR / "economy_monthly_macro.csv")
    market_daily = pd.read_csv(ECONOMY_DIR / "economy_daily_market.csv")

    speech["date"] = pd.to_datetime(speech["date"])
    edges["date"] = pd.to_datetime(edges["date"])
    macro_monthly["date"] = pd.to_datetime(macro_monthly["date"])
    market_daily["date"] = pd.to_datetime(market_daily["date"])

    speech = speech[(speech["date"] >= term_start) & (speech["date"] <= term_end)].copy()
    edges = edges[(edges["date"] >= term_start) & (edges["date"] <= term_end)].copy()
    macro_monthly = macro_monthly[(macro_monthly["date"] >= term_start) & (macro_monthly["date"] <= term_end)].copy()
    market_daily = market_daily[(market_daily["date"] >= (term_start - pd.Timedelta(days=30))) & (market_daily["date"] <= term_end)].copy()

    speech["month_date"] = monthly_floor(speech["date"])
    edges["month_date"] = monthly_floor(edges["date"])
    macro_monthly = macro_monthly.sort_values("date").reset_index(drop=True)
    market_daily = market_daily.sort_values("date").reset_index(drop=True)
    month_index = pd.date_range(term_start.to_period("M").to_timestamp(), term_end.to_period("M").to_timestamp(), freq="MS")
    return LoadedInputs(
        speech=speech.reset_index(drop=True),
        edges=edges.reset_index(drop=True),
        macro_monthly=macro_monthly.reset_index(drop=True),
        market_daily=market_daily.reset_index(drop=True),
        month_index=month_index,
        start_date=term_start,
        end_date=term_end,
    )


def build_monthly_speech_panel(speech: pd.DataFrame, month_index: pd.DatetimeIndex) -> pd.DataFrame:
    panel = pd.DataFrame({"date": month_index})
    if speech.empty:
        return panel

    overall = (
        speech.groupby("month_date", dropna=False)
        .agg(
            speech_count=("speech_id", "nunique"),
            unique_speakers=("speaker", "nunique"),
            word_count_total=("word_count", "sum"),
            avg_words_per_speech=("word_count", "mean"),
            avg_beme_score=("beme_score", "mean"),
            positive_speech_share=("beme_label", lambda values: float((values == "positive").mean())),
            negative_speech_share=("beme_label", lambda values: float((values == "negative").mean())),
            mixed_speech_share=("beme_label", lambda values: float((values == "mixed").mean())),
        )
        .reset_index()
        .rename(columns={"month_date": "date"})
    )
    panel = panel.merge(overall, on="date", how="left")

    bloc = (
        speech.groupby(["month_date", "party_bloc"], dropna=False)
        .agg(
            speech_count=("speech_id", "nunique"),
            unique_speakers=("speaker", "nunique"),
            word_count_total=("word_count", "sum"),
            positive_speech_share=("beme_label", lambda values: float((values == "positive").mean())),
            negative_speech_share=("beme_label", lambda values: float((values == "negative").mean())),
        )
        .reset_index()
        .rename(columns={"month_date": "date"})
    )
    for metric in ["speech_count", "unique_speakers", "word_count_total", "positive_speech_share", "negative_speech_share"]:
        pivot = bloc.pivot(index="date", columns="party_bloc", values=metric)
        pivot = pivot.rename(columns={column: f"{column}_{metric}" for column in pivot.columns}).reset_index()
        panel = panel.merge(pivot, on="date", how="left")

    for bloc_name in ["government", "opposition"]:
        count_col = f"{bloc_name}_speech_count"
        speakers_col = f"{bloc_name}_unique_speakers"
        if count_col not in panel.columns:
            panel[count_col] = np.nan
        if speakers_col not in panel.columns:
            panel[speakers_col] = np.nan
        panel[f"{bloc_name}_speeches_per_active_speaker"] = safe_divide(panel[count_col], panel[speakers_col])

    panel["government_speech_share"] = safe_divide(panel.get("government_speech_count", np.nan), panel["speech_count"])
    panel["opposition_speech_share"] = safe_divide(panel.get("opposition_speech_count", np.nan), panel["speech_count"])
    return panel.sort_values("date").reset_index(drop=True)


def aggregate_monthly_layer_edges(edge_df: pd.DataFrame, layer: str) -> pd.DataFrame:
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

    layer_df["month_date"] = monthly_floor(layer_df["date"])
    layer_df["layer_weight_component"] = layer_df.apply(
        lambda row: row["salience_weight"] * base.clipped_edge_strength(float(row["beme_score"]), layer),
        axis=1,
    )
    if layer == "total":
        layer_df["layer_weight_component"] = layer_df["salience_weight"]

    grouped = (
        layer_df.groupby(
            ["month_date", "party", "concept_slug", "concept", "concept_category"],
            dropna=False,
        )
        .agg(
            speech_count=("speech_id", "nunique"),
            mention_count=("mention_count", "sum"),
            beme_positive_hits=("beme_positive_hits", "sum"),
            beme_negative_hits=("beme_negative_hits", "sum"),
            salience_weight=("salience_weight", "sum"),
            layer_weight=("layer_weight_component", "sum"),
        )
        .reset_index()
        .rename(columns={"month_date": "date"})
    )
    grouped["beme_score"] = (
        (grouped["beme_positive_hits"] - grouped["beme_negative_hits"])
        / (grouped["beme_positive_hits"] + grouped["beme_negative_hits"] + 1)
    )
    grouped["layer"] = layer
    return grouped.sort_values(["date", "layer_weight"], ascending=[True, False]).reset_index(drop=True)


def build_monthly_network_outputs(term: str, edges: pd.DataFrame, month_index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_edges_frames: list[pd.DataFrame] = []
    monthly_party_metric_frames: list[pd.DataFrame] = []
    monthly_concept_metric_frames: list[pd.DataFrame] = []
    monthly_summary_frames: list[pd.DataFrame] = []

    for layer in ["total", "positive", "negative"]:
        layer_edges = aggregate_monthly_layer_edges(edges, layer)
        if not layer_edges.empty:
            monthly_edges_frames.append(layer_edges)
        for month_date in month_index:
            month_subset = layer_edges[layer_edges["date"] == month_date].copy() if not layer_edges.empty else pd.DataFrame()
            party_metrics = base.build_party_metrics(term, month_subset, layer)
            concept_metrics = base.build_concept_metrics(month_subset, layer)
            summary = base.build_network_summary(term, month_subset, layer, party_metrics, concept_metrics)

            if not party_metrics.empty:
                party_metrics["date"] = month_date
                monthly_party_metric_frames.append(party_metrics)
            if not concept_metrics.empty:
                concept_metrics["date"] = month_date
                monthly_concept_metric_frames.append(concept_metrics)

            summary["date"] = month_date
            monthly_summary_frames.append(summary)

    monthly_edges = pd.concat(monthly_edges_frames, ignore_index=True) if monthly_edges_frames else pd.DataFrame()
    monthly_party_metrics = pd.concat(monthly_party_metric_frames, ignore_index=True) if monthly_party_metric_frames else pd.DataFrame()
    monthly_concept_metrics = pd.concat(monthly_concept_metric_frames, ignore_index=True) if monthly_concept_metric_frames else pd.DataFrame()
    monthly_summary = pd.concat(monthly_summary_frames, ignore_index=True) if monthly_summary_frames else pd.DataFrame()
    return monthly_edges, monthly_party_metrics, monthly_concept_metrics, monthly_summary


def build_topic_ownership(monthly_edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if monthly_edges.empty:
        return pd.DataFrame(), pd.DataFrame()

    monthly = monthly_edges.copy()
    monthly["year"] = pd.to_datetime(monthly["date"]).dt.year
    monthly["concept_total_weight"] = monthly.groupby(["date", "layer", "concept_slug"])["layer_weight"].transform("sum")
    monthly["party_share_in_concept"] = safe_divide(monthly["layer_weight"], monthly["concept_total_weight"])
    yearly = (
        monthly.groupby(["year", "layer", "party", "concept_slug", "concept"], dropna=False)
        .agg(layer_weight=("layer_weight", "sum"))
        .reset_index()
    )
    yearly["concept_total_weight"] = yearly.groupby(["year", "layer", "concept_slug"])["layer_weight"].transform("sum")
    yearly["party_share_in_concept"] = safe_divide(yearly["layer_weight"], yearly["concept_total_weight"])
    return monthly.sort_values(["date", "layer", "concept_slug", "party"]).reset_index(drop=True), yearly.sort_values(["year", "layer", "concept_slug", "party"]).reset_index(drop=True)


def build_layer_feature_frame(monthly_edges: pd.DataFrame, month_index: pd.DatetimeIndex) -> pd.DataFrame:
    panel = pd.DataFrame({"date": month_index})
    if monthly_edges.empty:
        return panel

    total_salience = (
        monthly_edges[monthly_edges["layer"] == "total"]
        .groupby("date", dropna=False)["layer_weight"]
        .sum()
        .rename("total_salience_weight")
        .reset_index()
    )
    panel = panel.merge(total_salience, on="date", how="left")

    layer_totals = (
        monthly_edges.groupby(["date", "layer"], dropna=False)["layer_weight"]
        .sum()
        .reset_index()
        .pivot(index="date", columns="layer", values="layer_weight")
        .rename(columns=lambda layer: f"{layer}_layer_weight_total")
        .reset_index()
    )
    panel = panel.merge(layer_totals, on="date", how="left")

    category_weights = (
        monthly_edges.groupby(["date", "layer", "concept_category"], dropna=False)["layer_weight"]
        .sum()
        .reset_index()
    )
    for category in sorted(category_weights["concept_category"].dropna().unique()):
        subset = category_weights[category_weights["concept_category"] == category].copy()
        pivot = subset.pivot(index="date", columns="layer", values="layer_weight").reset_index()
        for layer in ["total", "positive", "negative"]:
            source_col = layer if layer in pivot.columns else None
            target_col = f"{slugify(category)}_{layer}_weight"
            pivot[target_col] = pivot[source_col] if source_col else np.nan
        keep_cols = ["date"] + [column for column in pivot.columns if column.endswith("_weight")]
        panel = panel.merge(pivot[keep_cols], on="date", how="left")

    concept_weights = (
        monthly_edges.groupby(["date", "layer", "concept_slug"], dropna=False)["layer_weight"]
        .sum()
        .reset_index()
    )
    for concept_slug in sorted(concept_weights["concept_slug"].dropna().unique()):
        subset = concept_weights[concept_weights["concept_slug"] == concept_slug].copy()
        pivot = subset.pivot(index="date", columns="layer", values="layer_weight").reset_index()
        for layer in ["total", "positive", "negative"]:
            source_col = layer if layer in pivot.columns else None
            target_col = f"{concept_slug}_{layer}_weight"
            pivot[target_col] = pivot[source_col] if source_col else np.nan
        keep_cols = ["date"] + [column for column in pivot.columns if column.endswith("_weight")]
        panel = panel.merge(pivot[keep_cols], on="date", how="left")

    if "total_layer_weight_total" not in panel.columns:
        panel["total_layer_weight_total"] = panel.get("total_salience_weight")
    panel["total_layer_weight_total"] = panel["total_layer_weight_total"].combine_first(panel.get("total_salience_weight"))

    for column in [column for column in panel.columns if column.endswith("_weight")]:
        if column in {"total_layer_weight_total", "total_salience_weight"}:
            continue
        panel[column.replace("_weight", "_share")] = safe_divide(panel[column], panel["total_layer_weight_total"])

    panel["reform_positive_share"] = (
        panel.get("democratic_reform_positive_weight", np.nan).fillna(0.0) / panel["total_layer_weight_total"].replace({0: np.nan})
    )
    crisis_weight = panel.get("constitutional_conflict_negative_weight", 0.0) + panel.get("security_conflict_negative_weight", 0.0)
    panel["crisis_negative_share"] = safe_divide(crisis_weight, panel["total_layer_weight_total"])
    panel["positive_negative_salience_ratio"] = safe_divide(
        panel.get("positive_layer_weight_total", np.nan),
        panel.get("negative_layer_weight_total", np.nan),
    )
    return panel.sort_values("date").reset_index(drop=True).copy()


def build_semantic_distance_frame(term: str, monthly_edges: pd.DataFrame, month_index: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month_date in month_index:
        month_subset = monthly_edges[monthly_edges["date"] == month_date].copy()
        for layer in ["total", "positive", "negative"]:
            layer_subset = month_subset[month_subset["layer"] == layer].copy()
            if layer_subset.empty:
                rows.append(
                    {
                        "date": month_date,
                        "layer": layer,
                        "gov_opp_cosine_distance": np.nan,
                        "akp_chp_cosine_distance": np.nan,
                    }
                )
                continue

            matrix = layer_subset.pivot_table(index="party", columns="concept_slug", values="layer_weight", aggfunc="sum", fill_value=0.0)
            gov_parties = [party for party in matrix.index if base.party_bloc(term, party) == "government"]
            opp_parties = [party for party in matrix.index if base.party_bloc(term, party) == "opposition"]
            gov_vector = matrix.loc[gov_parties].mean(axis=0).to_numpy(dtype=float) if gov_parties else np.array([])
            opp_vector = matrix.loc[opp_parties].mean(axis=0).to_numpy(dtype=float) if opp_parties else np.array([])
            gov_opp_distance = cosine_distance(gov_vector, opp_vector) if gov_vector.size and opp_vector.size else np.nan

            akp_distance = np.nan
            if TERM_KEY_PARTIES["akp"] in matrix.index and TERM_KEY_PARTIES["chp"] in matrix.index:
                akp_distance = cosine_distance(
                    matrix.loc[TERM_KEY_PARTIES["akp"]].to_numpy(dtype=float),
                    matrix.loc[TERM_KEY_PARTIES["chp"]].to_numpy(dtype=float),
                )

            rows.append(
                {
                    "date": month_date,
                    "layer": layer,
                    "gov_opp_cosine_distance": gov_opp_distance,
                    "akp_chp_cosine_distance": akp_distance,
                }
            )

    distance_df = pd.DataFrame(rows)
    if distance_df.empty:
        return pd.DataFrame({"date": month_index})

    wide_frames = []
    for metric in ["gov_opp_cosine_distance", "akp_chp_cosine_distance"]:
        pivot = distance_df.pivot(index="date", columns="layer", values=metric)
        pivot = pivot.rename(columns=lambda layer: f"{metric}_{layer}").reset_index()
        wide_frames.append(pivot)

    merged = pd.DataFrame({"date": month_index})
    for frame in wide_frames:
        merged = merged.merge(frame, on="date", how="left")
    return merged.sort_values("date").reset_index(drop=True)


def pivot_summary_features(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    wide = pd.DataFrame({"date": sorted(summary["date"].dropna().unique())})
    for metric in SELECTED_MONTHLY_SUMMARY_METRICS:
        pivot = summary.pivot(index="date", columns="layer", values=metric).reset_index()
        rename_map = {layer: f"{layer}_{metric}" for layer in ["total", "positive", "negative"] if layer in pivot.columns}
        pivot = pivot.rename(columns=rename_map)
        wide = wide.merge(pivot, on="date", how="left")
    return wide.sort_values("date").reset_index(drop=True)


def pivot_party_metrics(term: str, party_metrics: pd.DataFrame) -> pd.DataFrame:
    if party_metrics.empty:
        return pd.DataFrame()

    filtered = party_metrics[party_metrics["party"].isin(TERM_KEY_PARTIES.values())].copy()
    filtered["party_key"] = filtered["party"].map({value: key for key, value in TERM_KEY_PARTIES.items()})
    filtered["party_bloc"] = filtered["party"].map(lambda party: base.party_bloc(term, party))

    wide = pd.DataFrame({"date": sorted(filtered["date"].dropna().unique())})
    for metric in SELECTED_NODE_METRICS:
        pivot = filtered.pivot_table(index="date", columns=["party_key", "layer"], values=metric, aggfunc="mean")
        if not pivot.empty:
            pivot.columns = [f"{party_key}_{layer}_{metric}" for party_key, layer in pivot.columns]
            wide = wide.merge(pivot.reset_index(), on="date", how="left")

    bloc = (
        filtered.groupby(["date", "party_bloc", "layer"], dropna=False)[SELECTED_NODE_METRICS]
        .mean()
        .reset_index()
    )
    for metric in SELECTED_NODE_METRICS:
        pivot = bloc.pivot_table(index="date", columns=["party_bloc", "layer"], values=metric, aggfunc="mean")
        if not pivot.empty:
            pivot.columns = [f"{bloc_name}_{layer}_{metric}" for bloc_name, layer in pivot.columns]
            wide = wide.merge(pivot.reset_index(), on="date", how="left")

    return wide.sort_values("date").reset_index(drop=True)


def pivot_concept_metrics(concept_metrics: pd.DataFrame) -> pd.DataFrame:
    if concept_metrics.empty:
        return pd.DataFrame()

    filtered = concept_metrics[concept_metrics["concept"].isin(TERM_KEY_CONCEPTS.values())].copy()
    filtered["concept_slug_clean"] = filtered["concept"].map(CONCEPT_TO_SLUG)
    wide = pd.DataFrame({"date": sorted(filtered["date"].dropna().unique())})
    for metric in ["weighted_degree", "closeness_centrality", "betweenness_centrality"]:
        pivot = filtered.pivot_table(index="date", columns=["concept_slug_clean", "layer"], values=metric, aggfunc="mean")
        if not pivot.empty:
            pivot.columns = [f"{concept_slug}_{layer}_{metric}" for concept_slug, layer in pivot.columns]
            wide = wide.merge(pivot.reset_index(), on="date", how="left")
    return wide.sort_values("date").reset_index(drop=True)


def build_macro_panel(macro_monthly: pd.DataFrame, month_index: pd.DatetimeIndex) -> pd.DataFrame:
    macro = pd.DataFrame({"date": month_index}).merge(macro_monthly, on="date", how="left")
    macro["usd_return"] = pd.to_numeric(macro.get("usd_try_return_month_pct"), errors="coerce")
    macro["bist_return"] = pd.to_numeric(macro.get("bist100_return_month_pct"), errors="coerce")
    macro["inflation_yoy"] = pd.to_numeric(macro.get("inflation_yoy_percent_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(macro.get("cpi_yoy_percent_tuik"), errors="coerce")
    )
    macro["inflation_mom"] = pd.to_numeric(macro.get("inflation_mom_percent_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(macro.get("cpi_mom_percent_tuik"), errors="coerce")
    )
    macro["policy_rate"] = pd.to_numeric(macro.get("policy_rate_percent_tcmb"), errors="coerce").combine_first(
        pd.to_numeric(macro.get("short_term_interest_rate_percent"), errors="coerce")
    )
    macro["reserves_usd_million"] = pd.to_numeric(macro.get("gross_reserves_usd_million_tcmb"), errors="coerce")
    macro["industrial_production"] = pd.to_numeric(macro.get("industrial_production_index_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(macro.get("industrial_production_index_tuik"), errors="coerce")
    ).combine_first(pd.to_numeric(macro.get("industrial_production_index"), errors="coerce"))

    macro["reserves_mom_pct"] = macro["reserves_usd_million"].pct_change() * 100.0
    macro["industrial_production_mom_pct"] = macro["industrial_production"].pct_change() * 100.0
    macro["macro_volatility_index"] = row_mean_of_standardized(
        macro,
        [
            ("usd_return", 1.0),
            ("bist_return", -1.0),
            ("inflation_mom", 1.0),
        ],
    )
    macro["financial_stress_index"] = row_mean_of_standardized(
        macro,
        [
            ("inflation_yoy", 1.0),
            ("usd_return", 1.0),
            ("bist_return", -1.0),
            ("policy_rate", 1.0),
            ("reserves_mom_pct", -1.0),
        ],
    )
    macro["macro_improvement_index"] = -macro["financial_stress_index"]
    return macro


def add_time_series_leads_and_lags(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.sort_values("date").reset_index(drop=True).copy()
    lag_targets = [
        "opposition_speeches_per_active_speaker",
        "macro_volatility_index",
        "gov_opp_cosine_distance_total",
        "financial_stress_index",
        "bist_return",
        "usd_return",
        "inflation_yoy",
    ]
    for column in lag_targets:
        if column in output.columns:
            output[f"{column}_lag1"] = output[column].shift(1)

    lead_targets = [
        "financial_stress_index",
        "macro_volatility_index",
        "bist_return",
        "usd_return",
        "inflation_yoy",
    ]
    for column in lead_targets:
        if column in output.columns:
            output[f"lead1_{column}"] = output[column].shift(-1)
            output[f"lead2_{column}"] = output[column].shift(-2)
            output[f"lead3_{column}"] = output[column].shift(-3)

    crisis_threshold = output["financial_stress_index"].quantile(0.75)
    output["crisis_month"] = (output["financial_stress_index"] >= crisis_threshold).astype(float)
    return output


def build_yearly_analysis_panel(panel: pd.DataFrame) -> pd.DataFrame:
    yearly_columns = [
        "speech_count",
        "word_count_total",
        "positive_speech_share",
        "negative_speech_share",
        "government_speech_share",
        "opposition_speech_share",
        "opposition_speeches_per_active_speaker",
        "economy_total_share",
        "economy_positive_share",
        "economy_negative_share",
        "inflation_negative_share",
        "reform_positive_share",
        "crisis_negative_share",
        "gov_opp_cosine_distance_total",
        "gov_opp_cosine_distance_negative",
        "total_party_bloc_assortativity",
        "positive_avg_party_closeness",
        "negative_avg_party_betweenness",
        "democracy_positive_betweenness_centrality",
        "european_union_positive_closeness_centrality",
        "constitution_negative_betweenness_centrality",
        "security_negative_closeness_centrality",
        "financial_stress_index",
        "macro_volatility_index",
        "usd_return",
        "bist_return",
        "inflation_yoy",
        "policy_rate",
        "reserves_usd_million",
    ]
    available_columns = [column for column in yearly_columns if column in panel.columns]
    if not available_columns:
        return pd.DataFrame()
    yearly = panel.copy()
    yearly["year"] = pd.to_datetime(yearly["date"]).dt.year
    agg_map = {column: "mean" for column in available_columns}
    for column in ["speech_count", "word_count_total"]:
        if column in agg_map:
            agg_map[column] = "sum"
    grouped = yearly.groupby("year", dropna=False).agg(agg_map).reset_index()
    grouped["months_with_speeches"] = yearly.groupby("year")["speech_count"].apply(lambda values: int(values.fillna(0).gt(0).sum())).values
    return grouped.sort_values("year").reset_index(drop=True)


def build_analysis_panel(
    term: str,
    inputs: LoadedInputs,
    monthly_speech: pd.DataFrame,
    monthly_edges: pd.DataFrame,
    monthly_party_metrics: pd.DataFrame,
    monthly_concept_metrics: pd.DataFrame,
    monthly_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    macro_panel = build_macro_panel(inputs.macro_monthly, inputs.month_index)
    layer_features = build_layer_feature_frame(monthly_edges, inputs.month_index)
    distance_features = build_semantic_distance_frame(term, monthly_edges, inputs.month_index)
    summary_features = pivot_summary_features(monthly_summary)
    party_features = pivot_party_metrics(term, monthly_party_metrics)
    concept_features = pivot_concept_metrics(monthly_concept_metrics)

    panel = pd.DataFrame({"date": inputs.month_index})
    for frame in [macro_panel, monthly_speech, layer_features, distance_features, summary_features, party_features, concept_features]:
        if not frame.empty:
            panel = panel.merge(frame, on="date", how="left")
    panel = panel.loc[:, ~panel.columns.duplicated()].copy()
    panel = add_time_series_leads_and_lags(panel)

    monthly_topic_ownership, yearly_topic_ownership = build_topic_ownership(monthly_edges)
    yearly_panel = build_yearly_analysis_panel(panel)
    return panel, monthly_topic_ownership, yearly_topic_ownership, yearly_panel


def ordinary_least_squares(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    ssr = float(np.dot(residuals, residuals))
    sst = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = 1.0 - (ssr / sst) if sst > 0 else np.nan
    return beta, residuals, ssr, r2


def newey_west_covariance(x: np.ndarray, residuals: np.ndarray, lag: int) -> np.ndarray:
    n_obs, n_params = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    score = residuals[:, None] * x
    omega = score.T @ score
    for lag_step in range(1, lag + 1):
        weight = 1.0 - (lag_step / (lag + 1.0))
        gamma = score[lag_step:].T @ score[:-lag_step]
        omega += weight * (gamma + gamma.T)
    return xtx_inv @ omega @ xtx_inv


def fit_newey_west_model(panel: pd.DataFrame, outcome: str, predictors: list[str], controls: list[str], nw_lag: int = 1, standardize: bool = False) -> tuple[pd.DataFrame, dict[str, float]]:
    columns = [outcome] + predictors + controls
    model_df = panel[["date"] + columns].copy()
    model_df = model_df.dropna()
    if model_df.shape[0] < (len(columns) + 6):
        return pd.DataFrame(), {"n_obs": float(model_df.shape[0]), "r_squared": np.nan, "adj_r_squared": np.nan}

    if standardize:
        for column in columns:
            if pd.api.types.is_numeric_dtype(model_df[column]):
                model_df[column] = zscore(model_df[column])

    y = model_df[outcome].to_numpy(dtype=float)
    x_columns = predictors + controls
    x = model_df[x_columns].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(model_df)), x])
    regressor_names = ["intercept"] + x_columns

    beta, residuals, _, r_squared = ordinary_least_squares(y, x)
    cov = newey_west_covariance(x, residuals, lag=nw_lag)
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se
    degrees_freedom = max(len(model_df) - x.shape[1], 1)
    p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=degrees_freedom))
    critical = stats.t.ppf(0.975, df=degrees_freedom)
    ci_low = beta - (critical * se)
    ci_high = beta + (critical * se)
    adj_r_squared = 1.0 - ((1.0 - r_squared) * (len(model_df) - 1) / degrees_freedom) if not np.isnan(r_squared) else np.nan

    coefficients = pd.DataFrame(
        {
            "term": panel.get("term", pd.Series(dtype=float)).iloc[0] if "term" in panel.columns and not panel.empty else np.nan,
            "outcome": outcome,
            "variable": regressor_names,
            "beta": beta,
            "std_error": se,
            "t_stat": t_stats,
            "p_value": p_values,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_obs": len(model_df),
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "standardized": standardize,
        }
    )
    return coefficients, {"n_obs": float(len(model_df)), "r_squared": r_squared, "adj_r_squared": adj_r_squared}


def build_hypothesis_registry() -> pd.DataFrame:
    rows = []
    for spec in HYPOTHESIS_MODELS:
        rows.append(
            {
                "hypothesis_id": spec["hypothesis_id"],
                "label": spec["label"],
                "family": spec["family"],
                "outcome": spec["outcome"],
                "predictors": ", ".join(spec["predictors"]),
                "controls": ", ".join(spec["controls"]),
                "primary_predictor": spec["primary_predictor"],
                "expected_sign": spec["expected_sign"],
            }
        )
    return pd.DataFrame(rows)


def run_hypothesis_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for spec in HYPOTHESIS_MODELS:
        raw_coeffs, raw_meta = fit_newey_west_model(panel, spec["outcome"], spec["predictors"], spec["controls"], nw_lag=1, standardize=False)
        std_coeffs, _ = fit_newey_west_model(panel, spec["outcome"], spec["predictors"], spec["controls"], nw_lag=1, standardize=True)
        if raw_coeffs.empty:
            summary_rows.append(
                {
                    "hypothesis_id": spec["hypothesis_id"],
                    "label": spec["label"],
                    "n_obs": raw_meta["n_obs"],
                    "r_squared": np.nan,
                    "adj_r_squared": np.nan,
                    "primary_predictor": spec["primary_predictor"],
                    "primary_beta": np.nan,
                    "primary_p_value": np.nan,
                    "primary_standardized_beta": np.nan,
                    "expected_sign": spec["expected_sign"],
                    "supported_at_10pct": False,
                }
            )
            continue

        raw_coeffs["hypothesis_id"] = spec["hypothesis_id"]
        raw_coeffs["label"] = spec["label"]
        std_coeffs = std_coeffs.rename(columns={"beta": "standardized_beta"})[["variable", "standardized_beta"]]
        merged = raw_coeffs.merge(std_coeffs, on="variable", how="left")
        coefficient_frames.append(merged)

        primary_row = merged[merged["variable"] == spec["primary_predictor"]].iloc[0]
        expected_positive = spec["expected_sign"] == "+"
        supported = bool(primary_row["p_value"] <= 0.10 and ((primary_row["beta"] > 0 and expected_positive) or (primary_row["beta"] < 0 and not expected_positive)))
        summary_rows.append(
            {
                "hypothesis_id": spec["hypothesis_id"],
                "label": spec["label"],
                "n_obs": int(primary_row["n_obs"]),
                "r_squared": raw_meta["r_squared"],
                "adj_r_squared": raw_meta["adj_r_squared"],
                "primary_predictor": spec["primary_predictor"],
                "primary_beta": primary_row["beta"],
                "primary_p_value": primary_row["p_value"],
                "primary_standardized_beta": primary_row["standardized_beta"],
                "expected_sign": spec["expected_sign"],
                "supported_at_10pct": supported,
            }
        )

    coefficients = pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return coefficients, summary


def compute_correlation_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_vars = [
        "opposition_speeches_per_active_speaker",
        "economy_negative_share",
        "inflation_negative_share",
        "reform_positive_share",
        "crisis_negative_share",
        "gov_opp_cosine_distance_total",
        "gov_opp_cosine_distance_negative",
        "total_party_bloc_assortativity",
        "democracy_positive_betweenness_centrality",
        "constitution_negative_betweenness_centrality",
        "financial_stress_index",
        "macro_volatility_index",
        "usd_return",
        "bist_return",
        "inflation_yoy",
        "lead1_financial_stress_index",
        "lead1_bist_return",
    ]
    available = [column for column in selected_vars if column in panel.columns]
    corr_rows: list[dict[str, object]] = []
    corr_matrix = pd.DataFrame(index=available, columns=available, dtype=float)
    for left in available:
        for right in available:
            if left == right:
                corr_matrix.loc[left, right] = 1.0
                corr_rows.append(
                    {
                        "var_left": left,
                        "var_right": right,
                        "n_obs": int(panel[left].notna().sum()),
                        "pearson_r": 1.0,
                        "pearson_p_value": 0.0,
                        "spearman_rho": 1.0,
                        "spearman_p_value": 0.0,
                    }
                )
                continue
            subset = panel[[left, right]].dropna()
            if subset.shape[0] < 8:
                corr_matrix.loc[left, right] = np.nan
                continue
            pearson_r, pearson_p = stats.pearsonr(subset[left], subset[right])
            spearman_r, spearman_p = stats.spearmanr(subset[left], subset[right])
            corr_matrix.loc[left, right] = pearson_r
            if left < right:
                corr_rows.append(
                    {
                        "var_left": left,
                        "var_right": right,
                        "n_obs": subset.shape[0],
                        "pearson_r": pearson_r,
                        "pearson_p_value": pearson_p,
                        "spearman_rho": spearman_r,
                        "spearman_p_value": spearman_p,
                    }
                )
    return corr_matrix, pd.DataFrame(corr_rows)


def granger_test(panel: pd.DataFrame, cause: str, effect: str, max_lag: int = 3) -> pd.DataFrame:
    series = panel[["date", cause, effect]].dropna().sort_values("date").reset_index(drop=True)
    if series.shape[0] < 12:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for lag in range(1, max_lag + 1):
        design = pd.DataFrame({"effect": series[effect]})
        for lag_step in range(1, lag + 1):
            design[f"{effect}_lag{lag_step}"] = series[effect].shift(lag_step)
            design[f"{cause}_lag{lag_step}"] = series[cause].shift(lag_step)
        design = design.dropna().reset_index(drop=True)
        if design.shape[0] <= (2 * lag + 4):
            continue

        y = design["effect"].to_numpy(dtype=float)
        restricted_cols = [f"{effect}_lag{lag_step}" for lag_step in range(1, lag + 1)]
        unrestricted_cols = restricted_cols + [f"{cause}_lag{lag_step}" for lag_step in range(1, lag + 1)]

        x_restricted = np.column_stack([np.ones(len(design)), design[restricted_cols].to_numpy(dtype=float)])
        x_unrestricted = np.column_stack([np.ones(len(design)), design[unrestricted_cols].to_numpy(dtype=float)])
        _, _, ssr_restricted, _ = ordinary_least_squares(y, x_restricted)
        beta_unrestricted, _, ssr_unrestricted, _ = ordinary_least_squares(y, x_unrestricted)

        df_num = lag
        df_den = len(design) - x_unrestricted.shape[1]
        if df_den <= 0:
            continue
        numerator = max(ssr_restricted - ssr_unrestricted, 0.0) / df_num
        denominator = ssr_unrestricted / df_den if ssr_unrestricted > 0 else np.nan
        f_stat = numerator / denominator if denominator and not np.isnan(denominator) else np.nan
        p_value = 1.0 - stats.f.cdf(f_stat, df_num, df_den) if not np.isnan(f_stat) else np.nan
        rows.append(
            {
                "cause": cause,
                "effect": effect,
                "lag": lag,
                "n_obs": len(design),
                "f_stat": f_stat,
                "p_value": p_value,
                "sum_cause_lag_beta": float(beta_unrestricted[-lag:].sum()),
            }
        )
    return pd.DataFrame(rows)


def run_granger_suite(panel: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("economy_negative_share", "financial_stress_index"),
        ("economy_negative_share", "usd_return"),
        ("reform_positive_share", "bist_return"),
        ("gov_opp_cosine_distance_negative", "financial_stress_index"),
    ]
    frames = []
    for cause, effect in pairs:
        if cause in panel.columns and effect in panel.columns:
            frames.append(granger_test(panel, cause, effect, max_lag=3))
            frames.append(granger_test(panel, effect, cause, max_lag=3))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def hedges_g(pre: pd.Series, post: pd.Series) -> float:
    pre = pre.dropna()
    post = post.dropna()
    if len(pre) < 2 or len(post) < 2:
        return np.nan
    pooled_var = (((len(pre) - 1) * pre.var(ddof=1)) + ((len(post) - 1) * post.var(ddof=1))) / (len(pre) + len(post) - 2)
    if pooled_var <= 0 or np.isnan(pooled_var):
        return np.nan
    d_value = (post.mean() - pre.mean()) / math.sqrt(pooled_var)
    correction = 1.0 - (3.0 / (4.0 * (len(pre) + len(post)) - 9.0))
    return d_value * correction


def run_event_studies(panel: pd.DataFrame, daily_market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    event_variables = [
        "reform_positive_share",
        "economy_negative_share",
        "crisis_negative_share",
        "gov_opp_cosine_distance_total",
        "gov_opp_cosine_distance_negative",
        "financial_stress_index",
        "macro_volatility_index",
        "usd_return",
        "bist_return",
    ]
    for event in MONTHLY_EVENTS:
        event_month = event["date"].to_period("M").to_timestamp()
        pre_mask = panel["date"].apply(lambda value: -event["pre_months"] <= month_diff(value, event_month) <= -1)
        post_mask = panel["date"].apply(lambda value: 0 <= month_diff(value, event_month) <= (event["post_months"] - 1))
        pre = panel[pre_mask].copy()
        post = panel[post_mask].copy()

        for variable in [column for column in event_variables if column in panel.columns]:
            pre_series = pre[variable].dropna()
            post_series = post[variable].dropna()
            if len(pre_series) >= 2 and len(post_series) >= 2:
                t_stat, t_p = stats.ttest_ind(post_series, pre_series, equal_var=False, nan_policy="omit")
                u_stat, u_p = stats.mannwhitneyu(post_series, pre_series, alternative="two-sided")
            else:
                t_stat, t_p, u_stat, u_p = (np.nan, np.nan, np.nan, np.nan)
            event_rows.append(
                {
                    "event_id": event["event_id"],
                    "event_label": event["label"],
                    "event_month": event_month,
                    "variable": variable,
                    "pre_n": len(pre_series),
                    "post_n": len(post_series),
                    "pre_mean": pre_series.mean() if not pre_series.empty else np.nan,
                    "post_mean": post_series.mean() if not post_series.empty else np.nan,
                    "mean_diff_post_minus_pre": (post_series.mean() - pre_series.mean()) if not pre_series.empty and not post_series.empty else np.nan,
                    "welch_t_stat": t_stat,
                    "welch_p_value": t_p,
                    "mannwhitney_u": u_stat,
                    "mannwhitney_p_value": u_p,
                    "hedges_g": hedges_g(pre_series, post_series),
                }
            )

        for _, row in panel.iterrows():
            relative_month = month_diff(pd.Timestamp(row["date"]), event_month)
            if relative_month < -event["pre_months"] or relative_month > (event["post_months"] - 1):
                continue
            profile_rows.append(
                {
                    "event_id": event["event_id"],
                    "event_label": event["label"],
                    "relative_month": relative_month,
                    "date": row["date"],
                    "reform_positive_share": row.get("reform_positive_share"),
                    "crisis_negative_share": row.get("crisis_negative_share"),
                    "gov_opp_cosine_distance_total": row.get("gov_opp_cosine_distance_total"),
                    "gov_opp_cosine_distance_negative": row.get("gov_opp_cosine_distance_negative"),
                    "financial_stress_index": row.get("financial_stress_index"),
                    "usd_return": row.get("usd_return"),
                    "bist_return": row.get("bist_return"),
                }
            )

    market_rows: list[dict[str, object]] = []
    if not daily_market.empty:
        market = daily_market.copy()
        market["usd_close"] = pd.to_numeric(market.get("usd_try_close_harmonized"), errors="coerce").combine_first(
            pd.to_numeric(market.get("usd_try_close"), errors="coerce")
        )
        market["bist_close"] = pd.to_numeric(market.get("bist100_close"), errors="coerce")

        for event in MARKET_EVENTS:
            anchor_candidates = market.loc[
                (market["date"] >= event["date"])
                & (market["date"].dt.dayofweek < 5)
                & (market["usd_close"].notna() | market["bist_close"].notna()),
                "date",
            ]
            if anchor_candidates.empty:
                continue
            anchor_date = pd.Timestamp(anchor_candidates.iloc[0])
            window = market[(market["date"] >= (anchor_date - pd.Timedelta(days=event["window_days"] * 3))) & (market["date"] <= (anchor_date + pd.Timedelta(days=event["window_days"] * 3)))].copy()
            window = window[(window["date"].dt.dayofweek < 5) & (window["usd_close"].notna() | window["bist_close"].notna())].copy()
            window = window.sort_values("date").reset_index(drop=True)
            if window.empty:
                continue
            if anchor_date not in set(window["date"]):
                continue
            anchor_index = int(window.index[window["date"] == anchor_date][0])
            window["relative_trade_day"] = window.index - anchor_index
            window = window[window["relative_trade_day"].between(-event["window_days"], event["window_days"])].copy()
            if window.empty:
                continue
            anchor_usd = float(window.loc[window["relative_trade_day"] == 0, "usd_close"].iloc[0]) if window["usd_close"].notna().any() else np.nan
            anchor_bist = float(window.loc[window["relative_trade_day"] == 0, "bist_close"].iloc[0]) if window["bist_close"].notna().any() else np.nan
            window["usd_cum_return_pct"] = (window["usd_close"] / anchor_usd - 1.0) * 100.0 if anchor_usd and not np.isnan(anchor_usd) else np.nan
            window["bist_cum_return_pct"] = (window["bist_close"] / anchor_bist - 1.0) * 100.0 if anchor_bist and not np.isnan(anchor_bist) else np.nan
            for _, row in window.iterrows():
                market_rows.append(
                    {
                        "event_id": event["event_id"],
                        "event_label": event["label"],
                        "requested_event_date": event["date"],
                        "anchor_market_date": anchor_date,
                        "date": row["date"],
                        "relative_trade_day": int(row["relative_trade_day"]),
                        "usd_cum_return_pct": row.get("usd_cum_return_pct"),
                        "bist_cum_return_pct": row.get("bist_cum_return_pct"),
                    }
                )

    return pd.DataFrame(event_rows), pd.DataFrame(profile_rows), pd.DataFrame(market_rows)


def run_robustness_checks(panel: pd.DataFrame, event_profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold_rows: list[dict[str, object]] = []
    intensity_col = "opposition_speeches_per_active_speaker"
    if intensity_col in panel.columns and "financial_stress_index" in panel.columns:
        for quantile in [0.70, 0.75, 0.80]:
            cutoff = panel["financial_stress_index"].quantile(quantile)
            crisis_dummy = panel["financial_stress_index"] >= cutoff
            crisis_values = panel.loc[crisis_dummy, intensity_col].dropna()
            non_crisis_values = panel.loc[~crisis_dummy, intensity_col].dropna()
            if len(crisis_values) >= 2 and len(non_crisis_values) >= 2:
                t_stat, p_value = stats.ttest_ind(crisis_values, non_crisis_values, equal_var=False, nan_policy="omit")
            else:
                t_stat, p_value = (np.nan, np.nan)
            threshold_rows.append(
                {
                    "quantile": quantile,
                    "cutoff": cutoff,
                    "crisis_n": len(crisis_values),
                    "non_crisis_n": len(non_crisis_values),
                    "crisis_mean": crisis_values.mean() if not crisis_values.empty else np.nan,
                    "non_crisis_mean": non_crisis_values.mean() if not non_crisis_values.empty else np.nan,
                    "welch_t_stat": t_stat,
                    "welch_p_value": p_value,
                }
            )

    lead_rows: list[dict[str, object]] = []
    predictor_outcome_pairs = [
        ("economy_negative_share", "financial_stress_index"),
        ("reform_positive_share", "bist_return"),
        ("gov_opp_cosine_distance_negative", "usd_return"),
    ]
    for predictor, outcome in predictor_outcome_pairs:
        if predictor not in panel.columns:
            continue
        for lead in [1, 2, 3]:
            lead_col = f"lead{lead}_{outcome}"
            if lead_col not in panel.columns:
                continue
            subset = panel[[predictor, lead_col]].dropna()
            if subset.shape[0] < 8:
                continue
            pearson_r, pearson_p = stats.pearsonr(subset[predictor], subset[lead_col])
            spearman_r, spearman_p = stats.spearmanr(subset[predictor], subset[lead_col])
            lead_rows.append(
                {
                    "predictor": predictor,
                    "outcome_lead": lead_col,
                    "lead": lead,
                    "n_obs": subset.shape[0],
                    "pearson_r": pearson_r,
                    "pearson_p_value": pearson_p,
                    "spearman_rho": spearman_r,
                    "spearman_p_value": spearman_p,
                }
            )

    leave_one_event_rows: list[dict[str, object]] = []
    event_windows = {}
    for event in MONTHLY_EVENTS:
        start = event["date"].to_period("M").to_timestamp() - pd.DateOffset(months=1)
        end = event["date"].to_period("M").to_timestamp() + pd.DateOffset(months=max(event["post_months"] - 1, 1))
        event_windows[event["event_id"]] = (start, end)

    for spec in [model for model in HYPOTHESIS_MODELS if model["hypothesis_id"] in {"H4a", "H4b"}]:
        for excluded_event, (start, end) in event_windows.items():
            subset = panel[(panel["date"] < start) | (panel["date"] > end)].copy()
            coeffs, _ = fit_newey_west_model(subset, spec["outcome"], spec["predictors"], spec["controls"], nw_lag=1, standardize=True)
            if coeffs.empty:
                leave_one_event_rows.append(
                    {
                        "hypothesis_id": spec["hypothesis_id"],
                        "excluded_event": excluded_event,
                        "primary_predictor": spec["primary_predictor"],
                        "standardized_beta": np.nan,
                        "p_value": np.nan,
                    }
                )
                continue
            primary = coeffs[coeffs["variable"] == spec["primary_predictor"]].iloc[0]
            leave_one_event_rows.append(
                {
                    "hypothesis_id": spec["hypothesis_id"],
                    "excluded_event": excluded_event,
                    "primary_predictor": spec["primary_predictor"],
                    "standardized_beta": primary["beta"],
                    "p_value": primary["p_value"],
                }
            )

    return pd.DataFrame(threshold_rows), pd.DataFrame(lead_rows), pd.DataFrame(leave_one_event_rows)


def add_event_lines(ax: plt.Axes) -> None:
    color_map = {
        "tezkere_1mart": "#9c6644",
        "ab_muzakereleri": "#2a6f97",
        "367_e_muhtira": "#ae2012",
    }
    for event in MONTHLY_EVENTS:
        ax.axvline(event["date"], color=color_map.get(event["event_id"], "#64748b"), linestyle="--", linewidth=1.2, alpha=0.8)


def plot_stage3_dashboard(panel: pd.DataFrame, output_path: Path, term: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.patch.set_facecolor("#f7f5ef")
    axes = axes.flatten()
    palette = ["#c4493d", "#1f8f72"]

    for ax, (left, right, title) in zip(axes, DASHBOARD_PAIR_SPECS):
        if left not in panel.columns or right not in panel.columns:
            ax.axis("off")
            continue
        subset = panel[["date", left, right]].copy().sort_values("date")
        subset[left] = zscore(subset[left]).interpolate(limit_area="inside")
        subset[right] = zscore(subset[right]).interpolate(limit_area="inside")
        ax.plot(subset["date"], subset[left], color=palette[0], linewidth=2.2, label=display_label(left))
        ax.plot(subset["date"], subset[right], color=palette[1], linewidth=2.2, label=display_label(right))
        add_event_lines(ax)
        ax.axhline(0.0, color="#cbd5e1", linewidth=0.9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.text(0.0, 1.01, "X: calendar month | Y: standardized level", transform=ax.transAxes, fontsize=8.8, color="#475569", va="bottom")
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(f"Term {term} | Stage 3 Monthly Co-Movement Dashboard", fontsize=20, fontweight="bold", y=0.995)
    fig.supxlabel("Calendar time (month)", y=0.045)
    fig.supylabel("Standardized level", x=0.015)
    fig.text(
        0.5,
        0.012,
        "Note: This is not a slope chart; it shows whether two indicators rise or fall together over time.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout(rect=[0.02, 0.07, 1, 0.975])
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_pairwise_relationships(panel: pd.DataFrame, output_path: Path, term: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#f7f5ef")
    axes = axes.flatten()

    for ax, (left, right, title) in zip(axes, DASHBOARD_PAIR_SPECS):
        if left not in panel.columns or right not in panel.columns:
            ax.axis("off")
            continue
        subset = panel[[left, right]].dropna().copy()
        if subset.empty:
            ax.axis("off")
            continue
        sns.regplot(
            data=subset,
            x=left,
            y=right,
            ax=ax,
            scatter_kws={"s": 34, "alpha": 0.75, "color": "#33658a"},
            line_kws={"color": "#c4493d", "linewidth": 2.0},
        )
        rho, p_value = stats.spearmanr(subset[left], subset[right], nan_policy="omit")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(display_label(left))
        ax.set_ylabel(display_label(right))
        ax.text(0.02, 0.96, f"Spearman r={rho:.2f} | p={p_value:.3f}", transform=ax.transAxes, va="top", fontsize=8.8, color="#475569")
        ax.grid(alpha=0.15)

    fig.suptitle(f"Term {term} | Stage 3 Direct Relationship Plots", fontsize=20, fontweight="bold", y=0.995)
    fig.text(0.5, 0.015, "Each point represents one month. If Y rises as X rises, the relationship is positive; if it falls, the relationship is negative.", ha="center", fontsize=10, color="#475569")
    fig.tight_layout(rect=[0.02, 0.03, 1, 0.975])
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_topic_ownership(yearly_topic_ownership: pd.DataFrame, output_path: Path, term: str) -> None:
    if yearly_topic_ownership.empty:
        return

    focus_specs = [
        ("economy", "total", "Economy"),
        ("democracy", "positive", "Democracy"),
        ("european_union", "positive", "European Union"),
        ("constitution", "negative", "Constitution"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    fig.patch.set_facecolor("#f7f5ef")
    axes = axes.flatten()

    for ax, (concept_slug, layer, title) in zip(axes, focus_specs):
        subset = yearly_topic_ownership[
            (yearly_topic_ownership["concept_slug"] == concept_slug)
            & (yearly_topic_ownership["layer"] == layer)
            & (yearly_topic_ownership["party"].isin(TERM_KEY_PARTIES.values()))
        ].copy()
        if subset.empty:
            ax.axis("off")
            continue
        for party in TERM_KEY_PARTIES.values():
            party_subset = subset[subset["party"] == party].sort_values("year")
            if party_subset.empty:
                continue
            color = base.dominant_party_color(party)
            line_width = 3.0 if party in {TERM_KEY_PARTIES["akp"], TERM_KEY_PARTIES["chp"]} else 1.7
            alpha = 1.0 if party in {TERM_KEY_PARTIES["akp"], TERM_KEY_PARTIES["chp"]} else 0.55
            ax.plot(
                party_subset["year"],
                party_subset["party_share_in_concept"],
                marker="o",
                linewidth=line_width,
                alpha=alpha,
                color=color,
                label=base.display_party_label(party, short=True),
            )
        ax.set_title(f"{title} | {base.LAYER_STYLES[layer]['title']}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.15)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Yearly concept share")
        ax.set_xlabel("Year")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Term {term} | Stage 4 Yearly Concept Ownership", fontsize=19, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: Path, term: str) -> None:
    if corr_matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor("#f7f5ef")
    display_matrix = corr_matrix.copy()
    display_matrix.index = [wrapped_label(label, width=24) for label in display_matrix.index]
    display_matrix.columns = [wrapped_label(label, width=24) for label in display_matrix.columns]
    sns.heatmap(
        display_matrix.astype(float),
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title(f"Term {term} | Stage 5 Correlation Heatmap", fontsize=18, fontweight="bold")
    ax.set_xlabel("Compared indicators")
    ax.set_ylabel("Compared indicators")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_regression_coefficients(coefficients: pd.DataFrame, output_path: Path, term: str) -> None:
    if coefficients.empty:
        return

    filtered = coefficients[
        (~coefficients["variable"].isin({"intercept"}))
        & (~coefficients["variable"].str.endswith("_lag1"))
        & (~coefficients["variable"].isin({"financial_stress_index", "bist_return"}))
    ].copy()
    if filtered.empty:
        return

    plot_df = filtered.sort_values(["hypothesis_id", "variable"]).copy()
    plot_df["label"] = plot_df["hypothesis_id"] + " | " + plot_df["variable"].map(display_label)
    plot_df["label_wrapped"] = plot_df["label"].map(lambda value: fill(value, width=42))

    fig, ax = plt.subplots(figsize=(14, max(6, 0.65 * len(plot_df))))
    fig.patch.set_facecolor("#f7f5ef")
    colors = ["#1f8f72" if value <= 0.10 else "#9ca3af" for value in plot_df["p_value"]]
    x_min = min(-0.12, float(plot_df["standardized_beta"].min()) - 0.05)
    x_max = max(0.35, float(plot_df["standardized_beta"].max()) + 0.08)
    ax.axvline(0.0, color="#475569", linewidth=1.0)
    ax.barh(plot_df["label_wrapped"], plot_df["standardized_beta"], color=colors, alpha=0.9)
    for _, row in plot_df.iterrows():
        raw_x = float(row["standardized_beta"]) + (0.02 if float(row["standardized_beta"]) >= 0 else -0.02)
        text_x = min(raw_x, x_max - 0.012) if float(row["standardized_beta"]) >= 0 else max(raw_x, x_min + 0.012)
        ax.text(
            text_x,
            row["label_wrapped"],
            f"p={row['p_value']:.3f}",
            va="center",
            ha="left" if float(row["standardized_beta"]) >= 0 else "right",
            fontsize=8,
        )
    ax.set_title(f"Term {term} | Stage 5 Standardized Coefficients", fontsize=18, fontweight="bold")
    ax.set_xlabel("Standardized coefficient (Newey-West)")
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_granger_heatmap(granger_results: pd.DataFrame, output_path: Path, term: str) -> None:
    if granger_results.empty:
        return
    plot_df = granger_results.copy()
    plot_df["pair"] = plot_df["cause"].map(display_label) + " -> " + plot_df["effect"].map(display_label)
    pivot = plot_df.pivot(index="pair", columns="lag", values="p_value")
    transformed = -np.log10(pivot.astype(float))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(transformed))))
    fig.patch.set_facecolor("#f7f5ef")
    transformed.index = [fill(label, width=34) for label in transformed.index]
    pivot.index = transformed.index
    sns.heatmap(transformed, annot=pivot.round(3), fmt="", cmap="YlOrRd", linewidths=0.5, ax=ax, cbar_kws={"label": "Significance strength (-log10 p)"})
    ax.set_title(f"Term {term} | Stage 5 Granger Causality Heatmap", fontsize=17, fontweight="bold")
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_event_profiles(event_profiles: pd.DataFrame, output_path: Path, term: str) -> None:
    if event_profiles.empty:
        return

    focus = {
        "tezkere_1mart": ["gov_opp_cosine_distance_total", "economy_negative_share", "usd_return"],
        "ab_muzakereleri": ["reform_positive_share", "gov_opp_cosine_distance_total", "bist_return"],
        "367_e_muhtira": ["crisis_negative_share", "gov_opp_cosine_distance_negative", "financial_stress_index"],
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.patch.set_facecolor("#f7f5ef")

    for ax, event in zip(axes, MONTHLY_EVENTS):
        subset = event_profiles[event_profiles["event_id"] == event["event_id"]].copy()
        if subset.empty:
            ax.axis("off")
            continue
        for variable in focus[event["event_id"]]:
            if variable not in subset.columns:
                continue
            standardized = zscore(subset[variable]).interpolate(limit_area="inside")
            ax.plot(subset["relative_month"], standardized, marker="o", linewidth=2.2, label=display_label(variable))
        ax.axvline(0, color="#475569", linestyle="--", linewidth=1.0)
        ax.set_title(event["label"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Month relative to event")
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8, loc="best")

    axes[0].set_ylabel("Standardized level")
    fig.suptitle(f"Term {term} | Stage 6 Event-Window Profiles", fontsize=18, fontweight="bold", y=1.02)
    fig.text(0.5, 0.01, "The zero point marks the event month; lines make pre- and post-event direction changes easier to see.", ha="center", fontsize=10, color="#475569")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_daily_market_reactions(market_event_profiles: pd.DataFrame, output_path: Path, term: str) -> None:
    if market_event_profiles.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=False)
    fig.patch.set_facecolor("#f7f5ef")
    axes = axes.flatten()
    for ax, event in zip(axes, MARKET_EVENTS):
        subset = market_event_profiles[market_event_profiles["event_id"] == event["event_id"]].copy()
        if subset.empty:
            ax.axis("off")
            continue
        subset = subset.sort_values("relative_trade_day")
        subset["bist_plot"] = subset["bist_cum_return_pct"].interpolate(limit_direction="both")
        subset["usd_plot"] = subset["usd_cum_return_pct"].interpolate(limit_direction="both")
        ax.plot(subset["relative_trade_day"], subset["bist_plot"], color="#1f8f72", linewidth=2.5, label="BIST 100 cumulative change (%)")
        ax.plot(subset["relative_trade_day"], subset["usd_plot"], color="#c4493d", linewidth=2.5, label="USD/TRY cumulative change (%)")
        ax.scatter(subset["relative_trade_day"], subset["bist_cum_return_pct"], color="#1f8f72", s=14, alpha=0.7)
        ax.scatter(subset["relative_trade_day"], subset["usd_cum_return_pct"], color="#c4493d", s=14, alpha=0.7)
        ax.axvline(0, color="#475569", linestyle="--", linewidth=1.0)
        ax.axhline(0, color="#cbd5e1", linewidth=0.8)
        ax.set_title(event["label"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Trading day relative to event")
        ax.set_ylabel("Cumulative change (%)")
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"Term {term} | Stage 6 Daily Market Reactions", fontsize=18, fontweight="bold", y=0.99)
    fig.text(0.5, 0.015, "The zero point marks the first trading day after the event. Lines are traced over trading days.", ha="center", fontsize=10, color="#475569")
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_robustness_summary(lead_sensitivity: pd.DataFrame, leave_one_event_out: pd.DataFrame, output_path: Path, term: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#f7f5ef")

    if not lead_sensitivity.empty:
        pivot = lead_sensitivity.pivot(index="predictor", columns="lead", values="spearman_rho")
        pivot.index = [wrapped_label(label, width=24) for label in pivot.index]
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0.0, ax=axes[0], linewidths=0.5, cbar=False)
        axes[0].set_title("Lead sensitivity", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Months ahead")
        axes[0].set_ylabel("")
    else:
        axes[0].axis("off")

    if not leave_one_event_out.empty:
        plot_df = leave_one_event_out.copy()
        plot_df["label"] = plot_df["hypothesis_id"] + " | " + plot_df["excluded_event"].map(display_label)
        plot_df["label"] = plot_df["label"].map(lambda value: fill(value, width=34))
        colors = ["#1f8f72" if value <= 0.10 else "#9ca3af" for value in plot_df["p_value"]]
        axes[1].axvline(0.0, color="#475569", linewidth=1.0)
        axes[1].barh(plot_df["label"], plot_df["standardized_beta"], color=colors, alpha=0.9)
        axes[1].set_title("Leave-one-event-out coefficients", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Standardized coefficient")
        axes[1].grid(axis="x", alpha=0.15)
    else:
        axes[1].axis("off")

    fig.suptitle(f"Term {term} | Stage 7 Robustness Checks", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_output_index(term: str) -> list[str]:
    return [
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_monthly_network_macro_panel.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_monthly_layer_edges.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_monthly_party_metrics.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_monthly_concept_metrics.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_monthly_network_summary.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_monthly_topic_ownership.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_yearly_topic_ownership.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_yearly_analysis_panel.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_hypothesis_registry.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_model_coefficients.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_model_summary.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_correlation_matrix.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_correlation_tests.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_granger_results.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_event_tests.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_event_profiles.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_market_event_profiles.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_crisis_threshold_sensitivity.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_lead_sensitivity.csv",
        f"CSVs/term_{term}_{STAGE3_STAGE7_PREFIX}_leave_one_event_out.csv",
        f"Figures/term_{term}_{STAGE3_STAGE7_PREFIX}_monthly_dashboard.png",
        f"Figures/term_{term}_{STAGE3_STAGE7_PREFIX}_pairwise_relationships.png",
        f"Figures/term_{term}_{STAGE3_STAGE7_PREFIX}_topic_ownership.png",
        f"Figures/term_{term}_{STAGE3_STAGE7_PREFIX}_correlation_heatmap.png",
        f"Figures/term_{term}_{STAGE3_STAGE7_PREFIX}_regression_coefficients.png",
        f"Figures/term_{term}_{STAGE3_STAGE7_PREFIX}_granger_heatmap.png",
        f"Figures/term_{term}_{STAGE3_STAGE7_PREFIX}_event_profiles.png",
        f"Figures/term_{term}_{STAGE3_STAGE7_PREFIX}_daily_market_reactions.png",
        f"Figures/term_{term}_{STAGE3_STAGE7_PREFIX}_robustness_summary.png",
        f"Notes/term_{term}_{STAGE3_STAGE7_PREFIX}_summary.txt",
        f"Notes/term_{term}_{STAGE3_STAGE7_PREFIX}_output_index.txt",
    ]


def write_summary_note(
    term: str,
    panel: pd.DataFrame,
    monthly_edges: pd.DataFrame,
    model_summary: pd.DataFrame,
    event_tests: pd.DataFrame,
    granger_results: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    notes_dir: Path,
) -> None:
    lines = [
        f"Term {term} | Asama 3-7 ozet",
        "",
        f"Aylik panel gozlem sayisi: {len(panel)}",
        f"Konusma olan ay sayisi: {int(panel['speech_count'].fillna(0).gt(0).sum()) if 'speech_count' in panel.columns else 0}",
        f"Aylik katmanli kenar satiri: {len(monthly_edges)}",
        "",
        "Asama 3:",
        "- Aylik network-makro panel kuruldu.",
        "- Konusma, kavram, parti ve kavram merkeziligi ayni zaman eksenine tasindi.",
        "",
        "Asama 4:",
        "- H1-H4 hipotezleri icin olcumsellesmis degiskenler olusturuldu.",
        "- Topic ownership serileri ve yillik kavram sahipligi cikarildi.",
        "",
        "Asama 5:",
    ]
    if not model_summary.empty:
        for row in model_summary.sort_values("hypothesis_id").itertuples(index=False):
            lines.append(
                f"- {row.hypothesis_id}: beta={row.primary_beta:.4f} | p={row.primary_p_value:.4f} | supported_10pct={bool(row.supported_at_10pct)}"
            )
    else:
        lines.append("- Hipotez modelleri calismadi veya yeterli gozlem yok.")

    lines.extend(["", "Asama 6:"])
    if not event_tests.empty:
        significant_events = event_tests[event_tests["welch_p_value"] <= 0.10].sort_values(["event_id", "welch_p_value"]).head(8)
        if significant_events.empty:
            lines.append("- Olay pencerelerinde 10% seviyesinde anlamli fark bulunmadi.")
        else:
            for row in significant_events.itertuples(index=False):
                lines.append(
                    f"- {row.event_label} | {display_label(row.variable)}: diff={row.mean_diff_post_minus_pre:.4f} | p={row.welch_p_value:.4f} | g={row.hedges_g:.4f}"
                )
    else:
        lines.append("- Olay penceresi testi uretilemedi.")

    lines.extend(["", "Asama 7:"])
    if not granger_results.empty:
        sig_granger = granger_results[granger_results["p_value"] <= 0.10].sort_values("p_value").head(6)
        if sig_granger.empty:
            lines.append("- Granger testlerinde 10% seviyesinde anlamli sonuc yok.")
        else:
            for row in sig_granger.itertuples(index=False):
                lines.append(f"- {display_label(row.cause)} -> {display_label(row.effect)} | lag={int(row.lag)} | p={row.p_value:.4f}")
    if not threshold_sensitivity.empty:
        for row in threshold_sensitivity.itertuples(index=False):
            diff_value = (row.crisis_mean - row.non_crisis_mean) if not pd.isna(row.crisis_mean) and not pd.isna(row.non_crisis_mean) else np.nan
            lines.append(
                f"- Kriz esigi q={row.quantile:.2f}: diff={diff_value:.4f} | p={row.welch_p_value:.4f}"
            )

    lines.extend(["", "Uretilen dosyalar:"])
    for path in build_output_index(term):
        lines.append(f"- {path}")

    (notes_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    (notes_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_output_index.txt").write_text("\n".join(build_output_index(term)), encoding="utf-8")


def write_outputs(
    term: str,
    monthly_panel: pd.DataFrame,
    monthly_edges: pd.DataFrame,
    monthly_party_metrics: pd.DataFrame,
    monthly_concept_metrics: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    monthly_topic_ownership: pd.DataFrame,
    yearly_topic_ownership: pd.DataFrame,
    yearly_analysis_panel: pd.DataFrame,
    hypothesis_registry: pd.DataFrame,
    model_coefficients: pd.DataFrame,
    model_summary: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    corr_tests: pd.DataFrame,
    granger_results: pd.DataFrame,
    event_tests: pd.DataFrame,
    event_profiles: pd.DataFrame,
    market_event_profiles: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    lead_sensitivity: pd.DataFrame,
    leave_one_event_out: pd.DataFrame,
) -> None:
    csvs_dir, figures_dir, notes_dir = ensure_output_dirs(term)

    monthly_panel.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_monthly_network_macro_panel.csv", index=False, encoding="utf-8-sig")
    monthly_edges.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_monthly_layer_edges.csv", index=False, encoding="utf-8-sig")
    monthly_party_metrics.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_monthly_party_metrics.csv", index=False, encoding="utf-8-sig")
    monthly_concept_metrics.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_monthly_concept_metrics.csv", index=False, encoding="utf-8-sig")
    monthly_summary.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_monthly_network_summary.csv", index=False, encoding="utf-8-sig")
    monthly_topic_ownership.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_monthly_topic_ownership.csv", index=False, encoding="utf-8-sig")
    yearly_topic_ownership.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_yearly_topic_ownership.csv", index=False, encoding="utf-8-sig")
    yearly_analysis_panel.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_yearly_analysis_panel.csv", index=False, encoding="utf-8-sig")
    hypothesis_registry.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_hypothesis_registry.csv", index=False, encoding="utf-8-sig")
    model_coefficients.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_model_coefficients.csv", index=False, encoding="utf-8-sig")
    model_summary.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_model_summary.csv", index=False, encoding="utf-8-sig")
    corr_matrix.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_correlation_matrix.csv", encoding="utf-8-sig")
    corr_tests.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_correlation_tests.csv", index=False, encoding="utf-8-sig")
    granger_results.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_granger_results.csv", index=False, encoding="utf-8-sig")
    event_tests.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_event_tests.csv", index=False, encoding="utf-8-sig")
    event_profiles.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_event_profiles.csv", index=False, encoding="utf-8-sig")
    market_event_profiles.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_market_event_profiles.csv", index=False, encoding="utf-8-sig")
    threshold_sensitivity.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_crisis_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    lead_sensitivity.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_lead_sensitivity.csv", index=False, encoding="utf-8-sig")
    leave_one_event_out.to_csv(csvs_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_leave_one_event_out.csv", index=False, encoding="utf-8-sig")

    plot_stage3_dashboard(monthly_panel, figures_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_monthly_dashboard.png", term)
    plot_pairwise_relationships(monthly_panel, figures_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_pairwise_relationships.png", term)
    plot_topic_ownership(yearly_topic_ownership, figures_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_topic_ownership.png", term)
    plot_correlation_heatmap(corr_matrix, figures_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_correlation_heatmap.png", term)
    plot_regression_coefficients(model_coefficients, figures_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_regression_coefficients.png", term)
    plot_granger_heatmap(granger_results, figures_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_granger_heatmap.png", term)
    plot_event_profiles(event_profiles, figures_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_event_profiles.png", term)
    plot_daily_market_reactions(market_event_profiles, figures_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_daily_market_reactions.png", term)
    plot_robustness_summary(lead_sensitivity, leave_one_event_out, figures_dir / f"term_{term}_{STAGE3_STAGE7_PREFIX}_robustness_summary.png", term)

    write_summary_note(term, monthly_panel, monthly_edges, model_summary, event_tests, granger_results, threshold_sensitivity, notes_dir)


def build_term_analysis_pack(term: str) -> None:
    inputs = load_inputs(term)
    monthly_speech = build_monthly_speech_panel(inputs.speech, inputs.month_index)
    monthly_edges, monthly_party_metrics, monthly_concept_metrics, monthly_summary = build_monthly_network_outputs(term, inputs.edges, inputs.month_index)
    monthly_panel, monthly_topic_ownership, yearly_topic_ownership, yearly_analysis_panel = build_analysis_panel(
        term,
        inputs,
        monthly_speech,
        monthly_edges,
        monthly_party_metrics,
        monthly_concept_metrics,
        monthly_summary,
    )

    hypothesis_registry = build_hypothesis_registry()
    model_coefficients, model_summary = run_hypothesis_models(monthly_panel)
    corr_matrix, corr_tests = compute_correlation_tables(monthly_panel)
    granger_results = run_granger_suite(monthly_panel)
    event_tests, event_profiles, market_event_profiles = run_event_studies(monthly_panel, inputs.market_daily)
    threshold_sensitivity, lead_sensitivity, leave_one_event_out = run_robustness_checks(monthly_panel, event_profiles)

    write_outputs(
        term=term,
        monthly_panel=monthly_panel,
        monthly_edges=monthly_edges,
        monthly_party_metrics=monthly_party_metrics,
        monthly_concept_metrics=monthly_concept_metrics,
        monthly_summary=monthly_summary,
        monthly_topic_ownership=monthly_topic_ownership,
        yearly_topic_ownership=yearly_topic_ownership,
        yearly_analysis_panel=yearly_analysis_panel,
        hypothesis_registry=hypothesis_registry,
        model_coefficients=model_coefficients,
        model_summary=model_summary,
        corr_matrix=corr_matrix,
        corr_tests=corr_tests,
        granger_results=granger_results,
        event_tests=event_tests,
        event_profiles=event_profiles,
        market_event_profiles=market_event_profiles,
        threshold_sensitivity=threshold_sensitivity,
        lead_sensitivity=lead_sensitivity,
        leave_one_event_out=leave_one_event_out,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 3-7 research outputs for a TBMM term.")
    parser.add_argument("--term", default="22", help="TBMM term number to analyze.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_term_analysis_pack(str(args.term))


if __name__ == "__main__":
    main()
