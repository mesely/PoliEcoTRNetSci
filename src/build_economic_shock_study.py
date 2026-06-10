"""
build_economic_shock_study.py
────────────────────────────────────────────────────────────────────────────
Turn the TBMM signed-network project into an economy-centered event study.

Outputs
  Inter_Term/CSVs/inter_term_global_economic_positions.csv
  Inter_Term/CSVs/inter_term_economic_event_summary.csv
  Inter_Term/CSVs/inter_term_economic_event_party_metrics.csv
  Inter_Term/Figures/inter_term_economic_interval_axis.png
  Inter_Term/Figures/inter_term_economic_shock_comparison.png
  Inter_Term/Figures/inter_term_economic_pressure_alignment.png
  Inter_Term/Notes/inter_term_economic_summary.txt

  Term_XX/Figures/term_XX_economic_triptych_<event>.png
  Term_XX/Figures/term_XX_economic_overlay_<event>.png
  Term_XX/CSVs/term_XX_economic_event_summary.csv
  Term_XX/CSVs/term_XX_economic_event_party_metrics.csv
  Term_XX/CSVs/term_XX_economic_event_gatekeepers.csv
  Term_XX/Notes/term_XX_economic_brief.txt
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from build_term_interval_polarization import (
    BASE,
    IDEOLOGY_REF,
    PARTY_COLORS,
    PARTY_SHORT,
    TERMS,
    build_sparse_signed_graph,
    compute_gatekeeper_scores,
    signed_agenda_feature,
)


ECONOMIC_TOPICS = {
    "economy": "Economy",
    "inflation": "Inflation",
    "interest": "Interest",
    "imf": "IMF / external finance",
}
FOCUS_TOPIC_CATEGORIES = {
    "macro_economy": "Macroeconomy",
    "constitutional_conflict": "Constitutional conflict",
    "democratic_reform": "Democratic reform",
    "security_conflict": "Security conflict",
}
RNG = np.random.default_rng(42)
SIGNED_LABEL_BACKBONE = {
    "primary_model": "hybrid_beme_mixed_gate_logistic",
    "score_field": "beme_score",
    "label_field": "beme_label",
    "concept_edge_pattern": "term_{term}_beme_concept_edges.csv",
    "speech_label_pattern": "term_{term}_beme_speech_labels.csv",
}

EVENT_CATALOG: dict[int, list[dict[str, str]]] = {
    22: [
        {
            "target_month": "2006-05",
            "label": "2006 market turbulence",
            "kind": "economic",
            "paper_role": "support",
            "rationale": "Mayıs 2006 finansal türbülansı, veri içinde açık biçimde görülen ilk makro-finansal kırılma olduğu için cumhurbaşkanlığı krizinden önce erken bir ekonomik karşılaştırma noktası sağlıyor.",
        },
        {
            "target_month": "2007-04",
            "label": "E-memorandum",
            "kind": "control",
            "paper_role": "control",
            "rationale": "E-muhtıra ekonomik olmayan anayasal bir şoktur; bu yüzden ekonomiden anayasa/güvenlik eksenine kaymayı test eden güçlü bir kontrol vakası olarak kullanılıyor.",
        },
    ],
    23: [
        {
            "target_month": "2008-10",
            "label": "Global crisis reaction",
            "kind": "economic",
            "paper_role": "core",
            "rationale": "2008 küresel kriz tepkisi, erken çok partili dönemin en temiz makroekonomik şokudur ve gündem sahipliğinin yeniden kurulmasını test etmek için çekirdek vakadır.",
        },
        {
            "target_month": "2010-10",
            "label": "Referendum aftermath",
            "kind": "control",
            "paper_role": "support",
            "rationale": "Referandum sonrası dönem, aynı term içinde ekonomi-merkezli ve anayasa-merkezli gündem merkezileşmesini karşılaştırmaya yardım ediyor.",
        },
    ],
    24: [
        {
            "target_month": "2013-12",
            "label": "17-25 December rupture",
            "kind": "mixed",
            "paper_role": "support",
            "rationale": "Bu kırılma yolsuzluk, kurumsal gerilim ve ekonomik dili aynı anda taşıdığı için saf ekonomi vakası değil; karma şok kıyas noktası olarak değerli.",
        },
    ],
    25: [
        {
            "target_month": "2015-10",
            "label": "Pre-November election macro reset",
            "kind": "economic",
            "paper_role": "support",
            "rationale": "Kısa interregnum terminde zaman varyasyonu az olduğu için Kasım seçimi öncesindeki makro yeniden ayar, uygulanabilir tek ekonomik olay penceresidir.",
        },
    ],
    26: [
        {
            "target_month": "2016-07",
            "label": "15 July / OHAL shock",
            "kind": "control",
            "paper_role": "control",
            "rationale": "15 Temmuz/OHAL hattı güvenlik-anayasa eksenli bir mega şoktur; bu yüzden ekonomi dışı yer değiştirme testleri için güçlü bir kontrol vakasıdır.",
        },
    ],
    27: [
        {
            "target_month": "2020-11",
            "label": "Albayrak resignation",
            "kind": "economic_governance",
            "paper_role": "core",
            "rationale": "Albayrak istifası ekonomik yönetişimde görünür bir kırılma yarattığı için hem agenda ownership hem de broker turnover testleri açısından merkezi bir vakadır.",
        },
        {
            "target_month": "2022-01",
            "label": "FX crisis",
            "kind": "economic",
            "paper_role": "core",
            "rationale": "Kur krizi geç dönemin en güçlü makroekonomik şokudur ve ekonomik çatışma geometrisinin ana stres testini oluşturur.",
        },
    ],
    28: [
        {
            "target_month": "2023-06",
            "label": "Simsek return",
            "kind": "economic_governance",
            "paper_role": "core",
            "rationale": "Şimşek dönüşü, term başlangıcına yakın olsa da teknokratik ekonomik dilin geri dönüşünü simgelediği için çekirdek bir ekonomik yönetişim olayıdır.",
        },
        {
            "target_month": "2024-03",
            "label": "Local election shock",
            "kind": "mixed",
            "paper_role": "support",
            "rationale": "Yerel seçim şoku ekonomik hesap verme baskısını rejim rekabetiyle birleştirdiği için karşılaştırmalı olarak yararlı bir karma şok vakasıdır.",
        },
    ],
}

MAJOR_PARTIES = [
    "Adalet ve Kalkınma Partisi",
    "Cumhuriyet Halk Partisi",
    "Milliyetçi Hareket Partisi",
    "Halkların Demokratik Partisi",
    "Halkların Eşitlik ve Demokrasi Partisi",
    "DEM Parti",
    "İYİ Parti",
    "Saadet Partisi",
]


def slugify(value: str) -> str:
    return (
        value.lower()
        .replace("ı", "i")
        .replace("İ", "i")
        .replace("ş", "s")
        .replace("Ş", "s")
        .replace("ğ", "g")
        .replace("Ğ", "g")
        .replace("ü", "u")
        .replace("Ü", "u")
        .replace("ö", "o")
        .replace("Ö", "o")
        .replace("ç", "c")
        .replace("Ç", "c")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def zscore_frame(frame: pd.DataFrame) -> pd.DataFrame:
    std = frame.std(axis=0).replace(0, 1.0)
    return ((frame - frame.mean(axis=0)) / std).fillna(0.0)


def choose_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def read_term_edges(term: int) -> pd.DataFrame:
    path = BASE / f"Term_{term}" / "CSVs" / f"term_{term}_beme_concept_edges.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["month_key"] = df["date"].dt.to_period("M").astype(str)
    return df


def available_months(df: pd.DataFrame) -> list[str]:
    return sorted(df["month_key"].dropna().unique().tolist())


def resolve_event_month(target_month: str, month_keys: list[str]) -> str:
    if target_month in month_keys:
        return target_month
    target = pd.Period(target_month, freq="M")
    periods = [pd.Period(m, freq="M") for m in month_keys]
    future = [p for p in periods if p >= target]
    if future:
        return str(min(future))
    return str(max(periods))


def stage_months(event_month: str, pre: int = 3, post: int = 3) -> dict[str, list[str]]:
    event = pd.Period(event_month, freq="M")
    return {
        "pre": [str(event - i) for i in range(pre, 0, -1)],
        "shock": [str(event)],
        "post": [str(event + i) for i in range(1, post + 1)],
    }


def eligible_event_months(month_keys: list[str], pre: int = 3, post: int = 3) -> list[str]:
    periods = sorted(pd.Period(m, freq="M") for m in month_keys)
    available = set(periods)
    eligible: list[str] = []
    for period in periods:
        if all((period - i) in available for i in range(1, pre + 1)) and all((period + i) in available for i in range(1, post + 1)):
            eligible.append(str(period))
    return eligible


def load_monthly_macro() -> pd.DataFrame:
    df = pd.read_csv(BASE / "economy_data" / "preprocessed" / "economy_monthly_macro.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["month_key"] = df["date"].dt.to_period("M").astype(str)

    usd_col = choose_col(df, ["usd_try_return_month_pct", "usd_try_return_month_pct_avg"])
    infl_col = choose_col(df, ["inflation_yoy_percent_harmonized", "cpi_yoy_percent_tuik", "cpi_inflation_yoy_percent_fred"])
    bist_col = choose_col(df, ["bist100_return_month_pct", "bist100_return_month_pct_avg"])
    policy_col = choose_col(df, ["policy_rate_percent_tcmb", "short_term_interest_rate_percent"])

    base = pd.DataFrame({"month_key": df["month_key"]})
    base["fx_return"] = df[usd_col] if usd_col else np.nan
    base["inflation_yoy"] = df[infl_col] if infl_col else np.nan
    base["equity_drop"] = -(df[bist_col] if bist_col else 0.0)
    base["policy_rate"] = df[policy_col] if policy_col else np.nan
    base["policy_rate_change"] = base["policy_rate"].diff()

    cols = ["fx_return", "inflation_yoy", "equity_drop", "policy_rate_change"]
    z = base[cols].copy()
    for col in cols:
        std = z[col].std()
        if pd.isna(std) or std == 0:
            z[col] = 0.0
        else:
            z[col] = (z[col] - z[col].mean()) / std
    base["macro_pressure_index"] = z.mean(axis=1).fillna(0.0)
    return base


def build_global_axis() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    yearly_frames = []
    monthly_frames = []
    for term in TERMS:
        yearly_path = BASE / f"Term_{term}" / "CSVs" / f"term_{term}_yearly_layer_party_concept_edges.csv"
        yearly = pd.read_csv(yearly_path)
        yearly = yearly[(yearly["layer"] == "total") & (yearly["concept_slug"].isin(ECONOMIC_TOPICS.keys()))].copy()
        yearly["term"] = term
        yearly_feat = signed_agenda_feature(yearly, ["term", "year", "party", "concept_slug"])
        yearly_mat = yearly_feat.pivot_table(
            index=["term", "year", "party"],
            columns="concept_slug",
            values="feature",
            fill_value=0.0,
        )
        yearly_frames.append(yearly_mat)

        monthly = read_term_edges(term)
        monthly = monthly[monthly["concept_slug"].isin(ECONOMIC_TOPICS.keys())].copy()
        monthly["term"] = term
        monthly_feat = signed_agenda_feature(monthly, ["term", "month_key", "party", "concept_slug"])
        monthly_mat = monthly_feat.pivot_table(
            index=["term", "month_key", "party"],
            columns="concept_slug",
            values="feature",
            fill_value=0.0,
        )
        monthly_frames.append(monthly_mat)

    yearly_all = pd.concat(yearly_frames, axis=0).sort_index()
    monthly_all = pd.concat(monthly_frames, axis=0).sort_index()

    raw_mean = yearly_all.mean(axis=0)
    raw_std = yearly_all.std(axis=0).replace(0, 1.0)
    z_yearly = ((yearly_all - raw_mean) / raw_std).fillna(0.0)

    by_party = z_yearly.groupby(level="party").mean()
    anchor_parties = [p for p in by_party.index if p in IDEOLOGY_REF]
    ideology = pd.Series({p: IDEOLOGY_REF[p] for p in anchor_parties})
    weights: dict[str, float] = {}
    for col in by_party.columns:
        x = by_party.loc[anchor_parties, col].to_numpy(dtype=float)
        y = ideology.to_numpy(dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            weights[col] = 0.0
        else:
            weights[col] = float(np.corrcoef(x, y)[0, 1])
    weights = pd.Series(weights).fillna(0.0)
    if np.linalg.norm(weights.to_numpy()) == 0:
        _, _, vt = np.linalg.svd(by_party.to_numpy(dtype=float), full_matrices=False)
        weights = pd.Series(vt[0], index=by_party.columns)
    weights = weights / max(np.linalg.norm(weights.to_numpy()), 1e-9)

    yearly_pos = z_yearly.dot(weights).rename("position").reset_index()
    anchor_positions = yearly_pos[yearly_pos["party"].isin(anchor_parties)].groupby("party")["position"].mean()
    corr = np.corrcoef(anchor_positions.loc[ideology.index], ideology)[0, 1]
    if not np.isnan(corr) and corr < 0:
        weights = -weights
        yearly_pos["position"] = -yearly_pos["position"]

    z_monthly = ((monthly_all - raw_mean.reindex(monthly_all.columns, fill_value=0.0)) / raw_std.reindex(monthly_all.columns, fill_value=1.0)).fillna(0.0)
    monthly_pos = z_monthly.dot(weights).rename("position").reset_index()

    weights_df = weights.rename("weight").reset_index().rename(columns={"index": "concept_slug"})
    return yearly_pos, monthly_pos, weights_df, raw_mean, raw_std


def build_axis_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    yearly_frames = []
    for term in TERMS:
        yearly_path = BASE / f"Term_{term}" / "CSVs" / f"term_{term}_yearly_layer_party_concept_edges.csv"
        yearly = pd.read_csv(yearly_path)
        yearly = yearly[(yearly["layer"] == "total") & (yearly["concept_slug"].isin(ECONOMIC_TOPICS.keys()))].copy()
        yearly["term"] = term
        yearly_feat = signed_agenda_feature(yearly, ["term", "year", "party", "concept_slug"])
        yearly_mat = yearly_feat.pivot_table(
            index=["term", "year", "party"],
            columns="concept_slug",
            values="feature",
            fill_value=0.0,
        )
        yearly_frames.append(yearly_mat)

    yearly_all = pd.concat(yearly_frames, axis=0).sort_index()
    raw_mean = yearly_all.mean(axis=0)
    raw_std = yearly_all.std(axis=0).replace(0, 1.0)
    z_yearly = ((yearly_all - raw_mean) / raw_std).fillna(0.0)
    by_party = z_yearly.groupby(level="party").mean()
    anchor_parties = [p for p in by_party.index if p in IDEOLOGY_REF]
    ideology = pd.Series({p: IDEOLOGY_REF[p] for p in anchor_parties})

    def fit_weights(frame: pd.DataFrame, anchors: list[str]) -> pd.Series:
        if not anchors:
            return pd.Series(np.ones(frame.shape[1]), index=frame.columns) / max(math.sqrt(frame.shape[1]), 1.0)
        y = pd.Series({p: IDEOLOGY_REF[p] for p in anchors})
        weights_local: dict[str, float] = {}
        for col in frame.columns:
            x = frame.loc[anchors, col].to_numpy(dtype=float)
            vals = y.to_numpy(dtype=float)
            if np.std(x) == 0 or np.std(vals) == 0:
                weights_local[col] = 0.0
            else:
                weights_local[col] = float(np.corrcoef(x, vals)[0, 1])
        weights_series = pd.Series(weights_local).fillna(0.0)
        if np.linalg.norm(weights_series.to_numpy()) == 0:
            _, _, vt = np.linalg.svd(frame.loc[anchors].to_numpy(dtype=float), full_matrices=False)
            weights_series = pd.Series(vt[0], index=frame.columns)
        return weights_series / max(np.linalg.norm(weights_series.to_numpy()), 1e-9)

    weights = fit_weights(by_party, anchor_parties)
    anchor_positions = by_party.loc[anchor_parties].dot(weights)
    obs_corr = float(np.corrcoef(anchor_positions.loc[ideology.index], ideology)[0, 1]) if len(ideology) >= 2 else np.nan
    if pd.notna(obs_corr) and obs_corr < 0:
        weights = -weights
        anchor_positions = -anchor_positions
        obs_corr = -obs_corr

    perm_corrs = []
    ideology_values = ideology.to_numpy(dtype=float)
    for _ in range(500):
        shuffled = RNG.permutation(ideology_values)
        perm_corrs.append(float(np.corrcoef(anchor_positions.loc[ideology.index], shuffled)[0, 1]))
    perm_corrs_arr = np.array(perm_corrs, dtype=float)
    perm_p = float((np.abs(perm_corrs_arr) >= abs(obs_corr)).mean()) if pd.notna(obs_corr) else np.nan

    bootstrap_rows = []
    party_year = z_yearly.reset_index()
    sample_keys = party_year[["term", "year", "party"]]
    feature_cols = [c for c in party_year.columns if c not in {"term", "year", "party"}]
    boot_weight_samples = []
    for boot in range(250):
        take = RNG.integers(0, len(sample_keys), len(sample_keys))
        sampled = party_year.iloc[take].copy()
        sampled_by_party = sampled.groupby("party")[feature_cols].mean()
        anchors = [p for p in anchor_parties if p in sampled_by_party.index]
        boot_weights = fit_weights(sampled_by_party, anchors)
        boot_anchor_positions = sampled_by_party.loc[anchors].dot(boot_weights) if anchors else pd.Series(dtype=float)
        boot_corr = np.nan
        if len(anchors) >= 2:
            y = pd.Series({p: IDEOLOGY_REF[p] for p in anchors})
            boot_corr = float(np.corrcoef(boot_anchor_positions.loc[y.index], y)[0, 1])
        if pd.notna(boot_corr) and boot_corr < 0:
            boot_weights = -boot_weights
            boot_corr = -boot_corr
        boot_weight_samples.append(boot_weights)
        bootstrap_rows.append(
            {
                "metric": "anchor_corr_bootstrap",
                "sample_id": boot,
                "value": boot_corr,
            }
        )

    loo_rows = []
    full_party_positions = by_party.dot(weights)
    for left_out in anchor_parties:
        anchors = [p for p in anchor_parties if p != left_out]
        loo_weights = fit_weights(by_party, anchors)
        loo_positions = by_party.dot(loo_weights)
        corr = float(np.corrcoef(full_party_positions.reindex(by_party.index), loo_positions.reindex(by_party.index))[0, 1])
        if pd.notna(corr) and corr < 0:
            loo_weights = -loo_weights
            loo_positions = -loo_positions
            corr = -corr
        loo_rows.append(
            {
                "metric": "leave_one_anchor_out",
                "anchor": left_out,
                "correlation_with_full_axis": corr,
            }
        )

    weight_boot = pd.concat(boot_weight_samples, axis=1)
    weight_summary = pd.DataFrame(
        {
            "concept_slug": weights.index,
            "weight": weights.values,
            "bootstrap_mean": weight_boot.mean(axis=1).values,
            "bootstrap_std": weight_boot.std(axis=1).values,
            "bootstrap_q10": weight_boot.quantile(0.10, axis=1).values,
            "bootstrap_q90": weight_boot.quantile(0.90, axis=1).values,
        }
    )

    validation_rows = [
        {"metric": "observed_anchor_correlation", "value": obs_corr},
        {"metric": "anchor_permutation_p", "value": perm_p},
        {"metric": "bootstrap_anchor_corr_mean", "value": float(np.nanmean([r["value"] for r in bootstrap_rows]))},
        {"metric": "bootstrap_anchor_corr_std", "value": float(np.nanstd([r["value"] for r in bootstrap_rows]))},
    ]
    validation_df = pd.DataFrame(validation_rows)
    loo_df = pd.DataFrame(loo_rows)
    long_weights = weight_summary.copy()
    long_weights["label"] = long_weights["concept_slug"].map(ECONOMIC_TOPICS)
    loo_df["value"] = loo_df["correlation_with_full_axis"]
    loo_df = loo_df[["metric", "anchor", "value"]]
    boot_df = pd.DataFrame(bootstrap_rows)
    extra_df = pd.concat([validation_df, loo_df, boot_df], ignore_index=True, sort=False)
    return extra_df, long_weights


def compute_stage_snapshot(
    df: pd.DataFrame,
    monthly_pos_term: pd.DataFrame,
    months: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    mat, feat = build_party_topic_matrix(df, months, list(ECONOMIC_TOPICS.keys()))
    party_summary = party_summary_from_feat(feat)
    affinity = build_party_affinity(mat)
    pos = monthly_pos_term[monthly_pos_term["month_key"].isin(months)].groupby("party")["position"].mean()
    return mat, feat, party_summary, affinity, pos


def compute_stage_metric(
    stage: str,
    column: str,
    windows: dict[str, list[str]],
    topic_panel: pd.DataFrame,
    macro_panel: pd.DataFrame,
    stage_payload: dict[str, dict[str, pd.DataFrame]],
    stage_pos: dict[str, pd.Series],
) -> float:
    summary = stage_payload[stage]["party_summary"]
    if column == "spread":
        vals = stage_pos[stage].dropna()
        return float(vals.max() - vals.min()) if not vals.empty else np.nan
    if column == "macro_pressure":
        sub = macro_panel[macro_panel["month_key"].isin(windows[stage])]
        return float(sub["macro_pressure_index"].mean()) if not sub.empty else np.nan
    if column == "econ_salience":
        sub = topic_panel[(topic_panel["month_key"].isin(windows[stage])) & (topic_panel["concept_category"] == "macro_economy")]
        return float(sub["share"].mean()) if not sub.empty else np.nan
    if column == "constitutional_share":
        sub = topic_panel[(topic_panel["month_key"].isin(windows[stage])) & (topic_panel["concept_category"] == "constitutional_conflict")]
        return float(sub["share"].mean()) if not sub.empty else np.nan
    if column == "security_share":
        sub = topic_panel[(topic_panel["month_key"].isin(windows[stage])) & (topic_panel["concept_category"] == "security_conflict")]
        return float(sub["share"].mean()) if not sub.empty else np.nan
    if column == "owner_share":
        if summary.empty:
            return np.nan
        return float(summary["agenda_share"].max())
    if column == "owner_hhi":
        return owner_concentration_hhi(summary)
    if column == "negative_edge_share":
        return negative_edge_share(stage_payload[stage]["affinity"])
    if column == "signed_modularity":
        return signed_modularity(stage_payload[stage]["affinity"], stage_pos[stage])
    return np.nan


def build_party_topic_matrix(df: pd.DataFrame, month_keys: list[str], concepts: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[df["month_key"].isin(month_keys)].copy()
    if concepts is not None:
        sub = sub[sub["concept_slug"].isin(concepts)].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    feat = signed_agenda_feature(sub, ["party", "concept_slug"])
    mat = feat.pivot_table(index="party", columns="concept_slug", values="feature", fill_value=0.0)
    return mat, feat


def party_summary_from_feat(feat: pd.DataFrame) -> pd.DataFrame:
    if feat.empty:
        return pd.DataFrame(columns=["party", "mentions", "positive_hits", "negative_hits", "balance", "agenda_share", "negative_rate", "positive_rate"])
    party = feat.groupby("party", as_index=False)[["mention_count", "beme_positive_hits", "beme_negative_hits"]].sum()
    total_mentions = max(float(party["mention_count"].sum()), 1.0)
    party["balance"] = (party["beme_positive_hits"] - party["beme_negative_hits"]) / (party["beme_positive_hits"] + party["beme_negative_hits"] + 1.0)
    party["agenda_share"] = party["mention_count"] / total_mentions
    party["negative_rate"] = party["beme_negative_hits"] / party["mention_count"].replace(0, np.nan)
    party["positive_rate"] = party["beme_positive_hits"] / party["mention_count"].replace(0, np.nan)
    party = party.fillna(0.0).rename(
        columns={
            "mention_count": "mentions",
            "beme_positive_hits": "positive_hits",
            "beme_negative_hits": "negative_hits",
        }
    )
    return party.sort_values("mentions", ascending=False).reset_index(drop=True)


def monthly_topic_panel(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["month_key", "concept_category"], as_index=False)[["mention_count", "beme_positive_hits", "beme_negative_hits"]]
        .sum()
    )
    total = agg.groupby("month_key")["mention_count"].transform("sum").replace(0, np.nan)
    agg["share"] = agg["mention_count"] / total
    agg["neg_rate"] = agg["beme_negative_hits"] / agg["mention_count"].replace(0, np.nan)
    return agg.fillna(0.0)


def monthly_economic_party_panel(df: pd.DataFrame) -> pd.DataFrame:
    eco = df[df["concept_slug"].isin(ECONOMIC_TOPICS.keys())].copy()
    if eco.empty:
        return pd.DataFrame()
    agg = (
        eco.groupby(["month_key", "party"], as_index=False)[["mention_count", "beme_positive_hits", "beme_negative_hits"]]
        .sum()
    )
    total_month = agg.groupby("month_key")["mention_count"].transform("sum").replace(0, np.nan)
    agg["agenda_share"] = agg["mention_count"] / total_month
    agg["balance"] = (agg["beme_positive_hits"] - agg["beme_negative_hits"]) / (agg["beme_positive_hits"] + agg["beme_negative_hits"] + 1.0)
    agg["negative_rate"] = agg["beme_negative_hits"] / agg["mention_count"].replace(0, np.nan)
    return agg.fillna(0.0)


def build_party_affinity(mat: pd.DataFrame) -> pd.DataFrame:
    if mat.empty or len(mat) < 2:
        return pd.DataFrame()
    z = zscore_frame(mat.astype(float))
    norm = np.linalg.norm(z.to_numpy(dtype=float), axis=1)
    norm[norm == 0] = 1.0
    unit = z.div(norm, axis=0)
    sim = unit.to_numpy(dtype=float) @ unit.to_numpy(dtype=float).T
    arr = np.array(sim, dtype=float, copy=True)
    np.fill_diagonal(arr, 0.0)
    out = pd.DataFrame(arr, index=mat.index, columns=mat.index)
    return out


def owner_concentration_hhi(summary: pd.DataFrame) -> float:
    if summary.empty or "agenda_share" not in summary.columns:
        return np.nan
    shares = summary["agenda_share"].to_numpy(dtype=float)
    return float(np.square(shares).sum())


def negative_edge_share(affinity: pd.DataFrame) -> float:
    if affinity.empty:
        return np.nan
    pos_mass = 0.0
    neg_mass = 0.0
    for i in range(len(affinity.index)):
        for j in range(i + 1, len(affinity.columns)):
            val = float(affinity.iloc[i, j])
            if val < 0:
                neg_mass += abs(val)
            else:
                pos_mass += abs(val)
    total = pos_mass + neg_mass
    return float(neg_mass / total) if total > 0 else np.nan


def partition_from_positions(positions: pd.Series, parties: list[str]) -> np.ndarray:
    vals = positions.reindex(parties).fillna(0.0).astype(float)
    if vals.empty:
        return np.zeros(0, dtype=int)
    threshold = 0.0
    if bool((vals >= 0).all()) or bool((vals <= 0).all()):
        threshold = float(vals.median())
    labels = np.where(vals.to_numpy(dtype=float) >= threshold, 1, -1)
    if len(np.unique(labels)) == 1 and len(labels) > 1:
        order = np.argsort(vals.to_numpy(dtype=float))
        labels = np.ones(len(labels), dtype=int)
        labels[order[: max(1, len(labels) // 2)]] = -1
    return labels


def signed_modularity(affinity: pd.DataFrame, positions: pd.Series) -> float:
    if affinity.empty or len(affinity.index) < 2:
        return np.nan
    parties = list(affinity.index)
    comm = partition_from_positions(positions, parties)
    delta = (comm[:, None] == comm[None, :]).astype(float)
    arr = affinity.to_numpy(dtype=float)
    pos = np.clip(arr, 0.0, None)
    neg = np.clip(-arr, 0.0, None)

    def modularity_component(weights: np.ndarray) -> float:
        m = weights.sum() / 2.0
        if m <= 0:
            return 0.0
        k = weights.sum(axis=1)
        expected = np.outer(k, k) / (2.0 * m)
        return float(((weights - expected) * delta).sum() / (2.0 * m))

    return modularity_component(pos) - modularity_component(neg)


def brokerage_concentration_hhi(scores: pd.DataFrame) -> float:
    if scores.empty or "gatekeeper_score" not in scores.columns:
        return np.nan
    party_scores = (
        scores.groupby("party", as_index=False)["gatekeeper_score"]
        .mean()
        .assign(gatekeeper_score=lambda df: df["gatekeeper_score"].clip(lower=0.0))
    )
    total = float(party_scores["gatekeeper_score"].sum())
    if total <= 0:
        return np.nan
    shares = party_scores["gatekeeper_score"].to_numpy(dtype=float) / total
    return float(np.square(shares).sum())


def build_stage_gatekeepers(df: pd.DataFrame, month_keys: list[str]) -> pd.DataFrame:
    sub = df[df["month_key"].isin(month_keys) & df["concept_slug"].isin(ECONOMIC_TOPICS.keys())].copy()
    if sub.empty:
        return pd.DataFrame(columns=["speaker", "party", "gatekeeper_score", "cross_pressure"])
    feat = signed_agenda_feature(sub, ["speaker", "party", "concept_slug"])
    mat = feat.pivot_table(index=["speaker", "party"], columns="concept_slug", values="feature", fill_value=0.0)
    meta = feat.groupby(["speaker", "party"], as_index=False)["mention_count"].sum().rename(columns={"mention_count": "total_mentions"})
    if mat.shape[0] < 6:
        return pd.DataFrame(columns=["speaker", "party", "gatekeeper_score", "cross_pressure"])
    adj, _ = build_sparse_signed_graph(mat)
    eigvals, eigvecs = np.linalg.eigh(adj)
    lead_idx = int(np.argmax(np.abs(eigvals)))
    v = eigvecs[:, lead_idx]
    tau = float(np.quantile(np.abs(v), 0.60))
    membership = mat.reset_index()[["speaker", "party"]].copy().merge(meta, on=["speaker", "party"], how="left")
    membership["community"] = np.where(v >= tau, 1, np.where(v <= -tau, -1, 0))
    membership["community_label"] = membership["community"].map({1: "Pole +", -1: "Pole -", 0: "Neutral"})
    scores, _ = compute_gatekeeper_scores(membership, adj)
    return scores


def monthly_positions_for_term(monthly_positions: pd.DataFrame, term: int) -> pd.DataFrame:
    return monthly_positions[monthly_positions["term"] == term].copy()


def major_party_list(parties: list[str]) -> list[str]:
    ordered = [p for p in MAJOR_PARTIES if p in parties]
    ordered.extend([p for p in parties if p not in ordered])
    return ordered


def top_abs_pairs(affinity: pd.DataFrame, top_n: int = 7) -> list[tuple[str, str, float]]:
    pairs = []
    for i, src in enumerate(affinity.index):
        for j, dst in enumerate(affinity.columns):
            if j <= i:
                continue
            val = float(affinity.iloc[i, j])
            pairs.append((src, dst, val))
    pairs = sorted(pairs, key=lambda item: abs(item[2]), reverse=True)
    strong = [pair for pair in pairs if abs(pair[2]) >= 0.10]
    return strong[:top_n]


def centered_offsets(count: int, step: float) -> list[float]:
    if count <= 1:
        return [0.0]
    mid = (count - 1) / 2.0
    return [float((idx - mid) * step) for idx in range(count)]


def enforce_min_x_gap(xs: pd.Series, min_gap: float = 0.26, bound: float = 1.55) -> pd.Series:
    if xs.empty:
        return xs
    ordered = list(xs.sort_values().index)
    adjusted: dict[str, float] = {}
    cursor: float | None = None
    for party in ordered:
        value = float(xs.loc[party])
        if cursor is None:
            cursor = value
        else:
            cursor = max(value, cursor + min_gap)
        adjusted[party] = cursor
    arr = np.array([adjusted[p] for p in ordered], dtype=float)
    arr = arr - float(arr.mean())
    max_abs = float(np.max(np.abs(arr))) if arr.size else 1.0
    if max_abs > bound and max_abs > 0:
        arr = arr * (bound / max_abs)
    return pd.Series({party: float(val) for party, val in zip(ordered, arr)}).reindex(xs.index)


def layout_axis_parties(xs: pd.Series, proximity: float = 0.22, inner_step: float = 0.24) -> dict[str, tuple[float, float]]:
    ordered = sorted(xs.index, key=lambda party: (float(xs.get(party, 0.0)), party))
    clusters: list[list[tuple[str, float]]] = []
    for party in ordered:
        x = float(xs.get(party, 0.0))
        if not clusters or abs(x - clusters[-1][-1][1]) > proximity:
            clusters.append([(party, x)])
        else:
            clusters[-1].append((party, x))

    if len(clusters) == 1:
        base_positions = [0.0]
    else:
        raw = np.linspace(0.94, -0.94, len(clusters)).tolist()
        mid = (len(raw) - 1) / 2.0
        order = sorted(range(len(raw)), key=lambda i: (abs(i - mid), i))
        base_positions = [raw[i] for i in order]

    positions: dict[str, tuple[float, float]] = {}
    for base_y, cluster in zip(base_positions, clusters):
        for (party, x), offset in zip(cluster, centered_offsets(len(cluster), inner_step)):
            positions[party] = (x, float(base_y + offset))
    return positions


def nonlinear_affinity_layout(
    parties: list[str],
    xs_scaled: pd.Series,
    pairs: list[tuple[str, str, float]],
) -> dict[str, tuple[float, float]]:
    order = sorted(parties, key=lambda party: float(xs_scaled.get(party, 0.0)))
    n = max(len(order), 1)
    arc_angles = np.linspace(-1.05, 1.05, n)
    init_pos = {
        party: (
            float(xs_scaled.get(party, 0.0)),
            float(0.88 * np.sin(angle) + 0.18 * np.sin(2.0 * angle)),
        )
        for party, angle in zip(order, arc_angles)
    }

    G = nx.Graph()
    for party in parties:
        G.add_node(party)
    for src, dst, val in pairs:
        G.add_edge(src, dst, weight=max(abs(float(val)), 0.08))

    try:
        spring = nx.spring_layout(
            G,
            pos=init_pos,
            seed=42,
            iterations=300,
            k=1.6 / math.sqrt(max(len(parties), 2)),
            weight="weight",
        )
    except Exception:
        spring = init_pos

    spring_x = pd.Series({party: float(spring[party][0]) for party in parties})
    spring_y = pd.Series({party: float(spring[party][1]) for party in parties})
    spring_y = spring_y - float(spring_y.mean())
    max_abs_y = float(spring_y.abs().max()) or 1.0
    spring_y = spring_y / max_abs_y

    mixed_x = 0.82 * xs_scaled.reindex(parties).fillna(0.0) + 0.28 * spring_x.reindex(parties).fillna(0.0)
    mixed_x = enforce_min_x_gap(mixed_x, min_gap=0.34, bound=1.78)
    final_y = 1.02 * spring_y.reindex(parties).fillna(0.0)
    return {party: (float(mixed_x.loc[party]), float(final_y.loc[party])) for party in parties}


def arc_rank_layout(parties: list[str], xs_scaled: pd.Series) -> dict[str, tuple[float, float]]:
    ordered = sorted(parties, key=lambda party: float(xs_scaled.get(party, 0.0)))
    n = max(len(ordered), 1)
    base_x = np.linspace(-1.62, 1.62, n)
    wave = 0.42 * np.sin(np.linspace(-1.25, 1.25, n))
    zig = np.array(([0.22, -0.16, 0.10, -0.08] * ((n + 3) // 4))[:n], dtype=float)
    y = wave + zig - 0.10
    x = base_x.copy()
    if n > 1:
        x += 0.10 * np.cos(np.linspace(-1.0, 1.0, n))
    positions: dict[str, tuple[float, float]] = {}
    for party, xv, yv in zip(ordered, x, y):
        positions[party] = (float(xv), float(yv))
    return positions


def draw_curved_edge(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    linewidth: float,
    alpha: float,
    curvature: float,
) -> None:
    x1, y1 = start
    x2, y2 = end
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0
    dx = x2 - x1
    dy = y2 - y1
    nx = -dy
    ny = dx
    norm = math.hypot(nx, ny) or 1.0
    mx += curvature * nx / norm
    my += curvature * ny / norm
    t = np.linspace(0.0, 1.0, 50)
    xs = ((1 - t) ** 2) * x1 + 2 * (1 - t) * t * mx + (t**2) * x2
    ys = ((1 - t) ** 2) * y1 + 2 * (1 - t) * t * my + (t**2) * y2
    ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=1)


def draw_affinity_network(ax: plt.Axes, affinity: pd.DataFrame, positions: pd.Series, party_metrics: pd.DataFrame, title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    if affinity.empty:
        ax.text(0.5, 0.5, "No graph", ha="center", va="center")
        return

    parties = major_party_list(list(affinity.index))
    affinity = affinity.reindex(index=parties, columns=parties, fill_value=0.0)
    pairs = top_abs_pairs(affinity, top_n=min(12, max(8, len(parties) + 2)))
    if not pairs:
        ax.text(0.5, 0.5, "No strong edges", ha="center", va="center")
        return

    metric_map = party_metrics.set_index("party") if not party_metrics.empty else pd.DataFrame(index=parties)
    xs = positions.reindex(parties).fillna(0.0)
    if xs.max() == xs.min():
        xs_scaled = pd.Series(np.linspace(-1.15, 1.15, len(parties)), index=parties)
    else:
        xs_scaled = pd.Series(
            np.interp(xs.to_numpy(dtype=float), (float(xs.min()), float(xs.max())), (-1.15, 1.15)),
            index=parties,
        )
    xs_scaled = enforce_min_x_gap(xs_scaled, min_gap=0.30, bound=1.72)
    node_positions = arc_rank_layout(parties, xs_scaled)

    G = nx.Graph()
    for party in parties:
        G.add_node(party)
    for src, dst, val in pairs:
        G.add_edge(src, dst, weight=abs(val), sign=np.sign(val))

    party_rank = {party: idx for idx, party in enumerate(sorted(parties, key=lambda p: float(xs_scaled.get(p, 0.0))))}
    edge_mid = (len(pairs) - 1) / 2.0
    for edge_idx, (src, dst, val) in enumerate(pairs):
        x1, y1 = node_positions[src]
        x2, y2 = node_positions[dst]
        rank_gap = abs(party_rank[src] - party_rank[dst])
        sign_seed = -1.0 if ((party_rank[src] + party_rank[dst] + edge_idx) % 2 == 0) else 1.0
        curvature = sign_seed * (0.06 + 0.010 * rank_gap + 0.012 * abs(y1 - y2) + 0.008 * abs(edge_idx - edge_mid))
        draw_curved_edge(
            ax,
            (x1, y1),
            (x2, y2),
            color="white",
            linewidth=3.2 + 3.6 * min(abs(val), 1.0),
            alpha=0.90,
            curvature=curvature,
        )
        draw_curved_edge(
            ax,
            (x1, y1),
            (x2, y2),
            color="#2563eb" if val >= 0 else "#dc2626",
            linewidth=1.65 + 3.2 * min(abs(val), 1.0),
            alpha=0.74 + 0.16 * min(abs(val), 1.0),
            curvature=curvature,
        )

    max_mentions = float(metric_map["mentions"].max()) if (not metric_map.empty and "mentions" in metric_map.columns) else 1.0
    for idx, party in enumerate(parties):
        x, y = node_positions[party]
        mentions = float(metric_map.loc[party, "mentions"]) if party in metric_map.index else 1.0
        size = 108 + 470 * min(mentions / max(max_mentions, 1.0), 1.0)
        ax.scatter(
            [x],
            [y],
            s=size,
            color=PARTY_COLORS.get(party, "#64748b"),
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        norm = math.hypot(x, y) or 1.0
        label_x = x + 0.24 * (x / norm)
        label_y = y + 0.17 * (y / norm)
        ax.plot([x, label_x], [y, label_y], color="#9ca3af", linewidth=1.0, alpha=0.75, zorder=2)
        ax.text(
            label_x,
            label_y,
            PARTY_SHORT.get(party, party),
            ha="left" if label_x >= x else "right",
            va="center",
            fontsize=8.6,
            color="#111827",
            zorder=4,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#e5e7eb", "alpha": 0.97},
            clip_on=False,
        )

    ax.set_xlim(xs_scaled.min() - 1.18, xs_scaled.max() + 1.18)
    ax.set_ylim(-1.28, 1.28)


def draw_bipartite(ax: plt.Axes, mat: pd.DataFrame, party_metrics: pd.DataFrame, title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    if mat.empty:
        ax.text(0.5, 0.5, "No graph", ha="center", va="center")
        return
    parties = major_party_list(list(mat.index))
    topics = list(ECONOMIC_TOPICS.keys())
    if not topics:
        ax.text(0.5, 0.5, "No topics", ha="center", va="center")
        return
    mat = mat.reindex(index=parties, columns=topics, fill_value=0.0)
    party_y = np.linspace(1.02, -1.02, len(parties))
    topic_y = np.linspace(0.90, -0.90, len(topics))
    party_pos = {party: (0.12, float(party_y[i])) for i, party in enumerate(parties)}
    topic_pos = {topic: (2.18, float(topic_y[i])) for i, topic in enumerate(topics)}
    threshold = max(0.004, float(np.abs(mat.to_numpy()).mean() * 0.12))
    keep_edges: set[tuple[str, str]] = set()
    for party in parties:
        row = mat.loc[party].abs().sort_values(ascending=False)
        for topic, value in row.head(min(3, len(row))).items():
            if float(value) > 0:
                keep_edges.add((party, topic))
    for topic in topics:
        col = mat[topic].abs().sort_values(ascending=False)
        for party, value in col.head(min(4, len(col))).items():
            if float(value) > 0:
                keep_edges.add((party, topic))

    for party in parties:
        for topic in topics:
            val = float(mat.loc[party, topic])
            if abs(val) < threshold and (party, topic) not in keep_edges:
                continue
            x1, y1 = party_pos[party]
            x2, y2 = topic_pos[topic]
            party_center = (parties.index(party) - (len(parties) - 1) / 2.0) / max(len(parties), 1)
            topic_center = (topics.index(topic) - (len(topics) - 1) / 2.0) / max(len(topics), 1)
            curve = 0.12 * party_center - 0.16 * topic_center
            draw_curved_edge(
                ax,
                (x1, y1),
                (x2, y2),
                color="white",
                linewidth=2.8 + 8.8 * min(abs(val), 0.35),
                alpha=0.98,
                curvature=curve,
            )
            draw_curved_edge(
                ax,
                (x1, y1),
                (x2, y2),
                color="#0f766e" if val >= 0 else "#b91c1c",
                linewidth=1.8 + 9.8 * min(abs(val), 0.35),
                alpha=0.50 + 0.34 * min(abs(val) / 0.35, 1.0),
                curvature=curve,
            )

    metric_map = party_metrics.set_index("party") if not party_metrics.empty else pd.DataFrame(index=parties)
    max_mentions = float(metric_map["mentions"].max()) if (not metric_map.empty and "mentions" in metric_map.columns) else 1.0
    for party in parties:
        x, y = party_pos[party]
        mentions = float(metric_map.loc[party, "mentions"]) if party in metric_map.index else 1.0
        size = 128 + 520 * min(mentions / max(max_mentions, 1.0), 1.0)
        ax.scatter([x], [y], s=size, color=PARTY_COLORS.get(party, "#64748b"), edgecolor="white", linewidth=1.1, zorder=3)
        ax.text(
            x - 0.18,
            y,
            PARTY_SHORT.get(party, party),
            ha="right",
            va="center",
            fontsize=8.8,
            color="#111827",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#e5e7eb", "alpha": 0.97},
        )

    for topic in topics:
        x, y = topic_pos[topic]
        topic_strength = float(mat[topic].abs().sum()) if topic in mat.columns else 0.0
        size = 340 + 240 * min(topic_strength / max(float(mat.abs().sum().max()), 1.0), 1.0)
        ax.scatter([x], [y], s=size, color="#e7e5e4", edgecolor="#78716c", linewidth=1.0, zorder=3)
        ax.text(
            x + 0.18,
            y,
            ECONOMIC_TOPICS[topic],
            ha="left",
            va="center",
            fontsize=8.8,
            color="#44403c",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#e5e7eb", "alpha": 0.97},
        )

    ax.set_xlim(-0.62, 2.52)
    ax.set_ylim(-1.14, 1.14)


def plot_economic_triptych(
    term: int,
    event_slug: str,
    event_label: str,
    stage_data: dict[str, dict[str, pd.DataFrame]],
    stage_positions: dict[str, pd.Series],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(22.5, 12.6), facecolor="white")
    stage_order = ["pre", "shock", "post"]
    for col, stage in enumerate(stage_order):
        label = {"pre": "Pre-shock", "shock": "Shock month", "post": "Post-shock"}[stage]
        draw_affinity_network(
            axes[0, col],
            stage_data[stage]["affinity"],
            stage_positions[stage],
            stage_data[stage]["party_summary"],
            f"{label}: party-party economic alignment",
        )
        draw_bipartite(
            axes[1, col],
            stage_data[stage]["mat"],
            stage_data[stage]["party_summary"],
            f"{label}: party-topic economic graph",
        )

    fig.suptitle(
        f"Term {term} · {event_label}\nEconomic event-centred network triptych",
        fontsize=16,
        y=0.98,
    )
    fig.subplots_adjust(left=0.025, right=0.99, top=0.91, bottom=0.05, wspace=0.12, hspace=0.18)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_macro_overlay(
    term: int,
    event_label: str,
    event_month: str,
    macro_panel: pd.DataFrame,
    topic_panel: pd.DataFrame,
    party_panel: pd.DataFrame,
    monthly_positions: pd.DataFrame,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(17, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.25)

    event_ts = pd.Period(event_month, freq="M").to_timestamp()
    window_start = event_ts - pd.DateOffset(months=6)
    window_end = event_ts + pd.DateOffset(months=6)

    macro_sub = macro_panel[(macro_panel["date"] >= window_start) & (macro_panel["date"] <= window_end)].copy()
    topic_sub = topic_panel[(topic_panel["date"] >= window_start) & (topic_panel["date"] <= window_end)].copy()
    party_sub = party_panel[(party_panel["date"] >= window_start) & (party_panel["date"] <= window_end)].copy()
    pos_sub = monthly_positions[(monthly_positions["date"] >= window_start) & (monthly_positions["date"] <= window_end)].copy()

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(macro_sub["date"], macro_sub["macro_pressure_index"], color="#991b1b", linewidth=2.4, label="Macro pressure index")
    ax1.axvline(event_ts, color="#111827", linestyle="--", linewidth=1.3)
    ax1.set_title("External macro pressure")
    ax1.grid(alpha=0.2)
    ax1.legend(frameon=False, fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    for topic in ["macro_economy", "constitutional_conflict", "security_conflict"]:
        sub = topic_sub[topic_sub["concept_category"] == topic]
        if sub.empty:
            continue
        ax2.plot(sub["date"], sub["share"], linewidth=2.0, label=FOCUS_TOPIC_CATEGORIES.get(topic, topic))
    ax2.axvline(event_ts, color="#111827", linestyle="--", linewidth=1.3)
    ax2.set_title("Topic substitution around the shock")
    ax2.grid(alpha=0.2)
    ax2.legend(frameon=False, fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    for party in [p for p in MAJOR_PARTIES if p in pos_sub["party"].unique()][:5]:
        ps = pos_sub[pos_sub["party"] == party]
        ax3.plot(ps["date"], ps["position"], linewidth=2.0, marker="o", markersize=3, color=PARTY_COLORS.get(party, "#64748b"), label=PARTY_SHORT.get(party, party))
    ax3.axvline(event_ts, color="#111827", linestyle="--", linewidth=1.3)
    ax3.axhline(0.0, color="#cbd5e1", linestyle=":", linewidth=1.0)
    ax3.set_title("Common economic latent axis")
    ax3.grid(alpha=0.2)
    ax3.legend(frameon=False, fontsize=8, ncol=2)

    ax4 = fig.add_subplot(gs[1, 1])
    pivot = (
        party_sub[party_sub["party"].isin([p for p in MAJOR_PARTIES if p in party_sub["party"].unique()][:5])]
        .pivot_table(index="date", columns="party", values="balance")
        .sort_index()
    )
    for party in pivot.columns:
        ax4.plot(pivot.index, pivot[party], linewidth=2.0, marker="o", markersize=3, color=PARTY_COLORS.get(party, "#64748b"), label=PARTY_SHORT.get(party, party))
    ax4.axvline(event_ts, color="#111827", linestyle="--", linewidth=1.3)
    ax4.axhline(0.0, color="#cbd5e1", linestyle=":", linewidth=1.0)
    ax4.set_title("Economic attack / defense balance")
    ax4.grid(alpha=0.2)
    ax4.legend(frameon=False, fontsize=8, ncol=2)

    fig.suptitle(f"Term {term} · {event_label}\nMacro indicators and speech-network responses", fontsize=16, y=0.98)
    fig.text(
        0.01,
        0.015,
        "Panel 1 builds a macro pressure index by standardizing FX depreciation, inflation, equity stress, and policy-rate shifts. "
        "Panel 2 shows whether the shock pulls parliamentary attention into macroeconomy or displaces it toward constitutional / security conflict. "
        "Panel 3 projects parties onto a common economy-specific latent axis that is comparable across terms. "
        "Panel 4 tracks whether parties defend economic policy (positive balance) or attack it (negative balance).",
        fontsize=9.8,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def strongest_party(summary: pd.DataFrame, column: str, direction: str = "min") -> tuple[str | None, float | None]:
    if summary.empty or column not in summary.columns:
        return None, None
    row = summary.sort_values(column, ascending=(direction == "min")).iloc[0]
    return str(row["party"]), float(row[column])


def compute_term_event_outputs(
    term: int,
    monthly_positions: pd.DataFrame,
    macro_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    term_root = BASE / f"Term_{term}"
    fig_root = term_root / "Figures"
    csv_root = term_root / "CSVs"
    note_root = term_root / "Notes"
    fig_root.mkdir(parents=True, exist_ok=True)
    csv_root.mkdir(parents=True, exist_ok=True)
    note_root.mkdir(parents=True, exist_ok=True)

    df = read_term_edges(term)
    month_keys = available_months(df)
    topic_panel = monthly_topic_panel(df)
    topic_panel["date"] = pd.to_datetime(topic_panel["month_key"] + "-01")
    econ_party_panel = monthly_economic_party_panel(df)
    if not econ_party_panel.empty:
        econ_party_panel["date"] = pd.to_datetime(econ_party_panel["month_key"] + "-01")
    monthly_pos_term = monthly_positions_for_term(monthly_positions, term).copy()
    monthly_pos_term["date"] = pd.to_datetime(monthly_pos_term["month_key"] + "-01")

    event_rows = []
    party_rows = []
    gate_rows = []
    null_rows = []
    registry_rows = []
    figure_rows: list[dict[str, str]] = []
    brief_lines = [
        f"Term {term} — Economic publication brief",
        "=" * 72,
        "",
    ]

    for event in EVENT_CATALOG.get(term, []):
        effective_month = resolve_event_month(event["target_month"], month_keys)
        windows = stage_months(effective_month, pre=3, post=3)
        eligible_month_pool = eligible_event_months(month_keys, pre=3, post=3)
        event_slug = slugify(event["label"])
        stage_payload: dict[str, dict[str, pd.DataFrame]] = {}
        stage_pos: dict[str, pd.Series] = {}

        for stage, months in windows.items():
            mat, feat, party_summary, affinity, pos = compute_stage_snapshot(df, monthly_pos_term, months)
            stage_payload[stage] = {
                "mat": mat.reindex(columns=list(ECONOMIC_TOPICS.keys()), fill_value=0.0),
                "feat": feat,
                "party_summary": party_summary,
                "affinity": affinity,
            }
            stage_pos[stage] = pos

            gatekeeper_scores = build_stage_gatekeepers(df, months)
            if not gatekeeper_scores.empty:
                top_gate = gatekeeper_scores.sort_values("gatekeeper_score", ascending=False).iloc[0]
                stage_brokerage_hhi = brokerage_concentration_hhi(gatekeeper_scores)
                stage_party_scores = (
                    gatekeeper_scores.groupby("party", as_index=False)
                    .agg(
                        mean_gatekeeper_score=("gatekeeper_score", "mean"),
                        max_gatekeeper_score=("gatekeeper_score", "max"),
                        n_brokers=("speaker", "nunique"),
                    )
                    .sort_values("mean_gatekeeper_score", ascending=False)
                    .reset_index(drop=True)
                )
                top_party_gate = stage_party_scores.iloc[0]
                gate_rows.append(
                    {
                        "term": term,
                        "event_label": event["label"],
                        "event_slug": event_slug,
                        "event_month": effective_month,
                        "stage": stage,
                        "speaker": top_gate["speaker"],
                        "party": top_gate["party"],
                        "gatekeeper_score": float(top_gate["gatekeeper_score"]),
                        "cross_pressure": float(top_gate["cross_pressure"]),
                        "brokerage_concentration_hhi": stage_brokerage_hhi,
                        "top_broker_party_agg": top_party_gate["party"],
                        "top_party_mean_score": float(top_party_gate["mean_gatekeeper_score"]),
                        "top_party_max_score": float(top_party_gate["max_gatekeeper_score"]),
                        "top_party_n_brokers": int(top_party_gate["n_brokers"]),
                    }
                )

            for row in party_summary.itertuples(index=False):
                party_rows.append(
                    {
                        "term": term,
                        "event_label": event["label"],
                        "event_slug": event_slug,
                        "event_month": effective_month,
                        "stage": stage,
                        "party": row.party,
                        "mentions": float(row.mentions),
                        "agenda_share": float(row.agenda_share),
                        "balance": float(row.balance),
                        "negative_rate": float(row.negative_rate),
                        "positive_rate": float(row.positive_rate),
                        "position": float(stage_pos[stage].get(row.party, np.nan)),
                    }
                )

        shock_summary = stage_payload["shock"]["party_summary"]
        pre_summary = stage_payload["pre"]["party_summary"]
        post_summary = stage_payload["post"]["party_summary"]
        pre_owner = (
            str(pre_summary.sort_values("agenda_share", ascending=False).iloc[0]["party"])
            if not pre_summary.empty
            else None
        )
        top_attack_party, top_attack_balance = strongest_party(shock_summary, "balance", direction="min")
        top_defense_party, top_defense_balance = strongest_party(shock_summary, "balance", direction="max")
        top_owner = (
            str(shock_summary.sort_values("agenda_share", ascending=False).iloc[0]["party"])
            if not shock_summary.empty
            else None
        )
        post_owner = (
            str(post_summary.sort_values("agenda_share", ascending=False).iloc[0]["party"])
            if not post_summary.empty
            else None
        )

        spread_pre = compute_stage_metric("pre", "spread", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        spread_shock = compute_stage_metric("shock", "spread", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        spread_post = compute_stage_metric("post", "spread", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        macro_pre = compute_stage_metric("pre", "macro_pressure", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        macro_shock = compute_stage_metric("shock", "macro_pressure", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        macro_post = compute_stage_metric("post", "macro_pressure", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        sal_pre = compute_stage_metric("pre", "econ_salience", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        sal_shock = compute_stage_metric("shock", "econ_salience", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        sal_post = compute_stage_metric("post", "econ_salience", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        const_pre = compute_stage_metric("pre", "constitutional_share", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        const_shock = compute_stage_metric("shock", "constitutional_share", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        const_post = compute_stage_metric("post", "constitutional_share", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        sec_pre = compute_stage_metric("pre", "security_share", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        sec_shock = compute_stage_metric("shock", "security_share", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        sec_post = compute_stage_metric("post", "security_share", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        owner_hhi_pre = compute_stage_metric("pre", "owner_hhi", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        owner_hhi_shock = compute_stage_metric("shock", "owner_hhi", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        owner_hhi_post = compute_stage_metric("post", "owner_hhi", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        neg_share_pre = compute_stage_metric("pre", "negative_edge_share", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        neg_share_shock = compute_stage_metric("shock", "negative_edge_share", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        neg_share_post = compute_stage_metric("post", "negative_edge_share", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        mod_pre = compute_stage_metric("pre", "signed_modularity", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        mod_shock = compute_stage_metric("shock", "signed_modularity", windows, topic_panel, macro_panel, stage_payload, stage_pos)
        mod_post = compute_stage_metric("post", "signed_modularity", windows, topic_panel, macro_panel, stage_payload, stage_pos)

        competing_gain_shock = max(0.0, const_shock - const_pre) + max(0.0, sec_shock - sec_pre) if pd.notna(const_pre) and pd.notna(const_shock) and pd.notna(sec_pre) and pd.notna(sec_shock) else np.nan
        econ_post_delta = sal_post - sal_pre if pd.notna(sal_post) and pd.notna(sal_pre) else np.nan
        const_post_delta = const_post - const_pre if pd.notna(const_post) and pd.notna(const_pre) else np.nan
        sec_post_delta = sec_post - sec_pre if pd.notna(sec_post) and pd.notna(sec_pre) else np.nan
        competing_gain_post = max(0.0, const_post_delta) + max(0.0, sec_post_delta) if pd.notna(const_post_delta) and pd.notna(sec_post_delta) else np.nan
        displacement_shock = competing_gain_shock - (sal_shock - sal_pre) if pd.notna(competing_gain_shock) and pd.notna(sal_shock) and pd.notna(sal_pre) else np.nan
        displacement_post = competing_gain_post - econ_post_delta if pd.notna(competing_gain_post) and pd.notna(econ_post_delta) else np.nan

        registry_rows.append(
            {
                "term": term,
                "event_label": event["label"],
                "event_slug": event_slug,
                "event_kind": event["kind"],
                "paper_role": event["paper_role"],
                "rationale": event.get("rationale", ""),
                "event_month_requested": event["target_month"],
                "event_month_effective": effective_month,
                "pre_months": ",".join(windows["pre"]),
                "shock_months": ",".join(windows["shock"]),
                "post_months": ",".join(windows["post"]),
                "pre_length": len(windows["pre"]),
                "post_length": len(windows["post"]),
                "eligible_month_pool_size": len(eligible_month_pool),
            }
        )

        triptych_path = fig_root / f"term_{term}_economic_triptych_{event_slug}.png"
        overlay_path = fig_root / f"term_{term}_economic_overlay_{event_slug}.png"
        plot_economic_triptych(term, event_slug, event["label"], stage_payload, stage_pos, triptych_path)
        plot_macro_overlay(term, event["label"], effective_month, macro_panel, topic_panel, econ_party_panel, monthly_pos_term, overlay_path)
        figure_rows.extend(
            [
                {"term": term, "event_slug": event_slug, "event_label": event["label"], "figure_kind": "triptych", "path": str(triptych_path)},
                {"term": term, "event_slug": event_slug, "event_label": event["label"], "figure_kind": "overlay", "path": str(overlay_path)},
            ]
        )

        event_rows.append(
            {
                "term": term,
                "event_label": event["label"],
                "event_slug": event_slug,
                "event_month_requested": event["target_month"],
                "event_month_effective": effective_month,
                "event_kind": event["kind"],
                "paper_role": event["paper_role"],
                "pre_owner_party": pre_owner,
                "post_owner_party": post_owner,
                "macro_pressure_pre": macro_pre,
                "macro_pressure_shock": macro_shock,
                "macro_pressure_post": macro_post,
                "macro_pressure_delta": macro_shock - macro_pre if pd.notna(macro_pre) and pd.notna(macro_shock) else np.nan,
                "econ_salience_pre": sal_pre,
                "econ_salience_shock": sal_shock,
                "econ_salience_post": sal_post,
                "econ_salience_delta": sal_shock - sal_pre if pd.notna(sal_pre) and pd.notna(sal_shock) else np.nan,
                "econ_salience_post_delta": econ_post_delta,
                "spread_pre": spread_pre,
                "spread_shock": spread_shock,
                "spread_post": spread_post,
                "spread_delta": spread_shock - spread_pre if pd.notna(spread_pre) and pd.notna(spread_shock) else np.nan,
                "negative_edge_share_pre": neg_share_pre,
                "negative_edge_share_shock": neg_share_shock,
                "negative_edge_share_post": neg_share_post,
                "negative_edge_share_delta": neg_share_shock - neg_share_pre if pd.notna(neg_share_pre) and pd.notna(neg_share_shock) else np.nan,
                "signed_modularity_pre": mod_pre,
                "signed_modularity_shock": mod_shock,
                "signed_modularity_post": mod_post,
                "signed_modularity_delta": mod_shock - mod_pre if pd.notna(mod_pre) and pd.notna(mod_shock) else np.nan,
                "constitutional_share_pre": const_pre,
                "constitutional_share_shock": const_shock,
                "constitutional_share_post": const_post,
                "constitutional_share_delta": const_shock - const_pre if pd.notna(const_pre) and pd.notna(const_shock) else np.nan,
                "constitutional_share_post_delta": const_post_delta,
                "security_share_pre": sec_pre,
                "security_share_shock": sec_shock,
                "security_share_post": sec_post,
                "security_share_delta": sec_shock - sec_pre if pd.notna(sec_pre) and pd.notna(sec_shock) else np.nan,
                "security_share_post_delta": sec_post_delta,
                "top_attack_party": top_attack_party,
                "top_attack_balance": top_attack_balance,
                "top_defense_party": top_defense_party,
                "top_defense_balance": top_defense_balance,
                "top_owner_party": top_owner,
                "owner_shift": pre_owner != top_owner if pre_owner and top_owner else False,
                "owner_persistence": top_owner == post_owner if top_owner and post_owner else False,
                "owner_concentration_hhi_pre": owner_hhi_pre,
                "owner_concentration_hhi_shock": owner_hhi_shock,
                "owner_concentration_hhi_post": owner_hhi_post,
                "owner_concentration_hhi_delta": owner_hhi_shock - owner_hhi_pre if pd.notna(owner_hhi_pre) and pd.notna(owner_hhi_shock) else np.nan,
                "competing_gain_shock": competing_gain_shock,
                "competing_gain_post": competing_gain_post,
                "displacement_index_shock": displacement_shock,
                "displacement_index_post": displacement_post,
                "shock_to_post_persistence": displacement_post - displacement_shock if pd.notna(displacement_post) and pd.notna(displacement_shock) else np.nan,
                "displacement_persists": bool(pd.notna(displacement_shock) and pd.notna(displacement_post) and displacement_shock > 0 and displacement_post > 0),
            }
        )

        null_salience = []
        null_spread = []
        null_constitution = []
        null_security = []
        for candidate_month in eligible_month_pool:
            if candidate_month == effective_month:
                continue
            candidate_windows = stage_months(candidate_month, pre=3, post=3)
            candidate_payload: dict[str, dict[str, pd.DataFrame]] = {}
            candidate_pos: dict[str, pd.Series] = {}
            for stage, months in candidate_windows.items():
                c_mat, c_feat, c_summary, c_affinity, c_pos = compute_stage_snapshot(df, monthly_pos_term, months)
                candidate_payload[stage] = {
                    "mat": c_mat,
                    "feat": c_feat,
                    "party_summary": c_summary,
                    "affinity": c_affinity,
                }
                candidate_pos[stage] = c_pos
            cand_sal = compute_stage_metric("shock", "econ_salience", candidate_windows, topic_panel, macro_panel, candidate_payload, candidate_pos) - compute_stage_metric("pre", "econ_salience", candidate_windows, topic_panel, macro_panel, candidate_payload, candidate_pos)
            cand_spread = compute_stage_metric("shock", "spread", candidate_windows, topic_panel, macro_panel, candidate_payload, candidate_pos) - compute_stage_metric("pre", "spread", candidate_windows, topic_panel, macro_panel, candidate_payload, candidate_pos)
            cand_const = compute_stage_metric("shock", "constitutional_share", candidate_windows, topic_panel, macro_panel, candidate_payload, candidate_pos) - compute_stage_metric("pre", "constitutional_share", candidate_windows, topic_panel, macro_panel, candidate_payload, candidate_pos)
            cand_sec = compute_stage_metric("shock", "security_share", candidate_windows, topic_panel, macro_panel, candidate_payload, candidate_pos) - compute_stage_metric("pre", "security_share", candidate_windows, topic_panel, macro_panel, candidate_payload, candidate_pos)
            null_salience.append(cand_sal)
            null_spread.append(cand_spread)
            null_constitution.append(cand_const)
            null_security.append(cand_sec)

        null_sal_arr = np.array(null_salience, dtype=float)
        null_sp_arr = np.array(null_spread, dtype=float)
        null_const_arr = np.array(null_constitution, dtype=float)
        null_sec_arr = np.array(null_security, dtype=float)
        event_rows[-1]["econ_salience_null_mean"] = float(np.nanmean(null_sal_arr)) if len(null_sal_arr) else np.nan
        event_rows[-1]["econ_salience_null_std"] = float(np.nanstd(null_sal_arr)) if len(null_sal_arr) else np.nan
        event_rows[-1]["econ_salience_null_p"] = float((np.abs(null_sal_arr) >= abs(event_rows[-1]["econ_salience_delta"])).mean()) if len(null_sal_arr) and pd.notna(event_rows[-1]["econ_salience_delta"]) else np.nan
        event_rows[-1]["spread_null_mean"] = float(np.nanmean(null_sp_arr)) if len(null_sp_arr) else np.nan
        event_rows[-1]["spread_null_std"] = float(np.nanstd(null_sp_arr)) if len(null_sp_arr) else np.nan
        event_rows[-1]["spread_null_p"] = float((np.abs(null_sp_arr) >= abs(event_rows[-1]["spread_delta"])).mean()) if len(null_sp_arr) and pd.notna(event_rows[-1]["spread_delta"]) else np.nan
        event_rows[-1]["constitutional_null_mean"] = float(np.nanmean(null_const_arr)) if len(null_const_arr) else np.nan
        event_rows[-1]["constitutional_null_p"] = float((np.abs(null_const_arr) >= abs(event_rows[-1]["constitutional_share_delta"])).mean()) if len(null_const_arr) and pd.notna(event_rows[-1]["constitutional_share_delta"]) else np.nan
        event_rows[-1]["security_null_mean"] = float(np.nanmean(null_sec_arr)) if len(null_sec_arr) else np.nan
        event_rows[-1]["security_null_p"] = float((np.abs(null_sec_arr) >= abs(event_rows[-1]["security_share_delta"])).mean()) if len(null_sec_arr) and pd.notna(event_rows[-1]["security_share_delta"]) else np.nan
        null_rows.append(
            {
                "term": term,
                "event_label": event["label"],
                "event_slug": event_slug,
                "null_draws": len(null_sal_arr),
                "econ_salience_obs": event_rows[-1]["econ_salience_delta"],
                "econ_salience_null_mean": event_rows[-1]["econ_salience_null_mean"],
                "econ_salience_null_std": event_rows[-1]["econ_salience_null_std"],
                "econ_salience_null_p": event_rows[-1]["econ_salience_null_p"],
                "spread_obs": event_rows[-1]["spread_delta"],
                "spread_null_mean": event_rows[-1]["spread_null_mean"],
                "spread_null_std": event_rows[-1]["spread_null_std"],
                "spread_null_p": event_rows[-1]["spread_null_p"],
                "constitutional_obs": event_rows[-1]["constitutional_share_delta"],
                "constitutional_null_mean": event_rows[-1]["constitutional_null_mean"],
                "constitutional_null_p": event_rows[-1]["constitutional_null_p"],
                "security_obs": event_rows[-1]["security_share_delta"],
                "security_null_mean": event_rows[-1]["security_null_mean"],
                "security_null_p": event_rows[-1]["security_null_p"],
            }
        )

        brief_lines.extend(
            [
                f"{event['label']} ({effective_month})",
                f"- Macro pressure shift: {macro_shock - macro_pre:+.3f}" if pd.notna(macro_pre) and pd.notna(macro_shock) else "- Macro pressure shift: n/a",
                f"- Economy salience shift: {sal_shock - sal_pre:+.3f}" if pd.notna(sal_pre) and pd.notna(sal_shock) else "- Economy salience shift: n/a",
                f"- Economic spread shift: {spread_shock - spread_pre:+.3f}" if pd.notna(spread_pre) and pd.notna(spread_shock) else "- Economic spread shift: n/a",
                f"- Negative-edge share shift: {neg_share_shock - neg_share_pre:+.3f}" if pd.notna(neg_share_pre) and pd.notna(neg_share_shock) else "- Negative-edge share shift: n/a",
                f"- Signed modularity shift: {mod_shock - mod_pre:+.3f}" if pd.notna(mod_pre) and pd.notna(mod_shock) else "- Signed modularity shift: n/a",
                f"- Null p (economy salience / spread): {event_rows[-1]['econ_salience_null_p']:.3f} / {event_rows[-1]['spread_null_p']:.3f}" if pd.notna(event_rows[-1]["econ_salience_null_p"]) and pd.notna(event_rows[-1]["spread_null_p"]) else "- Null p (economy salience / spread): n/a",
                f"- Top attack party in shock month: {top_attack_party} ({top_attack_balance:+.3f})" if top_attack_party else "- Top attack party in shock month: n/a",
                f"- Top defense party in shock month: {top_defense_party} ({top_defense_balance:+.3f})" if top_defense_party else "- Top defense party in shock month: n/a",
                f"- Top owner of the economic agenda: {top_owner}" if top_owner else "- Top owner of the economic agenda: n/a",
                f"- Owner persistence into post window: {bool(top_owner == post_owner) if top_owner and post_owner else False}",
                "",
            ]
        )

    event_df = pd.DataFrame(event_rows)
    party_df = pd.DataFrame(party_rows)
    gate_df = pd.DataFrame(gate_rows)
    null_df = pd.DataFrame(null_rows)
    registry_df = pd.DataFrame(registry_rows)
    event_df.to_csv(csv_root / f"term_{term}_economic_event_summary.csv", index=False)
    party_df.to_csv(csv_root / f"term_{term}_economic_event_party_metrics.csv", index=False)
    gate_df.to_csv(csv_root / f"term_{term}_economic_event_gatekeepers.csv", index=False)
    null_df.to_csv(csv_root / f"term_{term}_economic_null_tests.csv", index=False)
    registry_df.to_csv(csv_root / f"term_{term}_economic_event_registry.csv", index=False)
    (note_root / f"term_{term}_economic_brief.txt").write_text("\n".join(brief_lines), encoding="utf-8")
    return event_df, party_df, gate_df, null_df, registry_df, figure_rows


def plot_global_interval_axis(yearly_pos: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    summary = (
        yearly_pos.groupby(["term", "party"], as_index=False)["position"]
        .agg(center="median", left="min", right="max")
    )
    fig, ax = plt.subplots(figsize=(16, 10), facecolor="white")
    term_order = sorted(summary["term"].unique())
    y_lookup = {term: idx for idx, term in enumerate(term_order[::-1])}
    for term in term_order:
        sub = summary[summary["term"] == term].sort_values("center")
        base_y = y_lookup[term]
        for row in sub.itertuples(index=False):
            color = PARTY_COLORS.get(row.party, "#64748b")
            ax.plot([row.left, row.right], [base_y, base_y], color=color, linewidth=2.6, alpha=0.75)
            ax.scatter([row.center], [base_y], color=color, s=70, edgecolor="white", linewidth=0.9, zorder=3)
        clusters: list[list[pd.Series]] = []
        for _, row in sub.iterrows():
            center = float(row["center"])
            if not clusters or abs(center - float(clusters[-1][-1]["center"])) > 0.20:
                clusters.append([row])
            else:
                clusters[-1].append(row)
        for cluster in clusters:
            y_offsets = centered_offsets(len(cluster), 0.12)
            x_offsets = centered_offsets(len(cluster), 0.045)
            for idx, row in enumerate(cluster):
                color = PARTY_COLORS.get(row["party"], "#64748b")
                ax.text(
                    float(row["center"]) + x_offsets[idx],
                    base_y + 0.18 + y_offsets[idx],
                    PARTY_SHORT.get(row["party"], row["party"]),
                    fontsize=8,
                    ha="center",
                    color=color,
                    fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.92},
                )
    ax.axvline(0.0, color="#cbd5e1", linestyle="--", linewidth=1.1)
    ax.set_yticks([y_lookup[t] for t in term_order])
    ax.set_yticklabels([f"Term {t}" for t in term_order])
    ax.set_xlabel("Common economy-specific latent axis")
    ax.set_title("Inter-term comparable economic intervals")
    ax.grid(alpha=0.15, axis="x")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_interterm_shock_comparison(event_df: pd.DataFrame, party_df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(18, 11), facecolor="white")
    gs = fig.add_gridspec(2, 2, wspace=0.20, hspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    colors = {"economic": "#991b1b", "economic_governance": "#d97706", "mixed": "#2563eb", "control": "#64748b"}
    for kind, sub in event_df.groupby("event_kind"):
        ax1.scatter(sub["macro_pressure_delta"], sub["econ_salience_delta"], s=120, alpha=0.85, color=colors.get(kind, "#475569"), label=kind.replace("_", " "))
        for row in sub.itertuples(index=False):
            ax1.text(row.macro_pressure_delta, row.econ_salience_delta, f"T{row.term}", fontsize=8, ha="left", va="bottom")
    ax1.axhline(0.0, color="#cbd5e1", linestyle=":")
    ax1.axvline(0.0, color="#cbd5e1", linestyle=":")
    ax1.set_title("Do stronger macro shocks pull parliament into economy?")
    ax1.set_xlabel("Macro pressure delta (shock - pre)")
    ax1.set_ylabel("Economy salience delta (shock - pre)")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.18)

    ax2 = fig.add_subplot(gs[0, 1])
    ordered = event_df.sort_values("spread_delta", ascending=False)
    ax2.barh(
        np.arange(len(ordered)),
        ordered["spread_delta"],
        color=[colors.get(kind, "#475569") for kind in ordered["event_kind"]],
        alpha=0.88,
    )
    ax2.axvline(0.0, color="#cbd5e1", linewidth=1.0)
    ax2.set_yticks(np.arange(len(ordered)))
    ax2.set_yticklabels([f"T{row.term} · {row.event_label}" for row in ordered.itertuples(index=False)])
    ax2.set_title("Which shocks widen economic distance the most?")
    ax2.set_xlabel("Economic spread delta")

    ax3 = fig.add_subplot(gs[1, 0])
    majors = [p for p in MAJOR_PARTIES if p in party_df["party"].unique()][:6]
    shock_only = party_df[party_df["stage"] == "shock"].copy()
    heat = shock_only.pivot_table(index=["term", "event_label"], columns="party", values="balance").reindex(columns=majors)
    im = ax3.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax3.set_yticks(np.arange(len(heat.index)))
    ax3.set_yticklabels([f"T{term} · {label}" for term, label in heat.index], fontsize=8)
    ax3.set_xticks(np.arange(len(majors)))
    ax3.set_xticklabels([PARTY_SHORT.get(p, p) for p in majors], rotation=35, ha="right")
    ax3.set_title("Shock-month economic attack / defense balance")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.03)
    cbar.set_label("Balance")

    ax4 = fig.add_subplot(gs[1, 1])
    tmp = event_df[["term", "event_label", "econ_salience_delta", "constitutional_share_delta", "security_share_delta"]].copy()
    x = np.arange(len(tmp))
    ax4.bar(x - 0.22, tmp["econ_salience_delta"], width=0.22, color="#991b1b", label="Economy")
    ax4.bar(x, tmp["constitutional_share_delta"], width=0.22, color="#1d4ed8", label="Constitution")
    ax4.bar(x + 0.22, tmp["security_share_delta"], width=0.22, color="#059669", label="Security")
    ax4.axhline(0.0, color="#cbd5e1", linewidth=1.0)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"T{row.term}" for row in tmp.itertuples(index=False)], rotation=0)
    ax4.set_title("Topic substitution around the shock")
    ax4.legend(frameon=False, fontsize=8)
    ax4.grid(alpha=0.15, axis="y")

    fig.suptitle("Economic-shock comparison across terms", fontsize=16, y=0.98)
    fig.text(
        0.01,
        0.01,
        "Panel 1 tests whether externally stronger macro pressure leads to more economy-centered parliamentary conflict. "
        "Panel 2 compares how much each shock widens the common economic latent spread. Panel 3 shows whether parties attack or defend the economic line in the shock month. "
        "Panel 4 asks whether shocks pull attention into economy or displace it toward constitutional / security conflict.",
        fontsize=9.8,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pressure_alignment(event_df: pd.DataFrame, party_df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(17, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(event_df["macro_pressure_shock"], event_df["spread_shock"], color="#991b1b", s=120, alpha=0.85)
    for row in event_df.itertuples(index=False):
        ax1.text(row.macro_pressure_shock, row.spread_shock, f"T{row.term}", fontsize=8, ha="left", va="bottom")
    ax1.set_title("Macro pressure vs economic spread")
    ax1.set_xlabel("Shock-month macro pressure")
    ax1.set_ylabel("Shock-month economic spread")
    ax1.grid(alpha=0.18)

    ax2 = fig.add_subplot(gs[0, 1])
    owner_counts = event_df["top_owner_party"].value_counts().head(8)
    ax2.barh(np.arange(len(owner_counts)), owner_counts.values, color=[PARTY_COLORS.get(p, "#64748b") for p in owner_counts.index])
    ax2.set_yticks(np.arange(len(owner_counts)))
    ax2.set_yticklabels([PARTY_SHORT.get(p, p) for p in owner_counts.index])
    ax2.set_title("Who most often owns the economic agenda?")
    ax2.set_xlabel("Number of shocks")

    ax3 = fig.add_subplot(gs[1, 0])
    attackers = (
        party_df[party_df["stage"] == "shock"]
        .groupby("party", as_index=False)["balance"]
        .mean()
        .sort_values("balance")
        .head(8)
    )
    ax3.barh(np.arange(len(attackers)), attackers["balance"], color=[PARTY_COLORS.get(p, "#64748b") for p in attackers["party"]])
    ax3.axvline(0.0, color="#cbd5e1", linewidth=1.0)
    ax3.set_yticks(np.arange(len(attackers)))
    ax3.set_yticklabels([PARTY_SHORT.get(p, p) for p in attackers["party"]])
    ax3.set_title("Parties with the most negative economic balance")
    ax3.set_xlabel("Average shock-month balance")

    ax4 = fig.add_subplot(gs[1, 1])
    role_tbl = event_df.groupby("event_kind", as_index=False).agg(
        avg_salience_delta=("econ_salience_delta", "mean"),
        avg_spread_delta=("spread_delta", "mean"),
        avg_constitutional_delta=("constitutional_share_delta", "mean"),
    )
    x = np.arange(len(role_tbl))
    ax4.bar(x - 0.22, role_tbl["avg_salience_delta"], width=0.22, color="#991b1b", label="Economy salience")
    ax4.bar(x, role_tbl["avg_spread_delta"], width=0.22, color="#2563eb", label="Spread")
    ax4.bar(x + 0.22, role_tbl["avg_constitutional_delta"], width=0.22, color="#059669", label="Constitutional shift")
    ax4.axhline(0.0, color="#cbd5e1", linewidth=1.0)
    ax4.set_xticks(x)
    ax4.set_xticklabels([k.replace("_", " ") for k in role_tbl["event_kind"]], rotation=15, ha="right")
    ax4.set_title("Economic shocks vs control shocks")
    ax4.legend(frameon=False, fontsize=8)

    fig.suptitle("Economic pressure alignment and agenda ownership", fontsize=16, y=0.98)
    fig.text(
        0.01,
        0.01,
        "These panels ask whether economic pressure really reorganizes parliamentary conflict instead of just increasing raw speech volume. "
        "The figure shows which parties repeatedly own the economy agenda, which actors most often turn economy into attack, and how economic shocks differ from constitutional / control shocks.",
        fontsize=9.8,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_event_design(registry_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    colors = {"economic": "#991b1b", "economic_governance": "#d97706", "mixed": "#2563eb", "control": "#64748b"}
    registry = registry_df.copy()
    registry["event_date"] = pd.to_datetime(registry["event_month_effective"] + "-01")
    terms = sorted(registry["term"].unique())
    y_map = {term: idx for idx, term in enumerate(terms[::-1])}

    for term, grp in registry.groupby("term"):
        grp = grp.sort_values("event_date").reset_index(drop=True)
        for rank, row in grp.iterrows():
            y = y_map[term]
            ax.scatter(row["event_date"], y, s=180, color=colors.get(row["event_kind"], "#475569"), edgecolor="white", linewidth=1.2, zorder=3)
            # Alternate labels above/below within each term to prevent collisions
            y_off, va = (0.16, "bottom") if rank % 2 == 0 else (-0.22, "top")
            ax.text(row["event_date"], y + y_off, row["event_label"], fontsize=8, ha="left", va=va)
            pre_months = [pd.Period(m, "M").to_timestamp() for m in row["pre_months"].split(",") if m]
            post_months = [pd.Period(m, "M").to_timestamp() for m in row["post_months"].split(",") if m]
            if pre_months and post_months:
                ax.plot([min(pre_months), max(post_months)], [y, y], color=colors.get(row["event_kind"], "#475569"), alpha=0.35, linewidth=3.0)

    ax.set_yticks([y_map[t] for t in terms])
    ax.set_yticklabels([f"Term {t}" for t in terms])
    ax.set_title("Event design registry: selected shock windows by term")
    ax.set_xlabel("Effective event month")
    ax.grid(axis="x", alpha=0.18)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_axis_validation(weight_df: pd.DataFrame, validation_df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(17, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, wspace=0.2, hspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ordered = weight_df.sort_values("weight")
    ax1.barh(np.arange(len(ordered)), ordered["weight"], color="#991b1b")
    ax1.errorbar(
        ordered["weight"],
        np.arange(len(ordered)),
        xerr=ordered["bootstrap_std"],
        fmt="none",
        ecolor="#111827",
        alpha=0.6,
        capsize=3,
    )
    ax1.set_yticks(np.arange(len(ordered)))
    ax1.set_yticklabels(ordered["label"])
    ax1.set_title("Economic axis weights and bootstrap spread")
    ax1.axvline(0.0, color="#cbd5e1", linewidth=1.0)

    ax2 = fig.add_subplot(gs[0, 1])
    boot = validation_df[validation_df["metric"] == "anchor_corr_bootstrap"]["value"].dropna().to_numpy(dtype=float)
    if len(boot):
        ax2.hist(boot, bins=24, color="#dbeafe", edgecolor="white")
    obs_corr = float(validation_df.loc[validation_df["metric"] == "observed_anchor_correlation", "value"].iloc[0])
    perm_p = float(validation_df.loc[validation_df["metric"] == "anchor_permutation_p", "value"].iloc[0])
    ax2.axvline(obs_corr, color="#991b1b", linewidth=2.2, linestyle="--")
    ax2.set_title("Bootstrap anchor correlation")
    ax2.text(0.03, 0.95, f"Observed corr = {obs_corr:.3f}\nPermutation p = {perm_p:.3f}", transform=ax2.transAxes, va="top", fontsize=9)

    ax3 = fig.add_subplot(gs[1, 0])
    loo = validation_df[validation_df["metric"] == "leave_one_anchor_out"].copy()
    if not loo.empty:
        ax3.barh(np.arange(len(loo)), loo["value"], color="#2563eb")
        ax3.set_yticks(np.arange(len(loo)))
        ax3.set_yticklabels([PARTY_SHORT.get(a, a) for a in loo["anchor"]])
    ax3.set_xlim(0.0, 1.02)
    ax3.set_title("Leave-one-anchor-out stability")
    ax3.set_xlabel("Correlation with full axis")

    ax4 = fig.add_subplot(gs[1, 1])
    spread = ordered[["label", "bootstrap_q10", "bootstrap_q90", "weight"]].copy()
    for idx, row in enumerate(spread.itertuples(index=False)):
        ax4.plot([row.bootstrap_q10, row.bootstrap_q90], [idx, idx], color="#475569", linewidth=3)
        ax4.scatter([row.weight], [idx], color="#991b1b", s=60, zorder=3)
    ax4.axvline(0.0, color="#cbd5e1", linewidth=1.0)
    ax4.set_yticks(np.arange(len(spread)))
    ax4.set_yticklabels(spread["label"])
    ax4.set_title("Bootstrap 10–90% weight intervals")
    ax4.set_xlabel("Weight range")

    fig.suptitle("Validation of the common economic latent axis", fontsize=16, y=0.98)
    fig.text(
        0.01,
        0.01,
        "The common axis is not a hand-written left/right scale. These panels show that topic weights are learned from the data, "
        "their direction is stable under bootstrap resampling, and the axis remains highly similar when one anchor party is removed.",
        fontsize=9.8,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_main_results_table(event_df: pd.DataFrame, gate_df: pd.DataFrame) -> pd.DataFrame:
    gate_pivot = gate_df.pivot_table(
        index=["term", "event_slug"],
        columns="stage",
        values=["party", "brokerage_concentration_hhi", "top_broker_party_agg", "top_party_mean_score"],
        aggfunc="first",
    ).reset_index()
    gate_pivot.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in gate_pivot.columns.to_flat_index()]
    merged = event_df.merge(gate_pivot, on=["term", "event_slug"], how="left")
    merged["speaker_gatekeeper_shift"] = merged["party_pre"].fillna("") != merged["party_shock"].fillna("")
    merged["party_gatekeeper_shift"] = merged["top_broker_party_agg_pre"].fillna("") != merged["top_broker_party_agg_shock"].fillna("")
    merged["gatekeeper_shift"] = merged["speaker_gatekeeper_shift"] | merged["party_gatekeeper_shift"]
    merged["top_broker_persistence"] = merged["top_broker_party_agg_shock"].fillna("") == merged["top_broker_party_agg_post"].fillna("")
    merged["eh1_supported"] = (
        merged["econ_salience_delta"].ge(0.10)
        & merged["event_kind"].isin(["economic", "economic_governance"])
    ) | (
        merged["econ_salience_delta"].gt(0.0)
        & merged["econ_salience_null_p"].le(0.10)
    )
    merged["eh2_supported"] = merged["owner_shift"].fillna(False)
    merged["eh3_supported"] = (
        ((merged["spread_null_p"] <= 0.10) & merged["spread_delta"].notna())
        | (merged["negative_edge_share_delta"].abs() >= 0.04)
        | (merged["signed_modularity_delta"].abs() >= 0.04)
    )
    merged["eh4_supported"] = merged["gatekeeper_shift"].fillna(False)
    merged["eh5_supported"] = (
        (
            (
                merged["displacement_index_shock"].gt(0.0)
                & merged["competing_gain_shock"].gt(0.0)
            )
            | merged["displacement_persists"].fillna(False)
        )
    )
    merged["support_count"] = merged[[c for c in merged.columns if c.startswith("eh") and c.endswith("_supported")]].sum(axis=1)
    merged["result_read"] = np.select(
        [
            merged["eh1_supported"] & merged["eh2_supported"] & merged["eh3_supported"],
            merged["eh5_supported"],
            merged["eh4_supported"],
        ],
        [
            "Ekonomi-merkezli güçlü yeniden örgütlenme",
            "Ekonomiden anayasa/güvenliğe kayış",
            "Tam yeniden örgütlenme olmadan broker değişimi",
        ],
        default="Kısmi ya da karma destek",
    )
    columns = [
        "term",
        "event_label",
        "event_kind",
        "paper_role",
        "macro_pressure_delta",
        "econ_salience_delta",
        "spread_delta",
        "negative_edge_share_delta",
        "signed_modularity_delta",
        "owner_concentration_hhi_shock",
        "owner_shift",
        "owner_persistence",
        "top_owner_party",
        "top_attack_party",
        "party_pre",
        "party_shock",
        "top_broker_party_agg_pre",
        "top_broker_party_agg_shock",
        "brokerage_concentration_hhi_shock",
        "top_broker_persistence",
        "displacement_index_shock",
        "displacement_index_post",
        "shock_to_post_persistence",
        "support_count",
        "eh1_supported",
        "eh2_supported",
        "eh3_supported",
        "eh4_supported",
        "eh5_supported",
        "result_read",
    ]
    return merged[columns].rename(
        columns={
            "party_pre": "pre_gatekeeper_party",
            "party_shock": "shock_gatekeeper_party",
            "top_broker_party_agg_pre": "pre_broker_party_agg",
            "top_broker_party_agg_shock": "shock_broker_party_agg",
        }
    ).sort_values(["paper_role", "support_count", "term"], ascending=[True, False, True])


def plot_main_results(results_df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(18, 10), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.15], wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    heat_cols = ["eh1_supported", "eh2_supported", "eh3_supported", "eh4_supported", "eh5_supported"]
    heat = results_df[heat_cols].astype(int).to_numpy(dtype=float)
    im = ax1.imshow(heat, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax1.set_yticks(np.arange(len(results_df)))
    ax1.set_yticklabels([f"T{row.term} · {row.event_label}" for row in results_df.itertuples(index=False)], fontsize=8)
    ax1.set_xticks(np.arange(len(heat_cols)))
    ax1.set_xticklabels(["EH1", "EH2", "EH3", "EH4", "EH5"])
    ax1.set_title("Hypothesis support matrix")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)

    ax2 = fig.add_subplot(gs[0, 1])
    ordered = results_df.sort_values("support_count", ascending=True)
    colors = {"economic": "#991b1b", "economic_governance": "#d97706", "mixed": "#2563eb", "control": "#64748b"}
    ax2.barh(
        np.arange(len(ordered)),
        ordered["support_count"],
        color=[colors.get(k, "#475569") for k in ordered["event_kind"]],
        alpha=0.9,
    )
    ax2.set_yticks(np.arange(len(ordered)))
    ax2.set_yticklabels([f"T{row.term} · {row.event_label}" for row in ordered.itertuples(index=False)], fontsize=8)
    ax2.set_xlabel("Number of supported hypotheses")
    ax2.set_title("Which events are most publication-ready?")
    for idx, row in enumerate(ordered.itertuples(index=False)):
        ax2.text(float(row.support_count) + 0.05, idx, row.result_read, va="center", fontsize=8, color="#374151")

    fig.suptitle("Main economic event results table", fontsize=16, y=0.98)
    fig.text(
        0.01,
        0.01,
        "The left panel turns the paper into an explicit hypothesis ledger: each event is checked against the five core claims. "
        "The right panel translates this into a publication reading by asking which events deliver the densest combination of agenda, distance, and brokerage evidence.",
        fontsize=9.8,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(event_df: pd.DataFrame, null_df: pd.DataFrame, validation_df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(18, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, wspace=0.38, hspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ord1 = event_df.sort_values("econ_salience_delta")
    ax1.errorbar(
        ord1["econ_salience_delta"],
        np.arange(len(ord1)),
        xerr=ord1["econ_salience_null_std"].fillna(0.0),
        fmt="o",
        color="#991b1b",
        ecolor="#cbd5e1",
        capsize=3,
    )
    ax1.axvline(0.0, color="#cbd5e1", linewidth=1.0)
    ax1.set_yticks(np.arange(len(ord1)))
    ax1.set_yticklabels([f"T{r.term} · {r.event_label}" for r in ord1.itertuples(index=False)], fontsize=8)
    ax1.set_title("Observed economy-salience shift vs null spread")

    ax2 = fig.add_subplot(gs[0, 1])
    ord2 = event_df.sort_values("spread_delta")
    ax2.errorbar(
        ord2["spread_delta"],
        np.arange(len(ord2)),
        xerr=ord2["spread_null_std"].fillna(0.0),
        fmt="o",
        color="#2563eb",
        ecolor="#cbd5e1",
        capsize=3,
    )
    ax2.axvline(0.0, color="#cbd5e1", linewidth=1.0)
    ax2.set_yticks(np.arange(len(ord2)))
    ax2.set_yticklabels([])
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("Observed spread shift vs null spread")

    ax3 = fig.add_subplot(gs[1, 0])
    pvals = null_df[["econ_salience_null_p", "spread_null_p", "constitutional_null_p", "security_null_p"]].mean().rename(
        {
            "econ_salience_null_p": "Economy salience",
            "spread_null_p": "Spread",
            "constitutional_null_p": "Constitution",
            "security_null_p": "Security",
        }
    )
    ax3.bar(np.arange(len(pvals)), pvals.values, color=["#991b1b", "#2563eb", "#059669", "#7c3aed"])
    ax3.axhline(0.10, color="#111827", linestyle="--", linewidth=1.2)
    ax3.set_xticks(np.arange(len(pvals)))
    ax3.set_xticklabels(pvals.index, rotation=15, ha="right")
    ax3.set_title("Average null-model p-values")

    ax4 = fig.add_subplot(gs[1, 1])
    loo = validation_df[validation_df["metric"] == "leave_one_anchor_out"].copy()
    if not loo.empty:
        ax4.barh(np.arange(len(loo)), loo["value"], color="#0f766e")
        ax4.set_yticks(np.arange(len(loo)))
        ax4.set_yticklabels([PARTY_SHORT.get(a, a) for a in loo["anchor"]])
    ax4.set_xlim(0.0, 1.02)
    ax4.set_title("Axis robustness when each anchor is removed")

    fig.suptitle("Robustness: null models and axis sensitivity", fontsize=16, y=0.98)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_interterm_note(event_df: pd.DataFrame, path: Path) -> None:
    core = event_df[event_df["paper_role"] == "core"].copy()
    if core.empty:
        path.write_text("No core economic events generated.", encoding="utf-8")
        return
    sal_series = core["econ_salience_delta"].abs()
    spread_series = core["spread_delta"].abs()
    pressure_series = core["macro_pressure_delta"].abs()
    strongest_salience = core.loc[sal_series.idxmax()] if sal_series.notna().any() else core.iloc[0]
    widest = core.loc[spread_series.idxmax()] if spread_series.notna().any() else core.iloc[0]
    strongest_pressure = core.loc[pressure_series.idxmax()] if pressure_series.notna().any() else core.iloc[0]
    owner_counts = core["top_owner_party"].value_counts()
    lines = [
        "Inter-term economic summary",
        "=" * 72,
        "",
        "Main take-away:",
        "Economic shocks do not only raise criticism. They reorganize who owns the economic agenda, how far parties move on a common economy-specific latent axis, and whether conflict stays in economy or gets displaced toward constitutional/security lanes.",
        "",
        f"Largest economy-salience shift: Term {int(strongest_salience['term'])} · {strongest_salience['event_label']} ({strongest_salience['econ_salience_delta']:+.3f})",
        f"Largest economic spread shift: Term {int(widest['term'])} · {widest['event_label']} ({widest['spread_delta']:+.3f})",
        f"Largest macro-pressure jump: Term {int(strongest_pressure['term'])} · {strongest_pressure['event_label']} ({strongest_pressure['macro_pressure_delta']:+.3f})",
        "",
        "Most frequent owners of the economic agenda:",
    ]
    for party, count in owner_counts.head(5).items():
        lines.append(f"- {party}: {int(count)} shock(s)")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    inter_csv = BASE / "Inter_Term" / "CSVs"
    inter_fig = BASE / "Inter_Term" / "Figures"
    inter_notes = BASE / "Inter_Term" / "Notes"
    inter_csv.mkdir(parents=True, exist_ok=True)
    inter_fig.mkdir(parents=True, exist_ok=True)
    inter_notes.mkdir(parents=True, exist_ok=True)

    yearly_pos, monthly_pos, weights_df, raw_mean, raw_std = build_global_axis()
    yearly_pos.to_csv(inter_csv / "inter_term_global_economic_positions.csv", index=False)
    weights_df.to_csv(inter_csv / "inter_term_global_economic_weights.csv", index=False)
    axis_validation_df, axis_weight_validation_df = build_axis_validation()
    axis_validation_df.to_csv(inter_csv / "inter_term_economic_axis_validation.csv", index=False)
    axis_weight_validation_df.to_csv(inter_csv / "inter_term_economic_axis_weight_validation.csv", index=False)

    macro_panel = load_monthly_macro()
    macro_panel["date"] = pd.to_datetime(macro_panel["month_key"] + "-01")

    all_events = []
    all_parties = []
    all_gates = []
    all_nulls = []
    all_registry = []
    all_figures = []
    for term in TERMS:
        event_df, party_df, gate_df, null_df, registry_df, fig_rows = compute_term_event_outputs(term, monthly_pos, macro_panel)
        if not event_df.empty:
            all_events.append(event_df)
        if not party_df.empty:
            all_parties.append(party_df)
        if not gate_df.empty:
            all_gates.append(gate_df)
        if not null_df.empty:
            all_nulls.append(null_df)
        if not registry_df.empty:
            all_registry.append(registry_df)
        all_figures.extend(fig_rows)

    event_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    party_df = pd.concat(all_parties, ignore_index=True) if all_parties else pd.DataFrame()
    gate_df = pd.concat(all_gates, ignore_index=True) if all_gates else pd.DataFrame()
    null_df = pd.concat(all_nulls, ignore_index=True) if all_nulls else pd.DataFrame()
    registry_df = pd.concat(all_registry, ignore_index=True) if all_registry else pd.DataFrame()
    fig_df = pd.DataFrame(all_figures)

    event_df.to_csv(inter_csv / "inter_term_economic_event_summary.csv", index=False)
    party_df.to_csv(inter_csv / "inter_term_economic_event_party_metrics.csv", index=False)
    gate_df.to_csv(inter_csv / "inter_term_economic_event_gatekeepers.csv", index=False)
    null_df.to_csv(inter_csv / "inter_term_economic_null_tests.csv", index=False)
    registry_df.to_csv(inter_csv / "inter_term_economic_event_registry.csv", index=False)
    fig_df.to_csv(inter_csv / "inter_term_economic_figure_index.csv", index=False)

    interval_summary = plot_global_interval_axis(yearly_pos, inter_fig / "inter_term_economic_interval_axis.png")
    interval_summary.to_csv(inter_csv / "inter_term_economic_interval_summary.csv", index=False)
    plot_axis_validation(axis_weight_validation_df, axis_validation_df, inter_fig / "inter_term_economic_axis_validation.png")
    if not registry_df.empty:
        plot_event_design(registry_df, inter_fig / "inter_term_economic_event_design.png")
    if not event_df.empty and not party_df.empty:
        plot_interterm_shock_comparison(event_df, party_df, inter_fig / "inter_term_economic_shock_comparison.png")
        plot_pressure_alignment(event_df, party_df, inter_fig / "inter_term_economic_pressure_alignment.png")
        results_df = build_main_results_table(event_df, gate_df)
        results_df.to_csv(inter_csv / "inter_term_economic_main_results.csv", index=False)
        plot_main_results(results_df, inter_fig / "inter_term_economic_main_results.png")
        if not null_df.empty:
            plot_robustness(event_df, null_df, axis_validation_df, inter_fig / "inter_term_economic_robustness.png")
    write_interterm_note(event_df, inter_notes / "inter_term_economic_summary.txt")


if __name__ == "__main__":
    main()
