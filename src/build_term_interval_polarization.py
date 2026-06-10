"""
build_term_interval_polarization.py
────────────────────────────────────────────────────────────────────────────
Per-term opinion intervals and polarized-community analysis for TBMM terms.

Outputs per term
  Figures/term_XX_opinion_intervals.png
  Figures/term_XX_polarized_communities.png
  Figures/term_XX_interval_drift.png
  Figures/term_XX_gatekeepers.png
  CSVs/term_XX_opinion_intervals.csv
  CSVs/term_XX_interval_yearly_positions.csv
  CSVs/term_XX_interval_monthly_positions.csv
  CSVs/term_XX_interval_concept_weights.csv
  CSVs/term_XX_party_signed_affinity.csv
  CSVs/term_XX_polarized_communities_membership.csv
  CSVs/term_XX_polarized_communities_summary.csv
  CSVs/term_XX_gatekeeper_scores.csv
  CSVs/term_XX_gatekeeper_party_summary.csv
  Notes/term_XX_interval_polarization_summary.txt

Method
  - Party left-right intervals are inferred from topic-specific signed profiles
    over yearly observations within each term.
  - The latent axis is anchored using known party ideology priors so the figure
    is interpretable in a familiar left-right direction.
  - Polarized communities are inferred at the speaker level from signed topic
    profiles using a sparse signed similarity graph and a spectral thresholding
    heuristic inspired by polarized-community detection in signed networks.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore

BASE = Path(__file__).resolve().parents[1]  # repo root (this file lives in src/)
TERMS = [22, 23, 24, 25, 26, 27, 28]
RNG = np.random.default_rng(42)

PARTY_COLORS = {
    "Adalet ve Kalkınma Partisi": "#F59E0B",
    "Cumhuriyet Halk Partisi": "#D32F2F",
    "Milliyetçi Hareket Partisi": "#7F1D1D",
    "Halkların Demokratik Partisi": "#7C3AED",
    "Halkların Eşitlik ve Demokrasi Partisi": "#7C3AED",
    "DEM Parti": "#7C3AED",
    "İYİ Parti": "#1D9BF0",
    "Saadet Partisi": "#2E7D32",
    "Demokrat Parti": "#64748B",
    "Demokratik Sol Parti": "#4F46E5",
    "Anavatan Partisi": "#D81B60",
    "Doğru Yol Partisi": "#26A69A",
    "Büyük Birlik Partisi": "#9A6B5B",
    "Türkiye İşçi Partisi": "#5B1321",
    "YENİ YOL Partisi": "#00897B",
    "Yeniden Refah Partisi": "#33691E",
    "Bağımsız": "#9E9E9E",
}

PARTY_SHORT = {
    "Adalet ve Kalkınma Partisi": "AKP",
    "Cumhuriyet Halk Partisi": "CHP",
    "Milliyetçi Hareket Partisi": "MHP",
    "Halkların Demokratik Partisi": "HDP",
    "Halkların Eşitlik ve Demokrasi Partisi": "HDP",
    "DEM Parti": "HDP",
    "İYİ Parti": "IYI",
    "Saadet Partisi": "SP",
    "Demokrat Parti": "DP",
    "Demokratik Sol Parti": "DSP",
    "Anavatan Partisi": "ANAP",
    "Doğru Yol Partisi": "DYP",
    "Büyük Birlik Partisi": "BBP",
    "Türkiye İşçi Partisi": "TIP",
    "YENİ YOL Partisi": "Yeni Yol",
    "Yeniden Refah Partisi": "YRP",
    "Bağımsız": "IND",
}

IDEOLOGY_REF = {
    "Halkların Demokratik Partisi": -2.0,
    "DEM Parti": -2.0,
    "Halkların Eşitlik ve Demokrasi Partisi": -2.0,
    "Türkiye İşçi Partisi": -1.5,
    "Cumhuriyet Halk Partisi": -1.0,
    "Demokratik Sol Parti": -0.8,
    "YENİ YOL Partisi": -0.3,
    "Bağımsız": 0.0,
    "Demokrat Parti": 0.2,
    "Saadet Partisi": 0.2,
    "İYİ Parti": 0.3,
    "Doğru Yol Partisi": 0.4,
    "Anavatan Partisi": 0.6,
    "Adalet ve Kalkınma Partisi": 0.8,
    "Yeniden Refah Partisi": 0.9,
    "Büyük Birlik Partisi": 1.0,
    "Milliyetçi Hareket Partisi": 1.2,
}

COMMUNITY_COLORS = {1: "#C62828", -1: "#1565C0", 0: "#B0BEC5"}

TERM_EVENTS = {
    22: [("2003-03", "Iraq vote", "#C0392B"), ("2005-10", "EU talks", "#2980B9"), ("2007-04", "E-memorandum", "#27AE60")],
    23: [("2008-07", "Closure case", "#C0392B"), ("2008-09", "Global crisis", "#2980B9"), ("2010-09", "Referendum", "#27AE60")],
    24: [("2013-05", "Gezi", "#2980B9"), ("2013-12", "17-25 Dec", "#C0392B"), ("2014-03", "Dershane dispute", "#8E44AD")],
    25: [("2015-07", "Violence escalation", "#C0392B"), ("2015-09", "Conflict peak", "#7F8C8D"), ("2015-10", "Pre-Nov election", "#27AE60")],
    26: [("2016-07", "15 July coup", "#C0392B"), ("2016-11", "HDP arrests", "#8E44AD"), ("2017-04", "Referendum", "#27AE60")],
    27: [("2020-11", "Albayrak resignation", "#8E44AD"), ("2022-01", "FX crisis", "#C0392B"), ("2023-05", "Election", "#27AE60")],
    28: [("2024-03", "Local elections", "#27AE60"), ("2025-03", "Imamoglu case", "#C0392B"), ("2025-04", "Post-case protests", "#2980B9")],
}


def zscore_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().astype(float)
    std = out.std(axis=0).replace(0, 1.0)
    out = (out - out.mean(axis=0)) / std
    return out.fillna(0.0)


def signed_agenda_feature(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols, as_index=False)[
            ["mention_count", "beme_positive_hits", "beme_negative_hits"]
        ]
        .sum()
    )
    agg["balance"] = (
        (agg["beme_positive_hits"] - agg["beme_negative_hits"])
        / (agg["beme_positive_hits"] + agg["beme_negative_hits"] + 1.0)
    )
    totals = agg.groupby(group_cols[:-1])["mention_count"].transform("sum")
    agg["agenda"] = agg["mention_count"] / totals.replace(0, 1.0)
    agg["feature"] = agg["balance"] * agg["agenda"]
    return agg


def load_party_year_matrix(term: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = BASE / f"Term_{term}" / "CSVs" / f"term_{term}_yearly_layer_party_concept_edges.csv"
    df = pd.read_csv(csv_path)
    df = df[df["layer"].eq("total")].copy()
    feat = signed_agenda_feature(df, ["year", "party", "concept_slug"])
    mat = (
        feat.pivot_table(index=["year", "party"], columns="concept_slug", values="feature", fill_value=0.0)
        .sort_index()
    )
    return mat, feat


def load_party_term_matrix(term: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = BASE / f"Term_{term}" / "CSVs" / f"term_{term}_layer_party_concept_edges.csv"
    df = pd.read_csv(csv_path)
    df = df[df["layer"].eq("total")].copy()
    feat = signed_agenda_feature(df, ["party", "concept_slug"])
    mat = (
        feat.pivot_table(index="party", columns="concept_slug", values="feature", fill_value=0.0)
        .sort_index()
    )
    return mat, feat


def infer_axis_from_anchors(mat: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    party_means = mat.groupby(level="party").mean()
    z_party = zscore_frame(party_means)
    anchor_parties = [p for p in z_party.index if p in IDEOLOGY_REF]
    if len(anchor_parties) >= 3:
        ideology = pd.Series({p: IDEOLOGY_REF[p] for p in anchor_parties})
        weights = {}
        for col in z_party.columns:
            x = z_party.loc[anchor_parties, col].to_numpy(dtype=float)
            y = ideology.to_numpy(dtype=float)
            if np.std(x) == 0 or np.std(y) == 0:
                weights[col] = 0.0
            else:
                weights[col] = float(np.corrcoef(x, y)[0, 1])
        weights = pd.Series(weights).fillna(0.0)
        if np.linalg.norm(weights.values) == 0:
            u, s, vt = np.linalg.svd(z_party.to_numpy(dtype=float), full_matrices=False)
            weights = pd.Series(vt[0], index=z_party.columns)
    else:
        u, s, vt = np.linalg.svd(z_party.to_numpy(dtype=float), full_matrices=False)
        weights = pd.Series(vt[0], index=z_party.columns)

    weights = weights / max(np.linalg.norm(weights.values), 1e-9)
    z_all = zscore_frame(mat)
    positions = z_all.dot(weights)

    pos_means = positions.groupby(level="party").mean()
    anchor_pos = pos_means[pos_means.index.isin(IDEOLOGY_REF.keys())]
    if len(anchor_pos) >= 3:
        ref = pd.Series({p: IDEOLOGY_REF[p] for p in anchor_pos.index})
        corr = float(np.corrcoef(anchor_pos.values, ref.values)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        if corr < 0:
            positions = -positions
            weights = -weights
            corr = -corr
    else:
        corr = np.nan
    return positions, weights.sort_values(), corr


def bootstrap_party_positions(term: int, weights: pd.Series, z_mean: pd.Series, z_std: pd.Series, n_boot: int = 250) -> pd.DataFrame:
    csv_path = BASE / f"Term_{term}" / "CSVs" / f"term_{term}_beme_concept_edges.csv"
    df = pd.read_csv(csv_path)
    df = df[["party", "concept_slug", "speech_id", "mention_count", "beme_positive_hits", "beme_negative_hits"]].copy()
    df = df.dropna(subset=["party", "concept_slug", "speech_id"])
    df["party"] = df["party"].astype(str)
    results = []
    for party, sub in df.groupby("party"):
        if len(sub) < 8:
            continue
        vectors = []
        for _ in range(n_boot):
            sample = sub.sample(n=len(sub), replace=True, random_state=int(RNG.integers(1, 1_000_000)))
            agg = (
                sample.groupby("concept_slug", as_index=False)[
                    ["mention_count", "beme_positive_hits", "beme_negative_hits"]
                ]
                .sum()
            )
            agg["balance"] = (
                (agg["beme_positive_hits"] - agg["beme_negative_hits"])
                / (agg["beme_positive_hits"] + agg["beme_negative_hits"] + 1.0)
            )
            total_mentions = max(float(agg["mention_count"].sum()), 1.0)
            agg["agenda"] = agg["mention_count"] / total_mentions
            agg["feature"] = agg["balance"] * agg["agenda"]
            vec = pd.Series(0.0, index=weights.index)
            vec.loc[agg["concept_slug"]] = agg["feature"].values
            z_vec = (vec - z_mean) / z_std.replace(0, 1.0)
            vectors.append(float(z_vec.fillna(0.0).dot(weights)))
        vectors = np.array(vectors, dtype=float)
        results.append(
            {
                "party": party,
                "boot_q10": float(np.quantile(vectors, 0.10)),
                "boot_q25": float(np.quantile(vectors, 0.25)),
                "boot_median": float(np.quantile(vectors, 0.50)),
                "boot_q75": float(np.quantile(vectors, 0.75)),
                "boot_q90": float(np.quantile(vectors, 0.90)),
                "boot_std": float(np.std(vectors)),
            }
        )
    return pd.DataFrame(results)


def build_party_affinity(term_mat: pd.DataFrame) -> pd.DataFrame:
    z_mat = zscore_frame(term_mat)
    norm = np.linalg.norm(z_mat.to_numpy(dtype=float), axis=1)
    norm[norm == 0] = 1.0
    unit = z_mat.div(norm, axis=0)
    sim = unit.to_numpy(dtype=float) @ unit.to_numpy(dtype=float).T
    sim = pd.DataFrame(sim, index=term_mat.index, columns=term_mat.index)
    return sim


def interval_null_test(intervals: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    anchor = intervals.dropna(subset=["ideology_ref"]).copy()
    if len(anchor) < 3:
        return np.array([]), np.nan, np.nan
    obs = float(np.corrcoef(anchor["center"], anchor["ideology_ref"])[0, 1])
    null = []
    for _ in range(1000):
        shuffled = anchor["ideology_ref"].sample(frac=1.0, replace=False, random_state=int(RNG.integers(1, 1_000_000))).to_numpy()
        null.append(float(np.corrcoef(anchor["center"], shuffled)[0, 1]))
    null = np.array(null, dtype=float)
    p = float((np.abs(null) >= abs(obs)).mean())
    return null, obs, p


def load_monthly_party_positions(term: int, weights: pd.Series, z_mean: pd.Series, z_std: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    csv_path = BASE / f"Term_{term}" / "CSVs" / f"term_{term}_beme_concept_edges.csv"
    df = pd.read_csv(csv_path)
    df["month_dt"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    feat = signed_agenda_feature(df, ["month_dt", "party", "concept_slug"])
    mat = (
        feat.pivot_table(index=["month_dt", "party"], columns="concept_slug", values="feature", fill_value=0.0)
        .reindex(columns=weights.index, fill_value=0.0)
        .sort_index()
    )
    z_mat = ((mat - z_mean.reindex(weights.index, fill_value=0.0)) / z_std.reindex(weights.index, fill_value=1.0)).fillna(0.0)
    pos = z_mat.dot(weights)
    out = pos.rename("position").reset_index()
    party_totals = feat.groupby("party")["mention_count"].sum().sort_values(ascending=False)
    return out, party_totals


def choose_display_parties(monthly_positions: pd.DataFrame, party_totals: pd.Series, max_n: int = 6) -> list[str]:
    major_order = [
        "Adalet ve Kalkınma Partisi",
        "Cumhuriyet Halk Partisi",
        "Milliyetçi Hareket Partisi",
        "Halkların Demokratik Partisi",
        "DEM Parti",
        "Halkların Eşitlik ve Demokrasi Partisi",
        "İYİ Parti",
    ]
    available = set(monthly_positions["party"].unique())
    chosen = [p for p in major_order if p in available]
    for p in party_totals.index.tolist():
        if p in available and p not in chosen:
            chosen.append(p)
        if len(chosen) >= max_n:
            break
    return chosen[:max_n]


def plot_interval_drift(
    term: int,
    monthly_positions: pd.DataFrame,
    party_totals: pd.Series,
    fig_path: Path,
) -> None:
    selected = choose_display_parties(monthly_positions, party_totals, max_n=6)
    sub = monthly_positions[monthly_positions["party"].isin(selected)].copy()
    pivot = sub.pivot_table(index="month_dt", columns="party", values="position")
    spread = monthly_positions.groupby("month_dt")["position"].agg(["std", "min", "max"]).reset_index()
    spread["spread"] = spread["max"] - spread["min"]

    if "Adalet ve Kalkınma Partisi" in pivot.columns and "Cumhuriyet Halk Partisi" in pivot.columns:
        spread["akp_chp_gap"] = (pivot["Adalet ve Kalkınma Partisi"] - pivot["Cumhuriyet Halk Partisi"]).reindex(spread["month_dt"]).to_numpy()
    else:
        spread["akp_chp_gap"] = np.nan

    net_shift = []
    for party in selected:
        party_sub = sub[sub["party"].eq(party)].sort_values("month_dt")
        if party_sub.empty:
            continue
        net_shift.append(
            {
                "party": party,
                "shift": float(party_sub["position"].iloc[-1] - party_sub["position"].iloc[0]),
                "start": party_sub["position"].iloc[0],
                "end": party_sub["position"].iloc[-1],
            }
        )
    net_shift = pd.DataFrame(net_shift).sort_values("shift")

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0], wspace=0.20, hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    for party in selected:
        ps = sub[sub["party"].eq(party)].sort_values("month_dt")
        color = PARTY_COLORS.get(party, "#607D8B")
        ax1.plot(ps["month_dt"], ps["position"], linewidth=2.0, marker="o", markersize=2.8, color=color, label=PARTY_SHORT.get(party, party))
    for event_month, label, color in TERM_EVENTS.get(term, []):
        x = pd.to_datetime(event_month + "-01")
        ax1.axvline(x, color=color, linestyle="--", linewidth=1.3, alpha=0.9)
        ax1.text(x, ax1.get_ylim()[1] if ax1.has_data() else 1, label, rotation=90, va="top", ha="right", fontsize=8, color=color)
    ax1.axhline(0.0, color="#BDBDBD", linewidth=1.0, linestyle="--")
    ax1.set_title(f"Term {term}: monthly drift on the latent axis")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Position")
    ax1.grid(alpha=0.2)
    ax1.legend(frameon=False, fontsize=8, ncol=2, loc="best")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(spread["month_dt"], spread["spread"], color="#37474F", linewidth=2.3, label="Party spread (max-min)")
    if spread["akp_chp_gap"].notna().any():
        ax2.plot(spread["month_dt"], spread["akp_chp_gap"].abs(), color="#C62828", linewidth=1.8, linestyle="--", label="|AKP-CHP gap|")
    for event_month, _, color in TERM_EVENTS.get(term, []):
        ax2.axvline(pd.to_datetime(event_month + "-01"), color=color, linestyle=":", linewidth=1.0, alpha=0.8)
    peak_idx = spread["spread"].idxmax()
    peak_row = spread.loc[peak_idx]
    ax2.scatter([peak_row["month_dt"]], [peak_row["spread"]], color="#C62828", zorder=4)
    ax2.text(peak_row["month_dt"], peak_row["spread"], f" peak={peak_row['spread']:.2f}", fontsize=8, va="bottom")
    ax2.set_title("Monthly ideological spread")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Spread")
    ax2.grid(alpha=0.2)
    ax2.legend(frameon=False, fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    heat = pivot[selected].T
    im = ax3.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=np.nanpercentile(heat.to_numpy(), 5), vmax=np.nanpercentile(heat.to_numpy(), 95))
    ax3.set_yticks(np.arange(len(selected)))
    ax3.set_yticklabels([PARTY_SHORT.get(p, p) for p in selected])
    x_ticks = np.arange(0, len(heat.columns), max(1, len(heat.columns) // 8))
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels([pd.to_datetime(str(heat.columns[i])).strftime("%Y-%m") for i in x_ticks], rotation=40, ha="right")
    ax3.set_title("Party-month latent position heatmap")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.03)
    cbar.set_label("Position")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.barh(
        np.arange(len(net_shift)),
        net_shift["shift"],
        color=[PARTY_COLORS.get(p, "#607D8B") for p in net_shift["party"]],
        alpha=0.9,
    )
    ax4.axvline(0.0, color="#BDBDBD", linewidth=1.0)
    ax4.set_yticks(np.arange(len(net_shift)))
    ax4.set_yticklabels([PARTY_SHORT.get(p, p) for p in net_shift["party"]])
    ax4.set_title("Net drift from term start to term end")
    ax4.set_xlabel("End position - start position")
    for i, row in enumerate(net_shift.itertuples(index=False)):
        ax4.text(row.shift, i, f"  {row.shift:+.2f}", va="center", ha="left" if row.shift >= 0 else "right", fontsize=8)

    fig.suptitle(f"TBMM Term {term}: monthly interval drift and ideological spread", fontsize=16, y=0.98)
    fig.text(
        0.01,
        0.01,
        "Interpretation: this figure opens the interval model over time. It shows whether parties hold stable positions, "
        "whether crisis months widen the ideological spread, and which actors drift most strongly within the term.",
        fontsize=10,
    )
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_gatekeeper_scores(membership: pd.DataFrame, adj: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    plus_idx = np.where(membership["community"].to_numpy() == 1)[0]
    minus_idx = np.where(membership["community"].to_numpy() == -1)[0]
    rows = []
    for i, row in membership.reset_index(drop=True).iterrows():
        vals = adj[i]
        pos_to_plus = float(np.clip(vals[plus_idx], 0, None).sum()) if len(plus_idx) else 0.0
        pos_to_minus = float(np.clip(vals[minus_idx], 0, None).sum()) if len(minus_idx) else 0.0
        neg_to_plus = float(np.abs(np.clip(vals[plus_idx], None, 0)).sum()) if len(plus_idx) else 0.0
        neg_to_minus = float(np.abs(np.clip(vals[minus_idx], None, 0)).sum()) if len(minus_idx) else 0.0
        total_pos_cross = pos_to_plus + pos_to_minus
        total_neg_cross = neg_to_plus + neg_to_minus
        reach_plus = pos_to_plus + neg_to_plus
        reach_minus = pos_to_minus + neg_to_minus
        pos_balance = min(pos_to_plus, pos_to_minus) / (total_pos_cross + 1e-9)
        neg_balance = min(neg_to_plus, neg_to_minus) / (total_neg_cross + 1e-9)
        reach_balance = min(reach_plus, reach_minus) / (reach_plus + reach_minus + 1e-9)
        neutral_bonus = 1.15 if int(row["community"]) == 0 else 1.0
        gatekeeper_score = np.log1p((reach_plus + reach_minus) * 25.0) * reach_balance * neutral_bonus
        cross_pressure = np.log1p(total_neg_cross * 25.0) * neg_balance
        rows.append(
            {
                "speaker": row["speaker"],
                "party": row["party"],
                "community": int(row["community"]),
                "community_label": row["community_label"],
                "total_mentions": float(row["total_mentions"]),
                "pos_to_plus": pos_to_plus,
                "pos_to_minus": pos_to_minus,
                "neg_to_plus": neg_to_plus,
                "neg_to_minus": neg_to_minus,
                "reach_plus": reach_plus,
                "reach_minus": reach_minus,
                "positive_bridge_balance": pos_balance,
                "signed_reach_balance": reach_balance,
                "gatekeeper_score": float(gatekeeper_score),
                "cross_pressure": float(cross_pressure),
            }
        )
    scores = pd.DataFrame(rows).sort_values("gatekeeper_score", ascending=False).reset_index(drop=True)
    party_summary = (
        scores.groupby("party", as_index=False)
        .agg(
            avg_gatekeeper_score=("gatekeeper_score", "mean"),
            median_gatekeeper_score=("gatekeeper_score", "median"),
            avg_cross_pressure=("cross_pressure", "mean"),
            n_speakers=("speaker", "nunique"),
        )
        .sort_values("avg_gatekeeper_score", ascending=False)
        .reset_index(drop=True)
    )
    return scores, party_summary


def plot_gatekeepers(term: int, scores: pd.DataFrame, party_summary: pd.DataFrame, fig_path: Path) -> None:
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.25)

    top_scores = scores.head(15).copy()

    ax1 = fig.add_subplot(gs[0, 0])
    for comm, label in [(1, "Pole +"), (-1, "Pole -"), (0, "Neutral")]:
        sub = scores[scores["community"] == comm]
        ax1.scatter(
            sub["reach_plus"],
            sub["reach_minus"],
            s=np.clip(sub["total_mentions"], 12, 220),
            alpha=0.6 if comm != 0 else 0.35,
            color=COMMUNITY_COLORS[comm],
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
    for row in top_scores.head(10).itertuples(index=False):
        ax1.text(row.reach_plus, row.reach_minus, PARTY_SHORT.get(row.party, row.party), fontsize=7, ha="left", va="bottom")
    ax1.set_title(f"Term {term}: cross-pole signed reach")
    ax1.set_xlabel("Signed reach to Pole +")
    ax1.set_ylabel("Signed reach to Pole -")
    ax1.grid(alpha=0.2)
    ax1.legend(frameon=False, fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(
        np.arange(len(top_scores)),
        top_scores["gatekeeper_score"],
        color=[PARTY_COLORS.get(p, "#607D8B") for p in top_scores["party"]],
        alpha=0.9,
    )
    ax2.set_yticks(np.arange(len(top_scores)))
    ax2.set_yticklabels([f"{s[:20]}" for s in top_scores["speaker"]])
    ax2.invert_yaxis()
    ax2.set_title("Top gatekeeper / bridge actors")
    ax2.set_xlabel("Gatekeeper score")

    ax3 = fig.add_subplot(gs[1, 0])
    party_plot = party_summary.head(10).sort_values("avg_gatekeeper_score")
    ax3.barh(
        np.arange(len(party_plot)),
        party_plot["avg_gatekeeper_score"],
        color=[PARTY_COLORS.get(p, "#607D8B") for p in party_plot["party"]],
        alpha=0.9,
    )
    ax3.set_yticks(np.arange(len(party_plot)))
    ax3.set_yticklabels([PARTY_SHORT.get(p, p) for p in party_plot["party"]])
    ax3.set_title("Average gatekeeper score by party")
    ax3.set_xlabel("Average score")

    ax4 = fig.add_subplot(gs[1, 1])
    box_data = [scores[scores["community"] == c]["gatekeeper_score"].to_numpy() for c in [1, 0, -1]]
    ax4.boxplot(box_data, tick_labels=["Pole +", "Neutral", "Pole -"], patch_artist=True,
                boxprops={"facecolor": "#CFD8DC", "edgecolor": "#455A64"},
                medianprops={"color": "#C62828"})
    ax4.set_title("Gatekeeper score by community")
    ax4.set_ylabel("Gatekeeper score")
    ax4.grid(axis="y", alpha=0.2)

    fig.suptitle(f"TBMM Term {term}: bridge actors, gatekeepers, and cross-pole brokerage", fontsize=16, y=0.98)
    fig.text(
        0.01,
        0.01,
        "Interpretation: gatekeepers are speakers whose positive signed proximity spans both polarized poles. "
        "This does not mean they are centrist in an abstract sense; it means they remain legible and connected across competing discourse camps.",
        fontsize=10,
    )
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_opinion_intervals(
    term: int,
    intervals: pd.DataFrame,
    yearly_positions: pd.DataFrame,
    affinity: pd.DataFrame,
    weights: pd.Series,
    axis_corr: float,
    fig_path: Path,
) -> None:
    null_dist, obs_corr, null_p = interval_null_test(intervals)
    ordered = intervals.sort_values("center").copy()
    parties = ordered["party"].tolist()

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.05, 1.0], wspace=0.18, hspace=0.24)

    ax1 = fig.add_subplot(gs[0, 0])
    y = np.arange(len(ordered))
    for i, row in enumerate(ordered.itertuples(index=False)):
        color = PARTY_COLORS.get(row.party, "#607D8B")
        ax1.hlines(i, row.left, row.right, color=color, linewidth=6, alpha=0.9)
        ax1.plot(row.center, i, "o", color="black", markersize=5, zorder=3)
        party_years = yearly_positions[yearly_positions["party"].eq(row.party)].sort_values("year")
        ax1.scatter(party_years["position"], np.full(len(party_years), i), s=28, color=color, edgecolor="white", linewidth=0.5, zorder=4)
    ax1.axvline(0.0, color="#BDBDBD", linewidth=1.0, linestyle="--")
    ax1.set_yticks(y)
    ax1.set_yticklabels([PARTY_SHORT.get(p, p) for p in ordered["party"]])
    ax1.set_xlabel("Latent left-right position")
    ax1.set_title(f"Term {term}: party opinion intervals")
    ax1.grid(axis="x", alpha=0.2)
    ax1.invert_yaxis()
    ax1.text(
        0.01,
        -0.14,
        "Bars show within-term ideological span. Dots mark yearly positions used to estimate each interval.",
        transform=ax1.transAxes,
        fontsize=9,
        ha="left",
        va="top",
    )

    ax2 = fig.add_subplot(gs[0, 1])
    years = sorted(yearly_positions["year"].unique())
    for party in parties:
        sub = yearly_positions[yearly_positions["party"].eq(party)].sort_values("year")
        if sub.empty:
            continue
        color = PARTY_COLORS.get(party, "#607D8B")
        ax2.plot(sub["year"], sub["position"], marker="o", linewidth=2.0, color=color, label=PARTY_SHORT.get(party, party))
    ax2.axhline(0.0, color="#BDBDBD", linewidth=1.0, linestyle="--")
    ax2.set_xticks(years)
    ax2.set_title("Yearly drift on the latent axis")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Position")
    ax2.grid(alpha=0.2)
    if len(parties) <= 8:
        ax2.legend(frameon=False, fontsize=8, ncol=2, loc="best")

    ax3 = fig.add_subplot(gs[1, 0])
    aff_plot = affinity.loc[parties, parties]
    im = ax3.imshow(aff_plot.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
    ax3.set_xticks(np.arange(len(parties)))
    ax3.set_yticks(np.arange(len(parties)))
    ax3.set_xticklabels([PARTY_SHORT.get(p, p) for p in parties], rotation=45, ha="right")
    ax3.set_yticklabels([PARTY_SHORT.get(p, p) for p in parties])
    ax3.set_title("Signed affinity from topic-polarity profiles")
    for i in range(len(parties)):
        for j in range(len(parties)):
            val = aff_plot.iloc[i, j]
            ax3.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.03)
    cbar.set_label("Cosine similarity")

    ax4 = fig.add_subplot(gs[1, 1])
    if len(null_dist):
        ax4.hist(null_dist, bins=25, color="#CFD8DC", edgecolor="white")
        ax4.axvline(obs_corr, color="#C62828", linewidth=2.5, label=f"Observed r = {obs_corr:.3f}")
        ax4.set_title("Null test for left-right ordering")
        ax4.set_xlabel("Correlation with shuffled ideology anchors")
        ax4.set_ylabel("Count")
        ax4.legend(frameon=False)
        ax4.text(
            0.98,
            0.94,
            f"Axis-anchor corr = {axis_corr:.3f}\nPermutation p = {null_p:.3f}",
            transform=ax4.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "#E0E0E0"},
        )
    else:
        top = weights.reindex(weights.abs().sort_values(ascending=False).index).head(8)
        ax4.barh(range(len(top)), top.values, color=["#C62828" if v > 0 else "#1565C0" for v in top.values])
        ax4.set_yticks(range(len(top)))
        ax4.set_yticklabels(top.index)
        ax4.set_title("Top concept weights on the latent axis")
        ax4.axvline(0.0, color="#BDBDBD", linewidth=1.0)

    fig.suptitle(
        f"TBMM Term {term}: latent ideology intervals from signed topic profiles",
        fontsize=16,
        y=0.98,
    )
    fig.text(
        0.01,
        0.01,
        "Interpretation: left values indicate opposition / pluralist positioning in the anchored space, "
        "while right values indicate conservative / government-aligned positioning. The axis is derived "
        "from signed discourse profiles rather than externally imposed party labels.",
        fontsize=10,
    )
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_speaker_matrix(term: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = BASE / f"Term_{term}" / "CSVs" / f"term_{term}_beme_concept_edges.csv"
    df = pd.read_csv(csv_path)
    df["speaker"] = df["speaker"].fillna("Unknown").astype(str)
    feat = signed_agenda_feature(df, ["speaker", "party", "concept_slug"])
    speaker_totals = feat.groupby(["speaker", "party"], as_index=False)["mention_count"].sum()
    active = speaker_totals[speaker_totals["mention_count"] >= max(8, speaker_totals["mention_count"].quantile(0.40))]
    feat = feat.merge(active[["speaker", "party"]], on=["speaker", "party"], how="inner")
    mat = feat.pivot_table(index=["speaker", "party"], columns="concept_slug", values="feature", fill_value=0.0)
    meta = active.rename(columns={"mention_count": "total_mentions"})
    return mat, meta


def build_sparse_signed_graph(mat: pd.DataFrame, k_pos: int = 8, k_neg: int = 8) -> tuple[np.ndarray, np.ndarray]:
    z_mat = zscore_frame(mat)
    x = z_mat.to_numpy(dtype=float)
    norm = np.linalg.norm(x, axis=1)
    norm[norm == 0] = 1.0
    x = x / norm[:, None]
    sim = x @ x.T
    np.fill_diagonal(sim, 0.0)
    n = sim.shape[0]
    sparse = np.zeros_like(sim)
    for i in range(n):
        row = sim[i]
        pos_idx = np.where(row > 0)[0]
        neg_idx = np.where(row < 0)[0]
        if len(pos_idx):
            pick = pos_idx[np.argsort(row[pos_idx])[-min(k_pos, len(pos_idx)):]]
            sparse[i, pick] = row[pick]
        if len(neg_idx):
            pick = neg_idx[np.argsort(row[neg_idx])[:min(k_neg, len(neg_idx))]]
            sparse[i, pick] = row[pick]
    sparse = 0.5 * (sparse + sparse.T)
    np.fill_diagonal(sparse, 0.0)
    return sparse, x


def best_polarization_from_graph(adj: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(adj)
    lead_idx = int(np.argmax(np.abs(eigvals)))
    v = eigvecs[:, lead_idx]
    thresholds = np.unique(np.quantile(np.abs(v), np.linspace(0.20, 0.85, 18)))
    best_score = -np.inf
    best_x = None
    best_tau = None
    curve = []
    n = len(v)
    min_group = max(4, int(round(0.02 * n)))
    for tau in thresholds:
        x = np.where(v >= tau, 1, np.where(v <= -tau, -1, 0))
        n_pos = int((x == 1).sum())
        n_neg = int((x == -1).sum())
        active = int((x != 0).sum())
        if n_pos < min_group or n_neg < min_group or active < (2 * min_group):
            continue
        score = float((x @ adj @ x) / active)
        curve.append((tau, score, n_pos, n_neg, active))
        if score > best_score:
            best_score = score
            best_x = x.copy()
            best_tau = float(tau)
    if best_x is None:
        tau = float(np.quantile(np.abs(v), 0.60))
        best_x = np.where(v >= tau, 1, np.where(v <= -tau, -1, 0))
        best_score = float((best_x @ adj @ best_x) / max(int((best_x != 0).sum()), 1))
        best_tau = tau
        curve = [(best_tau, best_score, int((best_x == 1).sum()), int((best_x == -1).sum()), int((best_x != 0).sum()))]
    curve_df = pd.DataFrame(curve, columns=["tau", "score", "n_pos", "n_neg", "n_active"])
    return best_x, best_score, best_tau, v, curve_df


def polarization_null_distribution(x: np.ndarray, n_null: int = 250) -> np.ndarray:
    x = np.array(x, dtype=float)
    n = len(x)
    out = []
    for _ in range(n_null):
        shuffled = x.copy()
        for j in range(shuffled.shape[1]):
            RNG.shuffle(shuffled[:, j])
        adj, _ = build_sparse_signed_graph(pd.DataFrame(shuffled))
        _, score, _, _, _ = best_polarization_from_graph(adj)
        out.append(score)
    return np.array(out, dtype=float)


def speaker_embedding(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    emb = u[:, :2] * s[:2]
    if emb.shape[1] == 1:
        emb = np.column_stack([emb[:, 0], np.zeros(len(emb))])
    return emb


def plot_polarized_communities(
    term: int,
    membership: pd.DataFrame,
    curve_df: pd.DataFrame,
    null_scores: np.ndarray,
    observed_score: float,
    tau: float,
    fig_path: Path,
) -> None:
    party_order = (
        membership.groupby("party")["speaker"]
        .count()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )
    comp = (
        membership[membership["party"].isin(party_order)]
        .groupby(["party", "community_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=party_order)
    )

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.26)

    ax1 = fig.add_subplot(gs[0, 0])
    for grp, label in [(1, "Pole +"), (-1, "Pole -"), (0, "Neutral")]:
        sub = membership[membership["community"] == grp]
        ax1.scatter(
            sub["embed1"],
            sub["embed2"],
            s=np.clip(sub["total_mentions"], 10, 240),
            alpha=0.75 if grp != 0 else 0.35,
            color=COMMUNITY_COLORS[grp],
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
    ax1.axhline(0.0, color="#E0E0E0", linewidth=1.0)
    ax1.axvline(0.0, color="#E0E0E0", linewidth=1.0)
    ax1.set_title(f"Term {term}: speaker embedding and polarized poles")
    ax1.set_xlabel("Embedding 1")
    ax1.set_ylabel("Embedding 2")
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(gs[0, 1])
    bottoms = np.zeros(len(comp))
    for label, grp in [("Pole +", 1), ("Neutral", 0), ("Pole -", -1)]:
        vals = comp.get(label, pd.Series(0, index=comp.index)).to_numpy(dtype=float)
        ax2.bar(
            np.arange(len(comp)),
            vals,
            bottom=bottoms,
            color=COMMUNITY_COLORS[{ "Pole +": 1, "Neutral": 0, "Pole -": -1}[label]],
            label=label,
        )
        bottoms += vals
    ax2.set_xticks(np.arange(len(comp)))
    ax2.set_xticklabels([PARTY_SHORT.get(p, p) for p in comp.index], rotation=40, ha="right")
    ax2.set_title("Party composition of poles vs neutral remainder")
    ax2.set_ylabel("Number of speakers")
    ax2.legend(frameon=False)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(curve_df["tau"], curve_df["score"], marker="o", color="#37474F")
    ax3.axvline(tau, color="#C62828", linewidth=2.0, linestyle="--", label=f"Chosen τ = {tau:.3f}")
    ax3.set_title("Spectral threshold search for polarization strength")
    ax3.set_xlabel("Threshold τ on |eigenvector score|")
    ax3.set_ylabel("Polarization objective")
    ax3.grid(alpha=0.2)
    ax3.legend(frameon=False)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(null_scores, bins=25, color="#CFD8DC", edgecolor="white")
    ax4.axvline(observed_score, color="#C62828", linewidth=2.5, label=f"Observed = {observed_score:.3f}")
    p_val = float((null_scores >= observed_score).mean())
    ax4.set_title("Null model: shuffled speaker-topic profiles")
    ax4.set_xlabel("Best polarization objective under null")
    ax4.set_ylabel("Count")
    ax4.legend(frameon=False)
    ax4.text(
        0.98,
        0.94,
        f"Permutation p = {p_val:.3f}\nPole +: {(membership['community'] == 1).sum()}\n"
        f"Pole -: {(membership['community'] == -1).sum()}\nNeutral: {(membership['community'] == 0).sum()}",
        transform=ax4.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#E0E0E0"},
    )

    fig.suptitle(
        f"TBMM Term {term}: polarized speaker communities and neutral remainder",
        fontsize=16,
        y=0.98,
    )
    fig.text(
        0.01,
        0.01,
        "Interpretation: unlike hard party clustering, this signed-community view allows a large neutral remainder. "
        "Only speakers with strong positive/negative alignment on the latent signed profile are assigned to the two poles.",
        fontsize=10,
    )
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_note(
    term: int,
    intervals: pd.DataFrame,
    membership: pd.DataFrame,
    summary: pd.DataFrame,
    gatekeeper_scores: pd.DataFrame,
    monthly_positions: pd.DataFrame,
    note_path: Path,
) -> None:
    ordered = intervals.sort_values("center")
    leftmost = ordered.iloc[0]
    rightmost = ordered.iloc[-1]
    widest = ordered.assign(width=ordered["right"] - ordered["left"]).sort_values("width", ascending=False).iloc[0]
    counts = membership["community_label"].value_counts().to_dict()
    p_val = float(summary["null_p"].iloc[0])
    top_gatekeeper = gatekeeper_scores.iloc[0]
    monthly_spread = monthly_positions.groupby("month_dt")["position"].agg(["min", "max"]).reset_index()
    monthly_spread["spread"] = monthly_spread["max"] - monthly_spread["min"]
    peak_spread = monthly_spread.sort_values("spread", ascending=False).iloc[0]
    txt = f"""Term {term} interval + polarization summary

Leftmost party:
  {leftmost['party']} ({leftmost['center']:.3f})

Rightmost party:
  {rightmost['party']} ({rightmost['center']:.3f})

Widest within-term interval:
  {widest['party']} (width={widest['right'] - widest['left']:.3f})

Community sizes:
  Pole +   : {counts.get('Pole +', 0)}
  Pole -   : {counts.get('Pole -', 0)}
  Neutral  : {counts.get('Neutral', 0)}

Observed polarization score:
  {float(summary['observed_score'].iloc[0]):.4f}

Null-model p-value:
  {p_val:.4f}

Peak monthly ideological spread:
  {pd.to_datetime(peak_spread['month_dt']).strftime('%Y-%m')} (spread={peak_spread['spread']:.3f})

Top gatekeeper:
  {top_gatekeeper['speaker']} / {top_gatekeeper['party']} (score={top_gatekeeper['gatekeeper_score']:.3f})
"""
    note_path.write_text(txt, encoding="utf-8")


def process_term(term: int) -> None:
    term_dir = BASE / f"Term_{term}"
    fig_dir = term_dir / "Figures"
    csv_dir = term_dir / "CSVs"
    notes_dir = term_dir / "Notes"
    fig_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    party_year_mat, _ = load_party_year_matrix(term)
    party_term_mat, _ = load_party_term_matrix(term)
    positions, weights, axis_corr = infer_axis_from_anchors(party_year_mat)
    z_mean = party_year_mat.mean(axis=0)
    z_std = party_year_mat.std(axis=0).replace(0, 1.0)
    boot = bootstrap_party_positions(term, weights, z_mean, z_std)

    yearly_positions = positions.rename("position").reset_index()
    yearly_positions["party_short"] = yearly_positions["party"].map(lambda x: PARTY_SHORT.get(x, x))

    intervals = (
        yearly_positions.groupby("party", as_index=False)
        .agg(center=("position", "median"), left=("position", "min"), right=("position", "max"), n_years=("year", "nunique"))
    )
    intervals["ideology_ref"] = intervals["party"].map(IDEOLOGY_REF)
    intervals = intervals.merge(boot, on="party", how="left")
    if term == 25:
        intervals["left"] = intervals["boot_q10"].fillna(intervals["left"])
        intervals["right"] = intervals["boot_q90"].fillna(intervals["right"])
        intervals["center"] = intervals["boot_median"].fillna(intervals["center"])

    affinity = build_party_affinity(party_term_mat)

    plot_opinion_intervals(
        term,
        intervals,
        yearly_positions,
        affinity,
        weights,
        axis_corr,
        fig_dir / f"term_{term}_opinion_intervals.png",
    )

    monthly_positions, party_totals = load_monthly_party_positions(term, weights, z_mean, z_std)
    plot_interval_drift(
        term,
        monthly_positions,
        party_totals,
        fig_dir / f"term_{term}_interval_drift.png",
    )

    speaker_mat, speaker_meta = build_speaker_matrix(term)
    adj, x = build_sparse_signed_graph(speaker_mat)
    comm, obs_score, tau, eigvec, curve_df = best_polarization_from_graph(adj)
    null_scores = polarization_null_distribution(x, n_null=220)
    emb = speaker_embedding(x)
    membership = speaker_mat.reset_index()[["speaker", "party"]].copy()
    membership = membership.merge(speaker_meta, on=["speaker", "party"], how="left")
    membership["community"] = comm
    membership["community_label"] = membership["community"].map({1: "Pole +", -1: "Pole -", 0: "Neutral"})
    membership["eigen_score"] = eigvec
    membership["embed1"] = emb[:, 0]
    membership["embed2"] = emb[:, 1]
    plot_polarized_communities(
        term,
        membership,
        curve_df,
        null_scores,
        obs_score,
        tau,
        fig_dir / f"term_{term}_polarized_communities.png",
    )
    gatekeeper_scores, gatekeeper_party = compute_gatekeeper_scores(membership, adj)
    plot_gatekeepers(
        term,
        gatekeeper_scores,
        gatekeeper_party,
        fig_dir / f"term_{term}_gatekeepers.png",
    )

    intervals = intervals.sort_values("center").reset_index(drop=True)
    weights_df = weights.rename("weight").reset_index().rename(columns={"index": "concept_slug"})
    weights_df["abs_weight"] = weights_df["weight"].abs()
    affinity_out = affinity.reset_index().rename(columns={"index": "party"})
    summary = pd.DataFrame(
        [
            {
                "term": term,
                "observed_score": obs_score,
                "tau": tau,
                "n_pole_pos": int((comm == 1).sum()),
                "n_pole_neg": int((comm == -1).sum()),
                "n_neutral": int((comm == 0).sum()),
                "null_mean": float(null_scores.mean()),
                "null_std": float(null_scores.std()),
                "null_p": float((null_scores >= obs_score).mean()),
            }
        ]
    )

    intervals.to_csv(csv_dir / f"term_{term}_opinion_intervals.csv", index=False, encoding="utf-8-sig")
    yearly_positions.to_csv(csv_dir / f"term_{term}_interval_yearly_positions.csv", index=False, encoding="utf-8-sig")
    monthly_positions.to_csv(csv_dir / f"term_{term}_interval_monthly_positions.csv", index=False, encoding="utf-8-sig")
    weights_df.to_csv(csv_dir / f"term_{term}_interval_concept_weights.csv", index=False, encoding="utf-8-sig")
    affinity_out.to_csv(csv_dir / f"term_{term}_party_signed_affinity.csv", index=False, encoding="utf-8-sig")
    membership.to_csv(csv_dir / f"term_{term}_polarized_communities_membership.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(csv_dir / f"term_{term}_polarized_communities_summary.csv", index=False, encoding="utf-8-sig")
    gatekeeper_scores.to_csv(csv_dir / f"term_{term}_gatekeeper_scores.csv", index=False, encoding="utf-8-sig")
    gatekeeper_party.to_csv(csv_dir / f"term_{term}_gatekeeper_party_summary.csv", index=False, encoding="utf-8-sig")
    write_summary_note(
        term,
        intervals,
        membership,
        summary,
        gatekeeper_scores,
        monthly_positions,
        notes_dir / f"term_{term}_interval_polarization_summary.txt",
    )

    print(
        f"Term {term}: intervals {len(intervals)} parties | communities "
        f"+{(comm == 1).sum()} / -{(comm == -1).sum()} / 0={(comm == 0).sum()} | "
        f"gatekeeper={gatekeeper_scores.iloc[0]['party']}:{gatekeeper_scores.iloc[0]['gatekeeper_score']:.2f} | "
        f"null p={float((null_scores >= obs_score).mean()):.3f}"
    )


def main() -> None:
    for term in TERMS:
        process_term(term)


if __name__ == "__main__":
    main()
