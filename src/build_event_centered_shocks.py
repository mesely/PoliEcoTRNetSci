"""
build_event_centered_shocks.py
────────────────────────────────────────────────────────────────────────────
Build event-centered pre/post shock summaries around major political shocks
for Terms 22–28, focused on interval drift and gatekeeper dynamics.

Outputs per term
  Figures/term_XX_event_centered_shocks.png
  CSVs/term_XX_event_shocks.csv
  CSVs/term_XX_event_shock_party_positions.csv
  CSVs/term_XX_event_shock_topic_layers.csv
  CSVs/term_XX_event_shock_gatekeepers.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_term_interval_polarization import (
    BASE,
    PARTY_COLORS,
    PARTY_SHORT,
    TERM_EVENTS,
    TERMS,
    build_sparse_signed_graph,
    build_speaker_matrix,
    compute_gatekeeper_scores,
    load_monthly_party_positions,
    load_party_year_matrix,
    infer_axis_from_anchors,
)

TOPIC_LABELS = {
    "macro_economy": "Macro economy",
    "constitutional_conflict": "Constitutional conflict",
    "democratic_reform": "Democratic reform",
    "security_conflict": "Security conflict",
}


def add_month_field(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month_dt"] = pd.to_datetime(out["date"]).dt.to_period("M").dt.to_timestamp()
    return out


def month_window(event_month: str, pre_months: int = 2, post_months: int = 2) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    event_dt = pd.to_datetime(event_month + "-01")
    pre_start = event_dt - pd.DateOffset(months=pre_months)
    pre_end = event_dt - pd.DateOffset(days=1)
    post_start = event_dt
    post_end = event_dt + pd.DateOffset(months=post_months, days=27)
    return pre_start, pre_end, post_start, post_end


def monthly_topic_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = (
        df.groupby(["month_dt", "party", "concept_category"], as_index=False)[
            ["mention_count", "beme_positive_hits", "beme_negative_hits"]
        ]
        .sum()
    )
    agg["neg_rate"] = agg["beme_negative_hits"] / agg["mention_count"].replace(0, np.nan)
    agg["pos_rate"] = agg["beme_positive_hits"] / agg["mention_count"].replace(0, np.nan)
    total_party_mentions = agg.groupby(["month_dt", "party"])["mention_count"].transform("sum")
    agg["agenda_share"] = agg["mention_count"] / total_party_mentions.replace(0, np.nan)
    agg = agg.fillna(0.0)

    overall = (
        agg.groupby(["month_dt", "concept_category"], as_index=False)[
            ["mention_count", "beme_positive_hits", "beme_negative_hits"]
        ]
        .sum()
    )
    total_month_mentions = overall.groupby("month_dt")["mention_count"].transform("sum")
    overall["neg_rate"] = overall["beme_negative_hits"] / overall["mention_count"].replace(0, np.nan)
    overall["pos_rate"] = overall["beme_positive_hits"] / overall["mention_count"].replace(0, np.nan)
    overall["salience_share"] = overall["mention_count"] / total_month_mentions.replace(0, np.nan)
    overall = overall.fillna(0.0)
    return agg, overall


def window_means(frame: pd.DataFrame, date_col: str, value_col: str, pre_start, pre_end, post_start, post_end) -> tuple[float, float]:
    pre = frame[(frame[date_col] >= pre_start) & (frame[date_col] <= pre_end)][value_col]
    post = frame[(frame[date_col] >= post_start) & (frame[date_col] <= post_end)][value_col]
    return float(pre.mean()) if not pre.empty else np.nan, float(post.mean()) if not post.empty else np.nan


def compute_window_gatekeepers(term_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    sub = term_df[(term_df["month_dt"] >= start) & (term_df["month_dt"] <= end)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["speaker", "party", "gatekeeper_score", "cross_pressure"])
    feat = (
        sub.groupby(["speaker", "party", "concept_slug"], as_index=False)[
            ["mention_count", "beme_positive_hits", "beme_negative_hits"]
        ]
        .sum()
    )
    feat["balance"] = (
        (feat["beme_positive_hits"] - feat["beme_negative_hits"])
        / (feat["beme_positive_hits"] + feat["beme_negative_hits"] + 1.0)
    )
    totals = feat.groupby(["speaker", "party"])["mention_count"].transform("sum")
    feat["agenda"] = feat["mention_count"] / totals.replace(0, np.nan)
    feat["feature"] = feat["balance"] * feat["agenda"]
    feat = feat.fillna(0.0)
    mat = feat.pivot_table(index=["speaker", "party"], columns="concept_slug", values="feature", fill_value=0.0)
    meta = feat.groupby(["speaker", "party"], as_index=False)["mention_count"].sum().rename(columns={"mention_count": "total_mentions"})
    if mat.shape[0] < 8:
        return pd.DataFrame(columns=["speaker", "party", "gatekeeper_score", "cross_pressure"])
    adj, _ = build_sparse_signed_graph(mat)
    membership = mat.reset_index()[["speaker", "party"]].copy().merge(meta, on=["speaker", "party"], how="left")
    # Reuse polarity heuristic locally
    eigvals, eigvecs = np.linalg.eigh(adj)
    lead_idx = int(np.argmax(np.abs(eigvals)))
    v = eigvecs[:, lead_idx]
    tau = float(np.quantile(np.abs(v), 0.60))
    community = np.where(v >= tau, 1, np.where(v <= -tau, -1, 0))
    membership["community"] = community
    membership["community_label"] = membership["community"].map({1: "Pole +", -1: "Pole -", 0: "Neutral"})
    scores, _ = compute_gatekeeper_scores(membership, adj)
    return scores


def compute_event_shocks_for_term(term: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    csv_path = BASE / f"Term_{term}" / "CSVs" / f"term_{term}_beme_concept_edges.csv"
    df = add_month_field(pd.read_csv(csv_path))

    party_year_mat, _ = load_party_year_matrix(term)
    positions, weights, _ = infer_axis_from_anchors(party_year_mat)
    z_mean = party_year_mat.mean(axis=0)
    z_std = party_year_mat.std(axis=0).replace(0, 1.0)
    monthly_positions, _ = load_monthly_party_positions(term, weights, z_mean, z_std)
    spread = monthly_positions.groupby("month_dt")["position"].agg(["min", "max"]).reset_index()
    spread["spread"] = spread["max"] - spread["min"]
    pivot_positions = monthly_positions.pivot_table(index="month_dt", columns="party", values="position")

    party_topic_monthly, topic_monthly = monthly_topic_metrics(df)

    event_rows = []
    party_delta_rows = []
    topic_rows = []
    gate_rows = []

    for event_month, label, color in TERM_EVENTS.get(term, []):
        pre_start, pre_end, post_start, post_end = month_window(event_month, pre_months=2, post_months=2)
        spread_pre, spread_post = window_means(spread, "month_dt", "spread", pre_start, pre_end, post_start, post_end)

        akp_chp_gap_pre = np.nan
        akp_chp_gap_post = np.nan
        if "Adalet ve Kalkınma Partisi" in pivot_positions.columns and "Cumhuriyet Halk Partisi" in pivot_positions.columns:
            gap = (pivot_positions["Adalet ve Kalkınma Partisi"] - pivot_positions["Cumhuriyet Halk Partisi"]).abs().rename("gap").reset_index()
            akp_chp_gap_pre, akp_chp_gap_post = window_means(gap, "month_dt", "gap", pre_start, pre_end, post_start, post_end)

        for party in pivot_positions.columns:
            series = pivot_positions[party].rename("position").reset_index()
            pre_mean, post_mean = window_means(series, "month_dt", "position", pre_start, pre_end, post_start, post_end)
            if np.isnan(pre_mean) and np.isnan(post_mean):
                continue
            party_delta_rows.append(
                {
                    "term": term,
                    "event_month": event_month,
                    "event_label": label,
                    "party": party,
                    "pre_position": pre_mean,
                    "post_position": post_mean,
                    "delta_position": post_mean - pre_mean if not np.isnan(pre_mean) and not np.isnan(post_mean) else np.nan,
                }
            )

        pre_gate = compute_window_gatekeepers(df, pre_start, pre_end)
        post_gate = compute_window_gatekeepers(df, post_start, post_end)
        if not pre_gate.empty or not post_gate.empty:
            pre_party = pre_gate.groupby("party", as_index=False)["gatekeeper_score"].mean().rename(columns={"gatekeeper_score": "pre_gatekeeper"})
            post_party = post_gate.groupby("party", as_index=False)["gatekeeper_score"].mean().rename(columns={"gatekeeper_score": "post_gatekeeper"})
            gate_merge = pre_party.merge(post_party, on="party", how="outer").fillna(0.0)
            gate_merge["delta_gatekeeper"] = gate_merge["post_gatekeeper"] - gate_merge["pre_gatekeeper"]
            gate_merge["term"] = term
            gate_merge["event_month"] = event_month
            gate_merge["event_label"] = label
            gate_rows.append(gate_merge)

        for topic in sorted(topic_monthly["concept_category"].unique()):
            overall = topic_monthly[topic_monthly["concept_category"].eq(topic)].copy()
            sal_pre, sal_post = window_means(overall, "month_dt", "salience_share", pre_start, pre_end, post_start, post_end)
            neg_pre, neg_post = window_means(overall, "month_dt", "neg_rate", pre_start, pre_end, post_start, post_end)

            party_topic = party_topic_monthly[party_topic_monthly["concept_category"].eq(topic)].copy()
            pre_pt = party_topic[(party_topic["month_dt"] >= pre_start) & (party_topic["month_dt"] <= pre_end)]
            post_pt = party_topic[(party_topic["month_dt"] >= post_start) & (party_topic["month_dt"] <= post_end)]
            top_pre = (
                pre_pt.groupby("party")["agenda_share"].mean().sort_values(ascending=False).index[0]
                if not pre_pt.empty else None
            )
            top_post = (
                post_pt.groupby("party")["agenda_share"].mean().sort_values(ascending=False).index[0]
                if not post_pt.empty else None
            )
            topic_rows.append(
                {
                    "term": term,
                    "event_month": event_month,
                    "event_label": label,
                    "topic": topic,
                    "topic_label": TOPIC_LABELS.get(topic, topic),
                    "salience_pre": sal_pre,
                    "salience_post": sal_post,
                    "salience_delta": sal_post - sal_pre if not np.isnan(sal_pre) and not np.isnan(sal_post) else np.nan,
                    "neg_rate_pre": neg_pre,
                    "neg_rate_post": neg_post,
                    "neg_rate_delta": neg_post - neg_pre if not np.isnan(neg_pre) and not np.isnan(neg_post) else np.nan,
                    "top_party_pre": top_pre,
                    "top_party_post": top_post,
                }
            )

        top_pre_row = pre_gate.iloc[0] if not pre_gate.empty else None
        top_post_row = post_gate.iloc[0] if not post_gate.empty else None
        event_rows.append(
            {
                "term": term,
                "event_month": event_month,
                "event_label": label,
                "spread_pre": spread_pre,
                "spread_post": spread_post,
                "spread_delta": spread_post - spread_pre if not np.isnan(spread_pre) and not np.isnan(spread_post) else np.nan,
                "akp_chp_gap_pre": akp_chp_gap_pre,
                "akp_chp_gap_post": akp_chp_gap_post,
                "akp_chp_gap_delta": akp_chp_gap_post - akp_chp_gap_pre if not np.isnan(akp_chp_gap_pre) and not np.isnan(akp_chp_gap_post) else np.nan,
                "top_gatekeeper_pre_speaker": top_pre_row["speaker"] if top_pre_row is not None else None,
                "top_gatekeeper_pre_party": top_pre_row["party"] if top_pre_row is not None else None,
                "top_gatekeeper_pre_score": float(top_pre_row["gatekeeper_score"]) if top_pre_row is not None else np.nan,
                "top_gatekeeper_post_speaker": top_post_row["speaker"] if top_post_row is not None else None,
                "top_gatekeeper_post_party": top_post_row["party"] if top_post_row is not None else None,
                "top_gatekeeper_post_score": float(top_post_row["gatekeeper_score"]) if top_post_row is not None else np.nan,
            }
        )

    events_df = pd.DataFrame(event_rows)
    party_df = pd.DataFrame(party_delta_rows)
    topic_df = pd.DataFrame(topic_rows)
    gate_df = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    return events_df, party_df, topic_df, gate_df


def plot_event_centered_shocks(term: int, events_df: pd.DataFrame, party_df: pd.DataFrame, topic_df: pd.DataFrame, gate_df: pd.DataFrame, fig_path: Path) -> None:
    fig = plt.figure(figsize=(16, 11), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.05], height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    labels = events_df["event_label"].tolist()
    x = np.arange(len(labels))
    ax1.bar(x - 0.16, events_df["spread_pre"], width=0.32, color="#B0BEC5", label="Pre")
    ax1.bar(x + 0.16, events_df["spread_post"], width=0.32, color="#C62828", label="Post")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_title(f"Term {term}: ideological spread before vs after shocks")
    ax1.set_ylabel("Average spread")
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(gs[0, 1])
    key_parties = [
        p for p in [
            "Adalet ve Kalkınma Partisi",
            "Cumhuriyet Halk Partisi",
            "Milliyetçi Hareket Partisi",
            "Halkların Demokratik Partisi",
            "DEM Parti",
            "İYİ Parti",
        ] if p in party_df["party"].unique()
    ][:6]
    heat = (
        party_df[party_df["party"].isin(key_parties)]
        .pivot_table(index="party", columns="event_label", values="delta_position")
        .reindex(index=key_parties)
    )
    im = ax2.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-np.nanmax(np.abs(heat.to_numpy())), vmax=np.nanmax(np.abs(heat.to_numpy())))
    ax2.set_yticks(np.arange(len(heat.index)))
    ax2.set_yticklabels([PARTY_SHORT.get(p, p) for p in heat.index])
    ax2.set_xticks(np.arange(len(heat.columns)))
    ax2.set_xticklabels(heat.columns, rotation=25, ha="right")
    ax2.set_title("Party position deltas around shocks")
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            val = heat.iloc[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03).set_label("Post - pre")

    ax3 = fig.add_subplot(gs[1, 0])
    if not gate_df.empty:
        party_heat = (
            gate_df.assign(short=gate_df["party"].map(lambda x: PARTY_SHORT.get(x, x)))
            .pivot_table(index="short", columns="event_label", values="delta_gatekeeper")
        )
        im2 = ax3.imshow(party_heat.to_numpy(dtype=float), aspect="auto", cmap="PuOr", vmin=-np.nanmax(np.abs(party_heat.to_numpy())), vmax=np.nanmax(np.abs(party_heat.to_numpy())))
        ax3.set_yticks(np.arange(len(party_heat.index)))
        ax3.set_yticklabels(party_heat.index)
        ax3.set_xticks(np.arange(len(party_heat.columns)))
        ax3.set_xticklabels(party_heat.columns, rotation=25, ha="right")
        ax3.set_title("Party-level gatekeeper change")
        fig.colorbar(im2, ax=ax3, fraction=0.046, pad=0.03).set_label("Post - pre")
    else:
        ax3.text(0.5, 0.5, "No gatekeeper deltas available", ha="center", va="center")
        ax3.set_axis_off()

    ax4 = fig.add_subplot(gs[1, 1])
    topic_heat = (
        topic_df.pivot_table(index="topic_label", columns="event_label", values="salience_delta")
        .reindex(index=[TOPIC_LABELS[k] for k in TOPIC_LABELS if TOPIC_LABELS[k] in topic_df["topic_label"].unique()])
    )
    im3 = ax4.imshow(topic_heat.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=np.nanmin(topic_heat.to_numpy()), vmax=np.nanmax(topic_heat.to_numpy()))
    ax4.set_yticks(np.arange(len(topic_heat.index)))
    ax4.set_yticklabels(topic_heat.index)
    ax4.set_xticks(np.arange(len(topic_heat.columns)))
    ax4.set_xticklabels(topic_heat.columns, rotation=25, ha="right")
    ax4.set_title("Topic-layer salience change around shocks")
    for i in range(len(topic_heat.index)):
        for j in range(len(topic_heat.columns)):
            val = topic_heat.iloc[i, j]
            if not np.isnan(val):
                ax4.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im3, ax=ax4, fraction=0.046, pad=0.03).set_label("Post - pre salience")

    fig.suptitle(f"TBMM Term {term}: event-centered interval and gatekeeper shocks", fontsize=16, y=0.98)
    fig.text(
        0.01,
        0.01,
        "Interpretation: these panels summarize how the latent ideological spread, party positions, gatekeeper distribution, "
        "and topic salience reconfigure around major political shocks within the term.",
        fontsize=10,
    )
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def process_term(term: int) -> None:
    events_df, party_df, topic_df, gate_df = compute_event_shocks_for_term(term)
    term_dir = BASE / f"Term_{term}"
    csv_dir = term_dir / "CSVs"
    fig_dir = term_dir / "Figures"
    events_df.to_csv(csv_dir / f"term_{term}_event_shocks.csv", index=False, encoding="utf-8-sig")
    party_df.to_csv(csv_dir / f"term_{term}_event_shock_party_positions.csv", index=False, encoding="utf-8-sig")
    topic_df.to_csv(csv_dir / f"term_{term}_event_shock_topic_layers.csv", index=False, encoding="utf-8-sig")
    gate_df.to_csv(csv_dir / f"term_{term}_event_shock_gatekeepers.csv", index=False, encoding="utf-8-sig")
    plot_event_centered_shocks(term, events_df, party_df, topic_df, gate_df, fig_dir / f"term_{term}_event_centered_shocks.png")
    print(f"Term {term}: wrote event-centered shocks for {len(events_df)} events")


def main() -> None:
    for term in TERMS:
        process_term(term)


if __name__ == "__main__":
    main()
