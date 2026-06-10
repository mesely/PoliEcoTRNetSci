from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_economic_shock_study import (
    BASE,
    PARTY_COLORS,
    PARTY_SHORT,
    brokerage_concentration_hhi,
    build_stage_gatekeepers,
    read_term_edges,
    resolve_event_month,
    stage_months,
)


FOCUS_EVENTS = {
    23: "Global crisis reaction",
    27: "FX crisis",
    28: "Local election shock",
}

OUT_CSV = BASE / "Inter_Term" / "CSVs" / "step3_structural_brokerage_turnover.csv"
OUT_FIG = BASE / "Inter_Term" / "Figures" / "figure_7_brokerage_turnover_panel.png"
OUT_NOTE = BASE / "Inter_Term" / "Notes" / "step3_structural_brokerage_results.txt"


def focus_rows() -> pd.DataFrame:
    rows = []
    for term, event_label in FOCUS_EVENTS.items():
        path = BASE / f"Term_{term}" / "CSVs" / f"term_{term}_economic_event_summary.csv"
        df = pd.read_csv(path)
        row = df[df["event_label"].eq(event_label)].iloc[0]
        rows.append(
            {
                "term": term,
                "event_label": event_label,
                "event_slug": row["event_slug"],
                "event_month_effective": row["event_month_effective"],
                "event_kind": row["event_kind"],
            }
        )
    return pd.DataFrame(rows)


def summarize_stage_gatekeepers(term: int, event_label: str, event_slug: str, event_month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = read_term_edges(term)
    event_month = resolve_event_month(event_month, sorted(edges["month_key"].dropna().unique()))
    windows = stage_months(event_month)

    speaker_rows: list[dict[str, object]] = []
    party_rows: list[dict[str, object]] = []
    for stage, months in windows.items():
        scores = build_stage_gatekeepers(edges, months)
        if scores.empty:
            continue

        top_speaker = scores.sort_values("gatekeeper_score", ascending=False).iloc[0]
        speaker_rows.append(
            {
                "term": term,
                "event_label": event_label,
                "event_slug": event_slug,
                "event_month_effective": event_month,
                "stage": stage,
                "top_broker_speaker": top_speaker["speaker"],
                "top_broker_party": top_speaker["party"],
                "top_broker_score": float(top_speaker["gatekeeper_score"]),
                "top_broker_cross_pressure": float(top_speaker["cross_pressure"]),
            }
        )

        party_summary = (
            scores.groupby("party", as_index=False)
            .agg(
                mean_gatekeeper_score=("gatekeeper_score", "mean"),
                max_gatekeeper_score=("gatekeeper_score", "max"),
                n_brokers=("speaker", "nunique"),
            )
            .sort_values("mean_gatekeeper_score", ascending=False)
            .reset_index(drop=True)
        )
        top_party = party_summary.iloc[0]
        pos_scores = party_summary["mean_gatekeeper_score"].clip(lower=0.0)
        total_pos = float(pos_scores.sum())
        top_party_share = float(top_party["mean_gatekeeper_score"] / total_pos) if total_pos > 0 else np.nan
        party_rows.append(
            {
                "term": term,
                "event_label": event_label,
                "event_slug": event_slug,
                "event_month_effective": event_month,
                "stage": stage,
                "top_broker_party": top_party["party"],
                "top_party_mean_score": float(top_party["mean_gatekeeper_score"]),
                "top_party_max_score": float(top_party["max_gatekeeper_score"]),
                "top_party_n_brokers": int(top_party["n_brokers"]),
                "top_party_share": top_party_share,
                "brokerage_concentration_hhi": brokerage_concentration_hhi(scores),
            }
        )
    return pd.DataFrame(speaker_rows), pd.DataFrame(party_rows)


def build_turnover_table() -> pd.DataFrame:
    speakers = []
    parties = []
    for row in focus_rows().itertuples(index=False):
        s_df, p_df = summarize_stage_gatekeepers(
            int(row.term),
            str(row.event_label),
            str(row.event_slug),
            str(row.event_month_effective),
        )
        speakers.append(s_df)
        parties.append(p_df)

    speaker_df = pd.concat(speakers, ignore_index=True)
    party_df = pd.concat(parties, ignore_index=True)

    s_piv = speaker_df.pivot_table(
        index=["term", "event_label", "event_slug", "event_month_effective"],
        columns="stage",
        values=["top_broker_party", "top_broker_speaker", "top_broker_score"],
        aggfunc="first",
    )
    s_piv.columns = ["_".join(col).strip() for col in s_piv.columns.to_flat_index()]
    s_piv = s_piv.reset_index()

    p_piv = party_df.pivot_table(
        index=["term", "event_label", "event_slug", "event_month_effective"],
        columns="stage",
        values=["top_broker_party", "top_party_mean_score", "brokerage_concentration_hhi"],
        aggfunc="first",
    )
    p_piv.columns = ["_".join(col).strip() for col in p_piv.columns.to_flat_index()]
    p_piv = p_piv.reset_index()

    out = s_piv.merge(
        p_piv,
        on=["term", "event_label", "event_slug", "event_month_effective"],
        how="outer",
        suffixes=("", "_partyagg"),
    )
    out["speaker_turnover"] = out["top_broker_speaker_pre"].fillna("") != out["top_broker_speaker_shock"].fillna("")
    out["party_turnover"] = out["top_broker_party_pre_partyagg"].fillna("") != out["top_broker_party_shock_partyagg"].fillna("")
    out["broker_score_delta"] = out["top_party_mean_score_shock"] - out["top_party_mean_score_pre"]
    out["brokerage_concentration_delta"] = out["brokerage_concentration_hhi_shock"] - out["brokerage_concentration_hhi_pre"]
    out["top_broker_persistence"] = out["top_broker_party_shock_partyagg"].fillna("") == out["top_broker_party_post_partyagg"].fillna("")
    return out.sort_values("term").reset_index(drop=True)


def plot_turnover_panel(df: pd.DataFrame) -> None:
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), facecolor="white")

    ax1, ax2, ax3 = axes

    x = np.arange(len(df))
    width = 0.26
    ax1.bar(x - width / 2, df["top_party_mean_score_pre"], width=width, color="#cbd5e1", label="Pre")
    ax1.bar(x + width / 2, df["top_party_mean_score_shock"], width=width, color="#dc2626", label="Shock")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"T{t}" for t in df["term"]])
    ax1.set_ylabel("Top party mean gatekeeper score")
    ax1.set_title("Broker strength in pre and shock windows")
    ax1.grid(axis="y", alpha=0.2)
    ax1.legend(frameon=False, fontsize=8)

    for i, row in enumerate(df.itertuples(index=False)):
        pre_party = getattr(row, "top_broker_party_pre_partyagg", None)
        shock_party = getattr(row, "top_broker_party_shock_partyagg", None)
        pre_color = PARTY_COLORS.get(pre_party, "#94a3b8")
        shock_color = PARTY_COLORS.get(shock_party, "#dc2626")
        ax2.plot([0, 1], [i, i], color="#cbd5e1", linewidth=1.2)
        ax2.scatter([0], [i], s=180, color=pre_color, edgecolor="white", linewidth=1.0)
        ax2.scatter([1], [i], s=180, color=shock_color, edgecolor="white", linewidth=1.0)
        ax2.text(-0.05, i, PARTY_SHORT.get(pre_party, str(pre_party)), ha="right", va="center", fontsize=9)
        ax2.text(1.05, i, PARTY_SHORT.get(shock_party, str(shock_party)), ha="left", va="center", fontsize=9)
    ax2.set_xlim(-0.25, 1.25)
    ax2.set_ylim(-0.5, len(df) - 0.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Pre broker party", "Shock broker party"])
    ax2.set_yticks(np.arange(len(df)))
    ax2.set_yticklabels([f"T{t}" for t in df["term"]])
    ax2.set_title("Brokerage turnover by party")
    ax2.grid(axis="x", alpha=0.15)

    colors = ["#dc2626" if bool(v) else "#0f766e" for v in df["party_turnover"]]
    ax3.bar(np.arange(len(df)), df["broker_score_delta"], color=colors, alpha=0.92)
    ax3.axhline(0.0, color="#9ca3af", linewidth=1.0)
    ax3.set_xticks(np.arange(len(df)))
    ax3.set_xticklabels([f"T{t}\n{label[:12]}" for t, label in zip(df["term"], df["event_label"])], fontsize=8)
    ax3.set_ylabel("Shock - pre broker score")
    ax3.set_title("Broker turnover and score change")
    ax3.grid(axis="y", alpha=0.2)

    fig.suptitle("structural brokerage turnover across core shock windows", fontsize=16, y=0.98)
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_note(df: pd.DataFrame) -> None:
    lines = ["Step 3 structural brokerage summary", ""]
    for row in df.itertuples(index=False):
        lines.extend(
            [
                f"Term {row.term} · {row.event_label}",
                f"  Pre broker speaker : {row.top_broker_speaker_pre} / {row.top_broker_party_pre}",
                f"  Shock broker speaker : {row.top_broker_speaker_shock} / {row.top_broker_party_shock}",
                f"  Pre broker party (mean score) : {row.top_broker_party_pre_partyagg} ({row.top_party_mean_score_pre:.3f})",
                f"  Shock broker party (mean score) : {row.top_broker_party_shock_partyagg} ({row.top_party_mean_score_shock:.3f})",
                f"  Speaker turnover : {bool(row.speaker_turnover)}",
                f"  Party turnover   : {bool(row.party_turnover)}",
                f"  Broker score Δ   : {row.broker_score_delta:.3f}",
                f"  Brokerage concentration Δ : {row.brokerage_concentration_delta:.3f}" if pd.notna(row.brokerage_concentration_delta) else "  Brokerage concentration Δ : n/a",
                f"  Top broker persistence into post : {bool(row.top_broker_persistence)}",
                "",
            ]
        )
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = build_turnover_table()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    plot_turnover_panel(df)
    write_note(df)
    print(
        df[
            [
                "term",
                "event_label",
                "top_broker_speaker_pre",
                "top_broker_speaker_shock",
                "top_broker_party_pre_partyagg",
                "top_broker_party_shock_partyagg",
                "speaker_turnover",
                "party_turnover",
                "broker_score_delta",
                "brokerage_concentration_delta",
                "top_broker_persistence",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
