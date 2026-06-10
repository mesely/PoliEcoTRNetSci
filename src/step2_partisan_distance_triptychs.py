from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_economic_shock_study import (
    BASE,
    build_global_axis,
    compute_stage_snapshot,
    monthly_positions_for_term,
    negative_edge_share,
    plot_economic_triptych,
    read_term_edges,
    resolve_event_month,
    signed_modularity,
    stage_months,
)


FOCUS_EVENTS = {
    23: "Global crisis reaction",
    27: "FX crisis",
    28: "Local election shock",
}
FIGURE_ID_MAP = {23: 3, 27: 4, 28: 5}

OUT_CSV = BASE / "Inter_Term" / "CSVs" / "step2_partisan_distance_triptychs.csv"
OUT_NOTE = BASE / "Inter_Term" / "Notes" / "step2_partisan_distance_results.txt"
OUT_FIG_DIR = BASE / "Inter_Term" / "Figures"


def mean_negative_affinity(affinity: pd.DataFrame) -> float:
    if affinity.empty:
        return np.nan
    vals = []
    for i in range(len(affinity.index)):
        for j in range(i + 1, len(affinity.columns)):
            val = float(affinity.iloc[i, j])
            if val < 0:
                vals.append(abs(val))
            else:
                vals.append(0.0)
    return float(np.mean(vals)) if vals else np.nan


def mean_absolute_affinity(affinity: pd.DataFrame) -> float:
    if affinity.empty:
        return np.nan
    vals = []
    for i in range(len(affinity.index)):
        for j in range(i + 1, len(affinity.columns)):
            vals.append(abs(float(affinity.iloc[i, j])))
    return float(np.mean(vals)) if vals else np.nan


def load_focus_rows() -> pd.DataFrame:
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
                "paper_role": row["paper_role"],
            }
        )
    return pd.DataFrame(rows)


def build_distance_table() -> pd.DataFrame:
    _, monthly_positions, _, _, _ = build_global_axis()
    focus = load_focus_rows()
    rows: list[dict[str, object]] = []

    for row in focus.itertuples(index=False):
        term = int(row.term)
        edges = read_term_edges(term)
        month_keys = sorted(edges["month_key"].dropna().unique())
        event_month = resolve_event_month(str(row.event_month_effective), month_keys)
        windows = stage_months(event_month)
        monthly_pos_term = monthly_positions_for_term(monthly_positions, term)
        stage_data: dict[str, dict[str, pd.DataFrame]] = {}
        stage_positions: dict[str, pd.Series] = {}

        for stage in ["pre", "shock", "post"]:
            mat, _, party_summary, affinity, pos = compute_stage_snapshot(
                edges,
                monthly_pos_term,
                windows[stage],
            )
            stage_data[stage] = {
                "mat": mat,
                "party_summary": party_summary,
                "affinity": affinity,
            }
            stage_positions[stage] = pos
            vals = pos.dropna()
            spread = float(vals.max() - vals.min()) if not vals.empty else np.nan
            rows.append(
                {
                    "term": term,
                    "event_label": row.event_label,
                    "event_slug": row.event_slug,
                    "event_kind": row.event_kind,
                    "stage": stage,
                    "event_month_effective": event_month,
                    "n_parties": int(len(affinity.index)) if not affinity.empty else 0,
                    "spread": spread,
                    "mean_negative_affinity": mean_negative_affinity(affinity),
                    "mean_absolute_affinity": mean_absolute_affinity(affinity),
                    "negative_edge_share": negative_edge_share(affinity),
                    "signed_modularity": signed_modularity(affinity, pos),
                    "leftmost_party": vals.idxmin() if not vals.empty else None,
                    "rightmost_party": vals.idxmax() if not vals.empty else None,
                    "top_owner_party": party_summary.sort_values("agenda_share", ascending=False).iloc[0]["party"] if not party_summary.empty else None,
                }
            )
        OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
        out_fig = OUT_FIG_DIR / f"figure_{FIGURE_ID_MAP[term]}_term_{term}_shock_triptych.png"
        plot_economic_triptych(term, row.event_slug, row.event_label, stage_data, stage_positions, out_fig)
    out = pd.DataFrame(rows)
    piv = out.pivot_table(
        index=["term", "event_label", "event_slug", "event_kind", "event_month_effective"],
        columns="stage",
        values=[
            "spread",
            "mean_negative_affinity",
            "mean_absolute_affinity",
            "negative_edge_share",
            "signed_modularity",
        ],
    )
    piv.columns = ["_".join(col).strip() for col in piv.columns.to_flat_index()]
    piv = piv.reset_index()
    piv["spread_delta"] = piv["spread_shock"] - piv["spread_pre"]
    piv["neg_affinity_delta"] = piv["mean_negative_affinity_shock"] - piv["mean_negative_affinity_pre"]
    piv["negative_edge_share_delta"] = piv["negative_edge_share_shock"] - piv["negative_edge_share_pre"]
    piv["signed_modularity_delta"] = piv["signed_modularity_shock"] - piv["signed_modularity_pre"]
    return out.merge(
        piv[
            [
                "term",
                "event_label",
                "spread_pre",
                "spread_shock",
                "spread_post",
                "spread_delta",
                "mean_negative_affinity_pre",
                "mean_negative_affinity_shock",
                "mean_negative_affinity_post",
                "neg_affinity_delta",
                "negative_edge_share_pre",
                "negative_edge_share_shock",
                "negative_edge_share_post",
                "negative_edge_share_delta",
                "signed_modularity_pre",
                "signed_modularity_shock",
                "signed_modularity_post",
                "signed_modularity_delta",
            ]
        ],
        on=["term", "event_label"],
        how="left",
    )


def write_note(df: pd.DataFrame) -> None:
    lines = ["Step 2 partisan distance summary", ""]
    for (term, event), g in df.groupby(["term", "event_label"], sort=False):
        g = g.sort_values("stage")
        head = g.iloc[0]
        lines.extend(
            [
                f"Term {term} · {event}",
                f"  Spread pre/shock/post : {head['spread_pre']:.3f} / {head['spread_shock']:.3f} / {head['spread_post']:.3f}",
                f"  Spread delta          : {head['spread_delta']:.3f}",
                f"  Negative affinity pre/shock/post : {head['mean_negative_affinity_pre']:.3f} / {head['mean_negative_affinity_shock']:.3f} / {head['mean_negative_affinity_post']:.3f}",
                f"  Negative affinity delta         : {head['neg_affinity_delta']:.3f}",
                f"  Negative-edge share pre/shock/post : {head['negative_edge_share_pre']:.3f} / {head['negative_edge_share_shock']:.3f} / {head['negative_edge_share_post']:.3f}",
                f"  Signed modularity pre/shock/post  : {head['signed_modularity_pre']:.3f} / {head['signed_modularity_shock']:.3f} / {head['signed_modularity_post']:.3f}",
                "",
            ]
        )
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = build_distance_table()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    write_note(df)
    summary = (
        df[["term", "event_label", "spread_pre", "spread_shock", "spread_post", "spread_delta", "mean_negative_affinity_pre", "mean_negative_affinity_shock", "neg_affinity_delta", "negative_edge_share_shock", "signed_modularity_shock"]]
        .drop_duplicates()
        .sort_values("term")
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
