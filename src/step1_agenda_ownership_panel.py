from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_economic_shock_study import (
    BASE,
    EVENT_CATALOG,
    PARTY_COLORS,
    PARTY_SHORT,
    ECONOMIC_TOPICS,
    read_term_edges,
    resolve_event_month,
    stage_months,
    monthly_topic_panel,
    build_party_topic_matrix,
    party_summary_from_feat,
    draw_bipartite,
)


FOCUS_TERMS = [23, 27, 28]
OUT_CSV = BASE / "Inter_Term" / "CSVs" / "step1_agenda_ownership_panel.csv"
OUT_FIG = BASE / "Inter_Term" / "Figures" / "figure_6_agenda_ownership_panel.png"
OUT_NOTE = BASE / "Inter_Term" / "Notes" / "step1_agenda_ownership_results.txt"


def choose_focus_events() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for term in FOCUS_TERMS:
        summary_path = BASE / f"Term_{term}" / "CSVs" / f"term_{term}_economic_event_summary.csv"
        summary = pd.read_csv(summary_path)
        summary["abs_salience_delta"] = summary["econ_salience_delta"].abs().fillna(-1.0)
        core = summary[summary["paper_role"].eq("core")].copy()
        if core.empty:
            core = summary.copy()
        pick = core.sort_values(["abs_salience_delta", "event_label"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "term": term,
                "event_label": pick["event_label"],
                "event_slug": pick["event_slug"],
                "event_kind": pick["event_kind"],
                "event_month_effective": pick["event_month_effective"],
                "paper_role": pick["paper_role"],
            }
        )
    return pd.DataFrame(rows)


def stage_summary_for_event(term: int, event_slug: str, event_month: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    edges_all = read_term_edges(term)
    edges_econ = edges_all[edges_all["concept_slug"].isin(ECONOMIC_TOPICS.keys())].copy()
    windows = stage_months(event_month)
    topic_panel = monthly_topic_panel(edges_all)

    stage_rows: list[dict[str, object]] = []
    stage_mats: dict[str, pd.DataFrame] = {}
    stage_parties: dict[str, pd.DataFrame] = {}
    for stage, months in windows.items():
        stage_topic = topic_panel[
            topic_panel["month_key"].isin(months) & topic_panel["concept_category"].eq("macro_economy")
        ].copy()
        salience = float(stage_topic["share"].mean()) if not stage_topic.empty else np.nan

        mat, feat = build_party_topic_matrix(edges_econ, months, list(ECONOMIC_TOPICS.keys()))
        party_summary = party_summary_from_feat(feat)
        stage_mats[stage] = mat
        stage_parties[stage] = party_summary

        if party_summary.empty:
            stage_rows.append(
                {
                    "term": term,
                    "event_slug": event_slug,
                    "event_month_effective": event_month,
                    "stage": stage,
                    "econ_salience": salience,
                    "top_owner_party": None,
                    "top_owner_share": np.nan,
                    "top_owner_balance": np.nan,
                    "owner_concentration_hhi": np.nan,
                }
            )
            continue

        party_summary = party_summary.sort_values("agenda_share", ascending=False).reset_index(drop=True)
        hhi = float((party_summary["agenda_share"] ** 2).sum())
        top = party_summary.iloc[0]
        stage_rows.append(
            {
                "term": term,
                "event_slug": event_slug,
                "event_month_effective": event_month,
                "stage": stage,
                "econ_salience": salience,
                "top_owner_party": top["party"],
                "top_owner_share": float(top["agenda_share"]),
                "top_owner_balance": float(top["balance"]),
                "owner_concentration_hhi": hhi,
            }
        )
    return pd.DataFrame(stage_rows), stage_mats, stage_parties


def build_panel() -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
    focus = choose_focus_events()
    panel_rows: list[pd.DataFrame] = []
    stage_payload: dict[int, dict[str, object]] = {}
    for row in focus.itertuples(index=False):
        event_month = resolve_event_month(str(row.event_month_effective), sorted(read_term_edges(int(row.term))["month_key"].unique()))
        summary, mats, parties = stage_summary_for_event(int(row.term), str(row.event_slug), event_month)
        summary["event_label"] = row.event_label
        summary["event_kind"] = row.event_kind
        summary["paper_role"] = row.paper_role
        panel_rows.append(summary)
        stage_payload[int(row.term)] = {
            "event_label": row.event_label,
            "event_slug": row.event_slug,
            "event_month_effective": event_month,
            "mats": mats,
            "parties": parties,
        }

    panel = pd.concat(panel_rows, ignore_index=True)
    pre = panel[panel["stage"].eq("pre")][["term", "top_owner_party", "top_owner_share", "econ_salience"]].rename(
        columns={
            "top_owner_party": "pre_owner_party",
            "top_owner_share": "pre_owner_share",
            "econ_salience": "pre_salience",
        }
    )
    shock = panel[panel["stage"].eq("shock")][["term", "top_owner_party", "top_owner_share", "econ_salience"]].rename(
        columns={
            "top_owner_party": "shock_owner_party",
            "top_owner_share": "shock_owner_share",
            "econ_salience": "shock_salience",
        }
    )
    post = panel[panel["stage"].eq("post")][["term", "top_owner_party", "top_owner_share", "econ_salience"]].rename(
        columns={
            "top_owner_party": "post_owner_party",
            "top_owner_share": "post_owner_share",
            "econ_salience": "post_salience",
        }
    )
    merged = (
        focus.merge(pre, on="term", how="left")
        .merge(shock, on="term", how="left")
        .merge(post, on="term", how="left")
    )
    merged["salience_delta"] = merged["shock_salience"] - merged["pre_salience"]
    merged["owner_shift"] = np.where(
        merged["pre_owner_party"].notna() & merged["shock_owner_party"].notna(),
        merged["pre_owner_party"] != merged["shock_owner_party"],
        np.nan,
    )
    merged["owner_share_delta"] = merged["shock_owner_share"] - merged["pre_owner_share"]
    merged["owner_persistence"] = np.where(
        merged["shock_owner_party"].notna() & merged["post_owner_party"].notna(),
        merged["shock_owner_party"] == merged["post_owner_party"],
        np.nan,
    )
    return merged, stage_payload


def plot_ownership_panel(panel: pd.DataFrame, stage_payload: dict[int, dict[str, object]]) -> None:
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 11), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 0.95], hspace=0.26, wspace=0.20)

    for idx, term in enumerate(FOCUS_TERMS):
        payload = stage_payload[term]
        ax = fig.add_subplot(gs[0, idx])
        draw_bipartite(
            ax,
            payload["mats"]["shock"],
            payload["parties"]["shock"],
            f"Term {term}: {payload['event_label']} shock-stage bipartite",
        )

    ax4 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(panel))
    width = 0.23
    ax4.bar(x - width, panel["pre_salience"], width=width, color="#cbd5e1", label="Pre")
    ax4.bar(x, panel["shock_salience"], width=width, color="#dc2626", label="Shock")
    ax4.bar(x + width, panel["post_salience"], width=width, color="#94a3b8", label="Post")
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"T{t}" for t in panel["term"]])
    ax4.set_ylabel("Economic salience")
    ax4.set_title("Economic salience across event windows")
    ax4.legend(frameon=False, fontsize=8)
    ax4.grid(axis="y", alpha=0.2)

    ax5 = fig.add_subplot(gs[1, 1])
    pre_label_rows: list[dict[str, object]] = []
    for i, row in enumerate(panel.itertuples(index=False)):
        pre_party = row.pre_owner_party if isinstance(row.pre_owner_party, str) else "N/A"
        shock_party = row.shock_owner_party if isinstance(row.shock_owner_party, str) else "N/A"
        color = PARTY_COLORS.get(shock_party, "#991b1b")
        if pd.notna(row.pre_owner_share):
            ax5.plot([0, 1], [row.pre_owner_share, row.shock_owner_share], color=color, linewidth=2.2, alpha=0.9)
            ax5.scatter([0], [row.pre_owner_share], color=PARTY_COLORS.get(pre_party, "#94a3b8"), s=60, zorder=3)
        ax5.scatter([1], [row.shock_owner_share], color=color, s=80, zorder=3)
        if pd.notna(row.pre_owner_share):
            pre_label_rows.append(
                {"term": row.term, "party": pre_party, "share": float(row.pre_owner_share), "color": color}
            )
        ax5.text(1.03, row.shock_owner_share, PARTY_SHORT.get(shock_party, shock_party), ha="left", va="center", fontsize=8)
        ax5.text(1.18, row.shock_owner_share, f"T{row.term}", ha="left", va="center", fontsize=9, color=color, fontweight="bold")

    # Group left-side pre-owner labels that would visually collide: rows
    # whose pre_owner_share values fall within COLLISION_MARGIN of each
    # other (chained transitively, so 3+ rows in a tight cluster are all
    # grouped together). Within such a cluster:
    #   - the per-line "T{term}" code labels are vertically staggered
    #     (symmetrically around the cluster's mean share) so each remains
    #     individually legible and color-matched to its line/marker;
    #   - the pre-owner party-name label is drawn ONCE at the cluster mean,
    #     in a neutral color, IF every row in the cluster shares the same
    #     pre_owner_party (otherwise each gets its own party-name label,
    #     staggered the same way as the T-codes).
    COLLISION_MARGIN = 0.02
    LABEL_STAGGER_STEP = 0.012
    pre_label_rows.sort(key=lambda d: d["share"])
    clusters: list[list[dict[str, object]]] = []
    for entry in pre_label_rows:
        if clusters and abs(entry["share"] - clusters[-1][-1]["share"]) < COLLISION_MARGIN:
            clusters[-1].append(entry)
        else:
            clusters.append([entry])

    for cluster in clusters:
        n = len(cluster)
        if n == 1:
            entry = cluster[0]
            ax5.text(-0.18, entry["share"], f"T{entry['term']}", ha="right", va="center", fontsize=9, color=entry["color"], fontweight="bold")
            ax5.text(-0.03, entry["share"], PARTY_SHORT.get(entry["party"], entry["party"]), ha="right", va="center", fontsize=8)
            continue

        # Order staggered T-code positions by term for a deterministic layout.
        cluster_by_term = sorted(cluster, key=lambda d: d["term"])
        mean_share = float(np.mean([e["share"] for e in cluster]))
        for j, entry in enumerate(cluster_by_term):
            offset = (j - (n - 1) / 2.0) * LABEL_STAGGER_STEP
            ax5.text(-0.18, mean_share + offset, f"T{entry['term']}", ha="right", va="center", fontsize=9, color=entry["color"], fontweight="bold")

        parties = {e["party"] for e in cluster}
        if len(parties) == 1:
            party = cluster[0]["party"]
            ax5.text(-0.03, mean_share, PARTY_SHORT.get(party, party), ha="right", va="center", fontsize=8, color="#44403c")
        else:
            for j, entry in enumerate(cluster_by_term):
                offset = (j - (n - 1) / 2.0) * LABEL_STAGGER_STEP
                ax5.text(-0.03, mean_share + offset, PARTY_SHORT.get(entry["party"], entry["party"]), ha="right", va="center", fontsize=8, color="#44403c")

    ax5.set_xlim(-0.4, 1.4)
    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(["Pre owner share", "Shock owner share"])
    ax5.set_ylabel("Top owner agenda share")
    ax5.set_title("Agenda ownership reallocation")
    ax5.grid(axis="y", alpha=0.2)

    ax6 = fig.add_subplot(gs[1, 2])
    colors = ["#dc2626" if bool(v) else "#0f766e" for v in panel["owner_shift"]]
    ax6.bar(np.arange(len(panel)), panel["salience_delta"], color=colors, alpha=0.92)
    ax6.axhline(0.0, color="#9ca3af", linewidth=1.0)
    ax6.set_xticks(np.arange(len(panel)))
    ax6.set_xticklabels(
        [f"T{term}\n{label[:12]}" for term, label in zip(panel["term"], panel["event_label"])],
        fontsize=8,
    )
    ax6.set_ylabel("Shock - pre salience")
    ax6.set_title("Salience jump and owner-shift indicator")
    ax6.grid(axis="y", alpha=0.2)

    fig.suptitle("economic salience and agenda ownership under core shock windows", fontsize=16, y=0.98)
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_note(panel: pd.DataFrame) -> None:
    lines = [
        "Step 1 agenda ownership summary",
        "",
    ]
    for row in panel.itertuples(index=False):
        lines.extend(
            [
                f"Term {row.term} · {row.event_label}",
                f"  Event month : {row.event_month_effective}",
                f"  Pre owner   : {row.pre_owner_party} ({row.pre_owner_share:.3f})" if pd.notna(row.pre_owner_share) else f"  Pre owner   : {row.pre_owner_party}",
                f"  Shock owner : {row.shock_owner_party} ({row.shock_owner_share:.3f})" if pd.notna(row.shock_owner_share) else f"  Shock owner : {row.shock_owner_party}",
                f"  Post owner  : {row.post_owner_party} ({row.post_owner_share:.3f})" if pd.notna(row.post_owner_share) else f"  Post owner  : {row.post_owner_party}",
                f"  Salience Δ  : {row.salience_delta:.3f}" if pd.notna(row.salience_delta) else "  Salience Δ  : n/a",
                f"  Owner shift : {bool(row.owner_shift)}",
                f"  Owner persistence : {bool(row.owner_persistence)}" if pd.notna(row.owner_persistence) else "  Owner persistence : n/a",
                "",
            ]
        )
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    panel, stage_payload = build_panel()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    plot_ownership_panel(panel, stage_payload)
    write_note(panel)
    print(panel.to_string(index=False))


if __name__ == "__main__":
    main()
