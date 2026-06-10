from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_economic_shock_study import BASE


SELECTED_EVENTS = {
    22: "E-memorandum",
    23: "Global crisis reaction",
    24: "17-25 December rupture",
    26: "15 July / OHAL shock",
    27: "FX crisis",
    28: "Simsek return",
}

# Short, human-readable display names for `event_kind` values, used on
# axis tick labels. Avoids blind character truncation (e.g. "economic_g"
# from "economic_governance"), which produces broken word fragments.
EVENT_KIND_DISPLAY = {
    "economic": "Economic",
    "economic_governance": "Econ. gov.",
    "mixed": "Mixed",
    "control": "Control",
}

OUT_CSV = BASE / "Inter_Term" / "CSVs" / "step4_shock_heterogeneity_displacement.csv"
OUT_FIG = BASE / "Inter_Term" / "Figures" / "figure_8_inter_term_shock_comparison.png"
OUT_NOTE = BASE / "Inter_Term" / "Notes" / "step4_shock_heterogeneity_results.txt"


def family_from_kind(kind: str) -> str:
    if kind in {"economic", "economic_governance"}:
        return "economic_family"
    return "mixed_control_family"


def load_selected_events() -> pd.DataFrame:
    rows = []
    for term, event_label in SELECTED_EVENTS.items():
        path = BASE / f"Term_{term}" / "CSVs" / f"term_{term}_economic_event_summary.csv"
        df = pd.read_csv(path)
        row = df[df["event_label"].eq(event_label)].iloc[0]
        econ = float(row["econ_salience_delta"]) if pd.notna(row["econ_salience_delta"]) else np.nan
        const = float(row["constitutional_share_delta"]) if pd.notna(row["constitutional_share_delta"]) else np.nan
        sec = float(row["security_share_delta"]) if pd.notna(row["security_share_delta"]) else np.nan
        competing_gain = max(0.0, 0.0 if pd.isna(const) else const) + max(0.0, 0.0 if pd.isna(sec) else sec)
        displacement_index = np.nan if pd.isna(econ) else competing_gain - econ
        ratio = np.nan if pd.isna(econ) else competing_gain / (abs(econ) + 1e-6)
        displacement_flag = bool((not pd.isna(displacement_index)) and (displacement_index > 0.0) and (competing_gain > 0.0))
        econ_post = float(row["econ_salience_post_delta"]) if pd.notna(row.get("econ_salience_post_delta")) else np.nan
        const_post = float(row["constitutional_share_post_delta"]) if pd.notna(row.get("constitutional_share_post_delta")) else np.nan
        sec_post = float(row["security_share_post_delta"]) if pd.notna(row.get("security_share_post_delta")) else np.nan
        competing_post = max(0.0, 0.0 if pd.isna(const_post) else const_post) + max(0.0, 0.0 if pd.isna(sec_post) else sec_post)
        displacement_post = np.nan if pd.isna(econ_post) else competing_post - econ_post
        displacement_persists = bool((not pd.isna(displacement_index)) and (not pd.isna(displacement_post)) and displacement_index > 0.0 and displacement_post > 0.0)
        rows.append(
            {
                "term": term,
                "event_label": event_label,
                "event_kind": row["event_kind"],
                "paper_role": row["paper_role"],
                "family": family_from_kind(str(row["event_kind"])),
                "event_month_effective": row["event_month_effective"],
                "macro_pressure_delta": row["macro_pressure_delta"],
                "econ_salience_delta": econ,
                "constitutional_share_delta": const,
                "security_share_delta": sec,
                "competing_gain": competing_gain,
                "displacement_index": displacement_index,
                "competing_to_econ_ratio": ratio,
                "displacement_flag": displacement_flag,
                "econ_salience_post_delta": econ_post,
                "constitutional_share_post_delta": const_post,
                "security_share_post_delta": sec_post,
                "competing_gain_post": competing_post,
                "displacement_index_post": displacement_post,
                "shock_to_post_persistence": displacement_post - displacement_index if (not pd.isna(displacement_post) and not pd.isna(displacement_index)) else np.nan,
                "displacement_persists": displacement_persists,
                "top_owner_party": row["top_owner_party"],
                "top_attack_party": row["top_attack_party"],
            }
        )
    return pd.DataFrame(rows).sort_values("term").reset_index(drop=True)


def build_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    analyzable = df[df["econ_salience_delta"].notna()].copy()
    return (
        analyzable.groupby("family", as_index=False)
        .agg(
            n_events=("event_label", "count"),
            mean_macro_pressure_delta=("macro_pressure_delta", "mean"),
            mean_econ_salience_delta=("econ_salience_delta", "mean"),
            mean_competing_gain=("competing_gain", "mean"),
            mean_displacement_index=("displacement_index", "mean"),
            displacement_share=("displacement_flag", "mean"),
            displacement_persistence_share=("displacement_persists", "mean"),
        )
        .sort_values("family")
        .reset_index(drop=True)
    )


def plot_panel(events: pd.DataFrame, family: pd.DataFrame) -> None:
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), facecolor="white")
    ax1, ax2, ax3 = axes

    x = np.arange(len(events))
    width = 0.25
    ax1.bar(x - width, events["econ_salience_delta"].fillna(0.0), width=width, color="#dc2626", label="Δ economy")
    ax1.bar(x, events["constitutional_share_delta"].fillna(0.0), width=width, color="#2563eb", label="Δ constitution")
    ax1.bar(x + width, events["security_share_delta"].fillna(0.0), width=width, color="#0f766e", label="Δ security")
    ax1.axhline(0.0, color="#9ca3af", linewidth=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"T{t}" for t in events["term"]])
    ax1.set_ylabel("Shock - pre topic share")
    ax1.set_title("Economy and competing-topic deltas")
    ax1.grid(axis="y", alpha=0.2)
    ax1.legend(frameon=False, fontsize=8)

    colors = ["#b91c1c" if bool(v) else "#334155" for v in events["displacement_flag"]]
    ax2.bar(np.arange(len(events)), events["displacement_index"].fillna(0.0), color=colors, alpha=0.92)
    ax2.axhline(0.0, color="#9ca3af", linewidth=1.0)
    ax2.set_xticks(np.arange(len(events)))
    ax2.set_xticklabels(
        [f"T{row.term}\n{EVENT_KIND_DISPLAY.get(row.event_kind, row.event_kind)}" for row in events.itertuples(index=False)],
        fontsize=8,
    )
    ax2.set_ylabel("Competing gain - economy gain")
    ax2.set_title("Displacement index by event")
    ax2.grid(axis="y", alpha=0.2)

    fam_order = ["economic_family", "mixed_control_family"]
    family = family.set_index("family").reindex(fam_order).reset_index()
    x2 = np.arange(len(family))
    ax3.bar(x2 - 0.15, family["mean_econ_salience_delta"], width=0.3, color="#dc2626", label="Mean Δ economy")
    ax3.bar(x2 + 0.15, family["mean_competing_gain"], width=0.3, color="#475569", label="Mean competing gain")
    ax3.axhline(0.0, color="#9ca3af", linewidth=1.0)
    ax3.set_xticks(x2)
    ax3.set_xticklabels(["Economic", "Mixed / control"])
    ax3.set_ylabel("Family mean")
    ax3.set_title("Shock-family comparison")
    ax3.grid(axis="y", alpha=0.2)
    ax3.legend(frameon=False, fontsize=8)

    fig.suptitle("shock heterogeneity and displacement across discourse networks", fontsize=16, y=0.98)
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_note(events: pd.DataFrame, family: pd.DataFrame) -> None:
    lines = ["Step 4 shock heterogeneity and displacement summary", ""]
    for row in events.itertuples(index=False):
        lines.extend(
            [
                f"Term {row.term} · {row.event_label}",
                f"  Family              : {row.family}",
                f"  Macro pressure Δ    : {row.macro_pressure_delta:.3f}" if pd.notna(row.macro_pressure_delta) else "  Macro pressure Δ    : n/a",
                f"  Economy salience Δ  : {row.econ_salience_delta:.3f}" if pd.notna(row.econ_salience_delta) else "  Economy salience Δ  : n/a",
                f"  Constitution Δ      : {row.constitutional_share_delta:.3f}" if pd.notna(row.constitutional_share_delta) else "  Constitution Δ      : n/a",
                f"  Security Δ          : {row.security_share_delta:.3f}" if pd.notna(row.security_share_delta) else "  Security Δ          : n/a",
                f"  Competing gain      : {row.competing_gain:.3f}",
                f"  Displacement index  : {row.displacement_index:.3f}" if pd.notna(row.displacement_index) else "  Displacement index  : n/a",
                f"  Displacement flag   : {bool(row.displacement_flag)}",
                "",
            ]
        )
    lines.append("Family summary")
    lines.append(family.to_string(index=False))
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    events = load_selected_events()
    family = build_family_summary(events)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    plot_panel(events, family)
    write_note(events, family)
    print(events.to_string(index=False))
    print("\nFamily summary")
    print(family.to_string(index=False))


if __name__ == "__main__":
    main()
