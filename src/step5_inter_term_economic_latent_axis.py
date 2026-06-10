from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from build_economic_shock_study import BASE, build_axis_validation, build_global_axis, plot_global_interval_axis


OUT_POS = BASE / "Inter_Term" / "CSVs" / "step5_inter_term_economic_latent_axis_positions.csv"
OUT_DRIFT = BASE / "Inter_Term" / "CSVs" / "step5_inter_term_economic_latent_axis_drift.csv"
OUT_WEIGHTS = BASE / "Inter_Term" / "CSVs" / "step5_inter_term_economic_latent_axis_weights.csv"
OUT_VALID = BASE / "Inter_Term" / "CSVs" / "step5_inter_term_economic_latent_axis_validation.csv"
OUT_FIG = BASE / "Inter_Term" / "Figures" / "figure_2_inter_term_economic_latent_axis.png"
OUT_NOTE = BASE / "Inter_Term" / "Notes" / "step5_inter_term_economic_latent_axis_results.txt"


def build_drift_summary(interval_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for party, sub in interval_summary.groupby("party"):
        sub = sub.sort_values("term")
        first = sub.iloc[0]
        last = sub.iloc[-1]
        rows.append(
            {
                "party": party,
                "n_terms": int(sub["term"].nunique()),
                "first_term": int(first["term"]),
                "last_term": int(last["term"]),
                "first_center": float(first["center"]),
                "last_center": float(last["center"]),
                "net_drift": float(last["center"] - first["center"]),
                "mean_center": float(sub["center"].mean()),
                "std_center": float(sub["center"].std(ddof=0)),
                "min_center": float(sub["center"].min()),
                "max_center": float(sub["center"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_terms", "party"], ascending=[False, True]).reset_index(drop=True)


def write_note(
    weights_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    interval_summary: pd.DataFrame,
    drift_df: pd.DataFrame,
) -> None:
    obs_corr = float(validation_df.loc[validation_df["metric"] == "observed_anchor_correlation", "value"].iloc[0])
    perm_p = float(validation_df.loc[validation_df["metric"] == "anchor_permutation_p", "value"].iloc[0])
    boot_mean = float(validation_df.loc[validation_df["metric"] == "bootstrap_anchor_corr_mean", "value"].iloc[0])
    boot_std = float(validation_df.loc[validation_df["metric"] == "bootstrap_anchor_corr_std", "value"].iloc[0])

    anchor_parties = [
        "Adalet ve Kalkınma Partisi",
        "Cumhuriyet Halk Partisi",
        "Milliyetçi Hareket Partisi",
        "Halkların Demokratik Partisi",
        "İYİ Parti",
        "DEM Parti",
    ]
    anchor_rows = interval_summary[interval_summary["party"].isin(anchor_parties)].sort_values(["party", "term"])

    lines = [
        "Step 5 inter-term economic latent axis summary",
        "",
        f"Observed anchor correlation: {obs_corr:.6f}",
        f"Anchor permutation p-value: {perm_p:.6f}",
        f"Bootstrap anchor correlation mean: {boot_mean:.6f}",
        f"Bootstrap anchor correlation std: {boot_std:.6f}",
        "",
        "Axis weights",
        weights_df.to_string(index=False),
        "",
        "Selected party interval centers across terms",
        anchor_rows.to_string(index=False),
        "",
        "Party drift summary",
        drift_df.head(15).to_string(index=False),
    ]
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_POS.parent.mkdir(parents=True, exist_ok=True)
    OUT_DRIFT.parent.mkdir(parents=True, exist_ok=True)
    OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_VALID.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    yearly_pos, _, weights_df, _, _ = build_global_axis()
    validation_df, _ = build_axis_validation()
    interval_summary = plot_global_interval_axis(yearly_pos, OUT_FIG)
    drift_df = build_drift_summary(interval_summary)

    yearly_pos.to_csv(OUT_POS, index=False, encoding="utf-8-sig")
    drift_df.to_csv(OUT_DRIFT, index=False, encoding="utf-8-sig")
    weights_df.to_csv(OUT_WEIGHTS, index=False, encoding="utf-8-sig")
    validation_df.to_csv(OUT_VALID, index=False, encoding="utf-8-sig")
    write_note(weights_df, validation_df, interval_summary, drift_df)

    print("Axis weights")
    print(weights_df.to_string(index=False))
    print("\nSelected drift summary")
    print(drift_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
