from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from build_economic_shock_study import BASE, plot_robustness


IN_EVENTS = BASE / "Inter_Term" / "CSVs" / "inter_term_economic_event_summary.csv"
IN_NULLS = BASE / "Inter_Term" / "CSVs" / "inter_term_economic_null_tests.csv"
IN_AXIS_VALID = BASE / "Inter_Term" / "CSVs" / "inter_term_economic_axis_validation.csv"

OUT_CSV = BASE / "Inter_Term" / "CSVs" / "step6_robustness_null_models.csv"
OUT_FIG = BASE / "Inter_Term" / "Figures" / "figure_9_robustness_null_models.png"
OUT_NOTE = BASE / "Inter_Term" / "Notes" / "step6_robustness_null_models_results.txt"


def add_z_scores(null_df: pd.DataFrame) -> pd.DataFrame:
    out = null_df.copy()
    for stem in ["econ_salience", "spread", "constitutional", "security"]:
        obs_col = f"{stem}_obs"
        mean_col = f"{stem}_null_mean"
        std_col = f"{stem}_null_std"
        z_col = f"{stem}_zscore"
        if std_col in out.columns:
            denom = out[std_col].replace(0, np.nan)
            out[z_col] = (out[obs_col] - out[mean_col]) / denom
        else:
            out[z_col] = np.nan
    return out


def build_summary_rows(null_df: pd.DataFrame, axis_valid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    analyzable = null_df[null_df["econ_salience_obs"].notna()].copy()
    for metric in ["econ_salience", "spread", "constitutional", "security"]:
        p_col = f"{metric}_null_p"
        z_col = f"{metric}_zscore"
        if p_col not in analyzable.columns:
            continue
        rows.append(
            {
                "summary_type": "null_metric",
                "metric": metric,
                "mean_p_value": float(analyzable[p_col].mean()),
                "min_p_value": float(analyzable[p_col].min()),
                "share_p_leq_0_10": float((analyzable[p_col] <= 0.10).mean()),
                "mean_abs_zscore": float(analyzable[z_col].abs().mean()) if z_col in analyzable.columns else np.nan,
                "max_abs_zscore": float(analyzable[z_col].abs().max()) if z_col in analyzable.columns else np.nan,
            }
        )

    loo = axis_valid[axis_valid["metric"] == "leave_one_anchor_out"]["value"].dropna()
    rows.extend(
        [
            {
                "summary_type": "axis_validation",
                "metric": "observed_anchor_correlation",
                "value": float(axis_valid.loc[axis_valid["metric"] == "observed_anchor_correlation", "value"].iloc[0]),
            },
            {
                "summary_type": "axis_validation",
                "metric": "anchor_permutation_p",
                "value": float(axis_valid.loc[axis_valid["metric"] == "anchor_permutation_p", "value"].iloc[0]),
            },
            {
                "summary_type": "axis_validation",
                "metric": "bootstrap_anchor_corr_mean",
                "value": float(axis_valid.loc[axis_valid["metric"] == "bootstrap_anchor_corr_mean", "value"].iloc[0]),
            },
            {
                "summary_type": "axis_validation",
                "metric": "bootstrap_anchor_corr_std",
                "value": float(axis_valid.loc[axis_valid["metric"] == "bootstrap_anchor_corr_std", "value"].iloc[0]),
            },
            {
                "summary_type": "axis_validation",
                "metric": "leave_one_anchor_out_min",
                "value": float(loo.min()) if len(loo) else np.nan,
            },
            {
                "summary_type": "axis_validation",
                "metric": "leave_one_anchor_out_mean",
                "value": float(loo.mean()) if len(loo) else np.nan,
            },
        ]
    )
    return pd.DataFrame(rows)


def write_note(null_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    core = null_df[null_df["event_label"].isin(["Global crisis reaction", "Albayrak resignation", "FX crisis"])].copy()
    lines = [
        "Step 6 robustness and null model summary",
        "",
        "Core-event observed vs null means",
        core[
            [
                "term",
                "event_label",
                "econ_salience_obs",
                "econ_salience_null_mean",
                "econ_salience_null_p",
                "spread_obs",
                "spread_null_mean",
                "spread_null_p",
            ]
        ].to_string(index=False),
        "",
        "Summary metrics",
        summary_df.to_string(index=False),
    ]
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    events = pd.read_csv(IN_EVENTS)
    null_df = pd.read_csv(IN_NULLS)
    axis_valid = pd.read_csv(IN_AXIS_VALID)

    null_z = add_z_scores(null_df)
    summary_df = build_summary_rows(null_z, axis_valid)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    null_z.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    plot_robustness(events, null_z, axis_valid, OUT_FIG)
    write_note(null_z, summary_df)

    print("Robustness summary")
    print(summary_df.to_string(index=False))
    print("\nNull z-score head")
    print(
        null_z[
            [
                "term",
                "event_label",
                "econ_salience_zscore",
                "spread_zscore",
                "constitutional_zscore",
                "security_zscore",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
