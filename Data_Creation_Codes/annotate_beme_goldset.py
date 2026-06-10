from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import fill

import pandas as pd


LABEL_MAP = {
    "p": "positive",
    "m": "mixed",
    "n": "negative",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively annotate the BEME validation gold set.")
    parser.add_argument(
        "--csv",
        default="Term_22/CSVs/term_22_beme_validation_goldset_draft.csv",
        help="Path to the draft gold-set CSV.",
    )
    return parser.parse_args()


def render_row(row: pd.Series, position: int, total: int) -> str:
    lines = [
        "",
        "=" * 88,
        f"[{position}/{total}] {row['candidate_id']} | {row['date']} | {row['source_party']} -> {row['target_party']}",
        f"Heuristic: {row['heuristic_label']} | pos_hits={row['heuristic_positive_hits']} | neg_hits={row['heuristic_negative_hits']} | score={row['heuristic_score']}",
        f"Speaker: {row['speaker']}",
        "-" * 88,
        fill(str(row["window_text"]), width=88),
        "-" * 88,
        "Label? [p=positive, m=mixed, n=negative, s=skip, q=quit]",
    ]
    current_label = str(row.get("manual_label", "")).strip()
    current_notes = str(row.get("manual_notes", "")).strip()
    if current_label:
        lines.insert(4, f"Current manual label: {current_label}")
    if current_notes:
        lines.insert(5, f"Current note: {current_notes}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    frame = pd.read_csv(csv_path)
    if "manual_label" not in frame.columns:
        frame.insert(1, "manual_label", "")
    if "manual_notes" not in frame.columns:
        frame.insert(2, "manual_notes", "")
    frame["manual_label"] = frame["manual_label"].fillna("").astype(str)
    frame["manual_notes"] = frame["manual_notes"].fillna("").astype(str)

    unlabeled_positions = [idx for idx, value in frame["manual_label"].items() if str(value).strip() == ""]
    if not unlabeled_positions:
        print("All rows are already labeled.")
        return

    total = len(frame)
    for idx in unlabeled_positions:
        row = frame.loc[idx]
        print(render_row(row, idx + 1, total), flush=True)
        command = input("> ").strip().lower()
        if command == "q":
            break
        if command == "s":
            continue
        if command not in LABEL_MAP:
            print("Unknown command, skipping.", flush=True)
            continue
        frame.at[idx, "manual_label"] = LABEL_MAP[command]
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved {row['candidate_id']} as {LABEL_MAP[command]}.", flush=True)

    remaining = int((frame["manual_label"].astype(str).str.strip() == "").sum())
    print(f"Remaining unlabeled rows: {remaining}", flush=True)


if __name__ == "__main__":
    main()
