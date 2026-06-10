from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import parser as transcript_parser


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Data" / "diagnostics"


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--txt-folder", type=Path, default=None)
    arg_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arg_parser.add_argument("--workers", type=int, default=None)
    arg_parser.add_argument("--sample-limit", type=int, default=50)
    return arg_parser.parse_args()


def main() -> None:
    args = parse_args()
    txt_folder = args.txt_folder or transcript_parser.resolve_existing_path(
        transcript_parser.DEFAULT_TXT_CANDIDATES,
        "TBMM_Clean_Text",
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "transcript_diagnostics.json"
    csv_path = output_dir / "transcript_diagnostics_summary.csv"

    transcript_parser.run_diagnostics(
        txt_folder=txt_folder,
        output_json=json_path,
        sample_limit=args.sample_limit,
        workers=args.workers,
    )

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "donem",
                "files",
                "speaker_inline",
                "speaker_header_only",
                "uppercase_like_unmatched",
                "first_examples",
            ],
        )
        writer.writeheader()
        for donem in sorted(data):
            row = dict(data[donem])
            row["donem"] = donem
            row["first_examples"] = " || ".join(row.get("examples", [])[:10])
            row.pop("examples", None)
            writer.writerow(row)

    print(f"📄 Özet CSV yazıldı: {csv_path}")


if __name__ == "__main__":
    main()
