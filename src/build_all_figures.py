"""
build_all_figures.py
────────────────────────────────────────────────────────────────────────────
One-command regeneration of every figure used in the manuscript.

Runs each figure-generating script in order, then copies the resulting PNGs
from Inter_Term/Figures/ into the canonical top-level figures/ folder that the
manuscript references.

Usage (from the repository root):

    python src/build_all_figures.py

All inputs are the processed per-term CSVs already committed under Term_2?/CSVs
and Inter_Term/CSVs, so this runs without the large raw dataset.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

# Figure-generating scripts, in dependency-free order.
SCRIPTS = [
    "regen_figure1.py",                       # inter_term_economic_event_design.png
    "step1_agenda_ownership_panel.py",        # figure_6
    "step2_partisan_distance_triptychs.py",   # figure_3, figure_4, figure_5
    "step3_structural_brokerage_turnover.py", # figure_7
    "step4_shock_heterogeneity_displacement.py",  # figure_8
    "step5_inter_term_economic_latent_axis.py",   # figure_2
    "step6_robustness_null_models.py",        # figure_9
]

# Figures that make up the paper, copied into figures/ after regeneration.
PAPER_FIGURES = [
    "inter_term_economic_event_design.png",
    "figure_2_inter_term_economic_latent_axis.png",
    "figure_3_term_23_shock_triptych.png",
    "figure_4_term_27_shock_triptych.png",
    "figure_5_term_28_shock_triptych.png",
    "figure_6_agenda_ownership_panel.png",
    "figure_7_brokerage_turnover_panel.png",
    "figure_8_inter_term_shock_comparison.png",
    "figure_9_robustness_null_models.png",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"=== running {script} ===", flush=True)
        subprocess.run([sys.executable, str(SRC / script)], check=True, cwd=str(SRC))

    generated = ROOT / "Inter_Term" / "Figures"
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("\n=== syncing figures into figures/ ===", flush=True)
    for name in PAPER_FIGURES:
        src_png = generated / name
        if src_png.exists():
            shutil.copy2(src_png, figures_dir / name)
            print(f"  synced {name}")
        else:
            print(f"  WARNING: {name} was not generated")

    print("\nDone. All paper figures are in figures/.")


if __name__ == "__main__":
    main()
