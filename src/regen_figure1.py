"""Regenerate only Figure 1 (event design) using the existing registry CSV."""
import sys
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from pathlib import Path
from build_economic_shock_study import plot_event_design, BASE

registry_path = BASE / "Inter_Term" / "CSVs" / "inter_term_economic_event_registry.csv"
out_path = BASE / "Inter_Term" / "Figures" / "inter_term_economic_event_design.png"

registry_df = pd.read_csv(registry_path)
plot_event_design(registry_df, out_path)
print(f"Saved: {out_path}")
