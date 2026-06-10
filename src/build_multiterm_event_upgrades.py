"""
build_multiterm_event_upgrades.py
────────────────────────────────────────────────────────────────────────────
Strengthen Terms 23–28 with term-specific event figures and clean notes.

Outputs per term
  Figures/term_XX_causal_identification.png
  Figures/term_XX_political_dynamics.png
  Notes/term_XX_publication_brief.txt

Cleanup
  - Remove copied Term 22 notes inside Term_23–28/Notes
  - Archive redundant layered metric dashboard figures
"""

from __future__ import annotations

import math
import os
import shutil
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from scipy.stats import norm

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]  # repo root (this file lives in src/)

PARTY_COLORS = {
    "Adalet ve Kalkınma Partisi": "#E63329",
    "Cumhuriyet Halk Partisi": "#E87722",
    "Milliyetçi Hareket Partisi": "#8B0000",
    "Halkların Demokratik Partisi": "#6A0DAD",
    "Halkların Eşitlik ve Demokrasi Partisi": "#6A0DAD",
    "DEM Parti": "#6A0DAD",
    "İYİ Parti": "#00A0E3",
    "YENİ YOL Partisi": "#2CA02C",
    "Bağımsız": "#7F8C8D",
}

PARTY_SHORT = {
    "Adalet ve Kalkınma Partisi": "AKP",
    "Cumhuriyet Halk Partisi": "CHP",
    "Milliyetçi Hareket Partisi": "MHP",
    "Halkların Demokratik Partisi": "HDP",
    "Halkların Eşitlik ve Demokrasi Partisi": "HDP",
    "DEM Parti": "HDP",
    "İYİ Parti": "IYI",
    "YENİ YOL Partisi": "Yeni Yol",
    "Bağımsız": "IND",
}

CAT_COLORS = {
    "macro_economy": "#E87722",
    "constitutional_conflict": "#C0392B",
    "democratic_reform": "#1F77B4",
    "security_conflict": "#7F8C8D",
}

CAT_LABELS = {
    "macro_economy": "Macro Economy",
    "constitutional_conflict": "Constitutional Conflict",
    "democratic_reform": "Democratic Reform",
    "security_conflict": "Security Conflict",
}

METRIC_LABELS = {
    "neg_rate": "Negative intensity",
    "pos_rate": "Positive intensity",
    "agenda_share": "Agenda share",
}

BG = "#FAFAFA"
PBG = "#F5F5F5"


@dataclass
class TestSpec:
    event: str
    party: str
    category: str
    metric: str
    label: str
    pre_window: int = 4
    post_window: int = 4


TERM_META = {
    23: {
        "label": "Term 23 (2007–2011)",
        "title": "Closure Case, Global Crisis, and Referendum Reframing",
        "events": [
            ("2008-07", "AKP closure case", "#C0392B"),
            ("2008-09", "Global crisis", "#2980B9"),
            ("2010-09", "Constitutional referendum", "#27AE60"),
        ],
        "tests": [
            TestSpec("2008-07", "Adalet ve Kalkınma Partisi", "constitutional_conflict", "neg_rate",
                     "AKP constitutional negativity collapses after the closure case"),
            TestSpec("2008-07", "Adalet ve Kalkınma Partisi", "democratic_reform", "pos_rate",
                     "AKP democratic-reform legitimation rises after the closure case"),
            TestSpec("2008-09", "Cumhuriyet Halk Partisi", "macro_economy", "neg_rate",
                     "CHP macroeconomic negativity surges with the global crisis"),
            TestSpec("2008-09", "Adalet ve Kalkınma Partisi", "macro_economy", "neg_rate",
                     "AKP also shifts into macroeconomic negativity after the crisis"),
            TestSpec("2010-09", "Adalet ve Kalkınma Partisi", "democratic_reform", "pos_rate",
                     "AKP referendum campaign leans on democratic-reform positivity"),
        ],
        "event_window_specs": [
            ("Cumhuriyet Halk Partisi", "macro_economy", "neg_rate", "CHP macro neg"),
            ("Adalet ve Kalkınma Partisi", "macro_economy", "neg_rate", "AKP macro neg"),
            ("Adalet ve Kalkınma Partisi", "constitutional_conflict", "neg_rate", "AKP const. neg"),
        ],
        "event_window_event": "2008-09",
        "trajectory_specs": [
            ("Cumhuriyet Halk Partisi", "macro_economy", "neg_rate", "CHP macro neg"),
            ("Adalet ve Kalkınma Partisi", "democratic_reform", "pos_rate", "AKP demo pos"),
            ("Adalet ve Kalkınma Partisi", "constitutional_conflict", "neg_rate", "AKP const. neg"),
        ],
        "dominance_category": "macro_economy",
        "paper_claim": (
            "Term 23 is strongest when read as a reframing sequence: the closure case lowers AKP's "
            "constitutional negativity, the global crisis pushes both AKP and CHP into macroeconomic "
            "attack, and the 2010 referendum restores AKP's democratic-reform legitimation."
        ),
        "targets": "Political Communication, Comparative Political Studies, BJPS (stretch)",
    },
    24: {
        "label": "Term 24 (2011–2015)",
        "title": "Gezi, 17–25 Aralık, and Fragmented Opposition Framing",
        "events": [
            ("2013-05", "Gezi", "#2980B9"),
            ("2013-12", "17–25 Dec", "#C0392B"),
            ("2014-03", "Dershane dispute", "#8E44AD"),
        ],
        "tests": [
            TestSpec("2013-05", "Milliyetçi Hareket Partisi", "security_conflict", "neg_rate",
                     "MHP securitizes Gezi more aggressively after May 2013"),
            TestSpec("2013-05", "Adalet ve Kalkınma Partisi", "democratic_reform", "pos_rate",
                     "AKP answers Gezi with higher democratic-reform positivity"),
            TestSpec("2013-12", "Milliyetçi Hareket Partisi", "macro_economy", "neg_rate",
                     "MHP converts 17–25 into macroeconomic attack"),
            TestSpec("2013-12", "Adalet ve Kalkınma Partisi", "constitutional_conflict", "neg_rate",
                     "AKP shifts into constitutional conflict after 17–25"),
            TestSpec("2014-03", "Cumhuriyet Halk Partisi", "macro_economy", "neg_rate",
                     "CHP pivots from constitutional to economic attack during the dershane rupture"),
            TestSpec("2014-03", "Adalet ve Kalkınma Partisi", "democratic_reform", "pos_rate",
                     "AKP democratic-reform positivity fades in the dershane conflict"),
        ],
        "event_window_specs": [
            ("Milliyetçi Hareket Partisi", "macro_economy", "neg_rate", "MHP macro neg"),
            ("Adalet ve Kalkınma Partisi", "constitutional_conflict", "neg_rate", "AKP const. neg"),
            ("Cumhuriyet Halk Partisi", "security_conflict", "neg_rate", "CHP security neg"),
        ],
        "event_window_event": "2013-12",
        "trajectory_specs": [
            ("Milliyetçi Hareket Partisi", "security_conflict", "neg_rate", "MHP security neg"),
            ("Adalet ve Kalkınma Partisi", "constitutional_conflict", "neg_rate", "AKP const. neg"),
            ("Cumhuriyet Halk Partisi", "macro_economy", "neg_rate", "CHP macro neg"),
        ],
        "dominance_category": "security_conflict",
        "paper_claim": (
            "Term 24 does not support a single aggregate shock story. It is stronger as a fragmented "
            "opposition period: Gezi hardens MHP's security frame, 17–25 splits the field between AKP's "
            "constitutional defense and MHP's economic attack, and the dershane rupture moves CHP further "
            "into macroeconomic opposition."
        ),
        "targets": "Political Communication, Party Politics, Government and Opposition",
    },
    25: {
        "label": "Term 25 (2015 Interregnum)",
        "title": "Security Compression and Pre-Election Economic Recentering",
        "events": [
            ("2015-07", "Violence escalation", "#C0392B"),
            ("2015-09", "Conflict peak", "#7F8C8D"),
            ("2015-10", "Pre-Nov election", "#27AE60"),
        ],
        "tests": [
            TestSpec("2015-10", "Adalet ve Kalkınma Partisi", "macro_economy", "agenda_share",
                     "AKP sharply recenters macroeconomy before the Nov. 1 election", pre_window=2, post_window=1),
            TestSpec("2015-10", "Adalet ve Kalkınma Partisi", "macro_economy", "pos_rate",
                     "AKP's macroeconomic positivity falls as it recenters the agenda", pre_window=2, post_window=1),
            TestSpec("2015-10", "Adalet ve Kalkınma Partisi", "democratic_reform", "pos_rate",
                     "AKP's democratic-reform positivity also drops before the election", pre_window=2, post_window=1),
        ],
        "event_window_specs": [
            ("Adalet ve Kalkınma Partisi", "macro_economy", "agenda_share", "AKP macro agenda"),
            ("Adalet ve Kalkınma Partisi", "democratic_reform", "pos_rate", "AKP demo pos"),
            ("Adalet ve Kalkınma Partisi", "security_conflict", "agenda_share", "AKP security agenda"),
        ],
        "event_window_event": "2015-10",
        "trajectory_specs": [
            ("Adalet ve Kalkınma Partisi", "security_conflict", "agenda_share", "AKP security agenda"),
            ("Adalet ve Kalkınma Partisi", "macro_economy", "agenda_share", "AKP macro agenda"),
            ("Milliyetçi Hareket Partisi", "security_conflict", "agenda_share", "MHP security agenda"),
        ],
        "dominance_category": "security_conflict",
        "paper_claim": (
            "Term 25 is too short for standard long-window inference, but it yields a sharp interregnum note: "
            "July and September are security-compressed months, whereas October re-centers AKP's agenda on "
            "macroeconomy before the Nov. 1 election while reformist positivity collapses."
        ),
        "targets": "Electoral Studies (research note), Turkish Studies, Party Politics (short-format angle)",
    },
    26: {
        "label": "Term 26 (2015–2018)",
        "title": "Coup, OHAL, HDP Compression, and Constitutionalization",
        "events": [
            ("2016-07", "15 July coup attempt", "#C0392B"),
            ("2016-11", "HDP arrests", "#8E44AD"),
            ("2017-04", "Constitutional referendum", "#27AE60"),
        ],
        "tests": [
            TestSpec("2016-07", "Halkların Demokratik Partisi", "macro_economy", "neg_rate",
                     "HDP macroeconomic negativity spikes after the coup attempt"),
            TestSpec("2016-07", "Adalet ve Kalkınma Partisi", "democratic_reform", "neg_rate",
                     "AKP's democratic-reform rhetoric turns markedly more negative after the coup"),
            TestSpec("2016-11", "Adalet ve Kalkınma Partisi", "constitutional_conflict", "neg_rate",
                     "AKP constitutional negativity falls after HDP arrests as repression normalizes"),
            TestSpec("2017-04", "Cumhuriyet Halk Partisi", "constitutional_conflict", "pos_rate",
                     "CHP referendum campaign invests in constitutional legitimation"),
            TestSpec("2016-07", "Halkların Demokratik Partisi", "macro_economy", "agenda_share",
                     "HDP's macroeconomic agenda expands sharply under OHAL shock"),
        ],
        "event_window_specs": [
            ("Halkların Demokratik Partisi", "macro_economy", "neg_rate", "HDP macro neg"),
            ("Adalet ve Kalkınma Partisi", "democratic_reform", "neg_rate", "AKP demo neg"),
            ("Cumhuriyet Halk Partisi", "macro_economy", "neg_rate", "CHP macro neg"),
        ],
        "event_window_event": "2016-07",
        "trajectory_specs": [
            ("Halkların Demokratik Partisi", "macro_economy", "agenda_share", "HDP macro agenda"),
            ("Adalet ve Kalkınma Partisi", "democratic_reform", "neg_rate", "AKP demo neg"),
            ("Cumhuriyet Halk Partisi", "constitutional_conflict", "agenda_share", "CHP const. agenda"),
        ],
        "dominance_category": "constitutional_conflict",
        "paper_claim": (
            "Term 26 is strongest as an authoritarian consolidation sequence. The coup and OHAL compress "
            "pluralism, intensify HDP's macroeconomic distress language, turn AKP's democratic-reform talk "
            "more negative, and then constitutionalize conflict during the referendum campaign."
        ),
        "targets": "Democratization, Journal of Democracy, Comparative Political Studies",
    },
    27: {
        "label": "Term 27 (2018–2023)",
        "title": "Presidential Atrophy, Albayrak Shock, and the FX-Crisis Narrative War",
        "events": [
            ("2020-11", "Albayrak resignation", "#8E44AD"),
            ("2022-01", "FX crisis", "#C0392B"),
            ("2023-05", "Election", "#27AE60"),
        ],
        "tests": [
            TestSpec("2020-11", "Cumhuriyet Halk Partisi", "constitutional_conflict", "neg_rate",
                     "CHP constitutional negativity doubles after the Albayrak break"),
            TestSpec("2020-11", "Adalet ve Kalkınma Partisi", "constitutional_conflict", "neg_rate",
                     "AKP itself also shifts toward constitutional conflict after Albayrak"),
            TestSpec("2022-01", "Cumhuriyet Halk Partisi", "macro_economy", "neg_rate",
                     "CHP macroeconomic negativity intensifies in the FX crisis"),
            TestSpec("2022-01", "İYİ Parti", "macro_economy", "agenda_share",
                     "IYI Party dramatically expands its macroeconomic agenda in the FX crisis"),
            TestSpec("2022-01", "Adalet ve Kalkınma Partisi", "security_conflict", "neg_rate",
                     "AKP security negativity softens as the agenda shifts back to the economy"),
        ],
        "event_window_specs": [
            ("Cumhuriyet Halk Partisi", "macro_economy", "neg_rate", "CHP macro neg"),
            ("Adalet ve Kalkınma Partisi", "macro_economy", "pos_rate", "AKP macro pos"),
            ("İYİ Parti", "macro_economy", "agenda_share", "IYI macro agenda"),
        ],
        "event_window_event": "2022-01",
        "trajectory_specs": [
            ("Cumhuriyet Halk Partisi", "macro_economy", "neg_rate", "CHP macro neg"),
            ("Adalet ve Kalkınma Partisi", "macro_economy", "pos_rate", "AKP macro pos"),
            ("İYİ Parti", "macro_economy", "agenda_share", "IYI macro agenda"),
        ],
        "dominance_category": "macro_economy",
        "paper_claim": (
            "Term 27 is most convincing as a monetary narrative war under presidential atrophy. Parliament's "
            "overall weight falls, yet Albayrak's resignation reactivates constitutional attack and the 2022 "
            "FX crisis reallocates opposition speech toward macroeconomic attack."
        ),
        "targets": "Party Politics, Comparative Politics, Political Economy journal outlets",
    },
    28: {
        "label": "Term 28 (2023–2026)",
        "title": "Local-Election Backlash, Judicial Shock, and Opposition Lane-Splitting",
        "events": [
            ("2024-03", "Local elections", "#27AE60"),
            ("2025-03", "İmamoğlu case", "#C0392B"),
            ("2025-04", "Post-case protests", "#2980B9"),
        ],
        "tests": [
            TestSpec("2024-03", "Milliyetçi Hareket Partisi", "security_conflict", "neg_rate",
                     "MHP security negativity surges after the local-election shock"),
            TestSpec("2025-03", "Adalet ve Kalkınma Partisi", "constitutional_conflict", "neg_rate",
                     "AKP constitutional negativity rises after the İmamoğlu case"),
            TestSpec("2025-03", "Cumhuriyet Halk Partisi", "democratic_reform", "agenda_share",
                     "CHP expands its democratic-reform agenda after the judicial shock"),
            TestSpec("2025-03", "İYİ Parti", "constitutional_conflict", "agenda_share",
                     "IYI Party shifts into constitutional conflict after the İmamoğlu shock"),
            TestSpec("2025-04", "Cumhuriyet Halk Partisi", "macro_economy", "neg_rate",
                     "CHP macroeconomic attack intensifies during the protest month"),
            TestSpec("2025-04", "Adalet ve Kalkınma Partisi", "macro_economy", "pos_rate",
                     "AKP macroeconomic positivity retreats after the protest escalation"),
        ],
        "event_window_specs": [
            ("Adalet ve Kalkınma Partisi", "constitutional_conflict", "neg_rate", "AKP const. neg"),
            ("Cumhuriyet Halk Partisi", "democratic_reform", "agenda_share", "CHP demo agenda"),
            ("İYİ Parti", "constitutional_conflict", "agenda_share", "IYI const. agenda"),
        ],
        "event_window_event": "2025-03",
        "trajectory_specs": [
            ("Adalet ve Kalkınma Partisi", "constitutional_conflict", "neg_rate", "AKP const. neg"),
            ("Cumhuriyet Halk Partisi", "democratic_reform", "agenda_share", "CHP demo agenda"),
            ("İYİ Parti", "constitutional_conflict", "agenda_share", "IYI const. agenda"),
        ],
        "dominance_category": "constitutional_conflict",
        "paper_claim": (
            "Term 28 is strongest as a judicial-backlash period. Local-election losses destabilize the ruling "
            "coalition, the İmamoğlu case pushes AKP into constitutional conflict, and the opposition splits "
            "into distinct democratic-reform and constitutional lanes rather than a single undifferentiated bloc."
        ),
        "targets": "Electoral Studies, Journal of Democracy, Comparative Political Studies",
    },
}


def setup_ax(ax):
    ax.set_facecolor(PBG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def short_party(name: str) -> str:
    return PARTY_SHORT.get(name, name.replace(" Partisi", "")[:12])


def wrap_text(text: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def load_term_data(term: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_dir = BASE / f"Term_{term}" / "CSVs"
    concept = pd.read_csv(csv_dir / f"term_{term}_beme_concept_edges.csv")
    speech = pd.read_csv(csv_dir / f"term_{term}_beme_speech_labels.csv")
    concept["date"] = pd.to_datetime(concept["date"])
    concept["ym"] = concept["date"].dt.to_period("M")
    speech["date"] = pd.to_datetime(speech["date"])
    speech["ym"] = speech["date"].dt.to_period("M")
    return concept, speech


def build_monthly_panel(concept: pd.DataFrame) -> pd.DataFrame:
    party_total = (
        concept.groupby(["ym", "party"], as_index=False)["mention_count"]
        .sum()
        .rename(columns={"mention_count": "party_total"})
    )
    by_cat = (
        concept.groupby(["ym", "party", "concept_category"], as_index=False)
        .agg(
            mention_count=("mention_count", "sum"),
            pos_hits=("beme_positive_hits", "sum"),
            neg_hits=("beme_negative_hits", "sum"),
        )
    )
    panel = by_cat.merge(party_total, on=["ym", "party"], how="left")
    panel["agenda_share"] = panel["mention_count"] / panel["party_total"].clip(lower=1)
    panel["pos_rate"] = panel["pos_hits"] / panel["mention_count"].clip(lower=1)
    panel["neg_rate"] = panel["neg_hits"] / panel["mention_count"].clip(lower=1)
    return panel.sort_values(["ym", "party", "concept_category"]).reset_index(drop=True)


def aggregate_series(panel: pd.DataFrame) -> pd.DataFrame:
    total = panel.groupby("ym", as_index=False)["mention_count"].sum().rename(columns={"mention_count": "all_mentions"})
    agg = (
        panel.groupby(["ym", "concept_category"], as_index=False)
        .agg(
            mention_count=("mention_count", "sum"),
            pos_hits=("pos_hits", "sum"),
            neg_hits=("neg_hits", "sum"),
        )
        .merge(total, on="ym", how="left")
    )
    agg["agenda_share"] = agg["mention_count"] / agg["all_mentions"].clip(lower=1)
    out = agg.pivot(index="ym", columns="concept_category", values="agenda_share").fillna(0)
    return out.reset_index()


def demo_stress_series(panel: pd.DataFrame) -> pd.DataFrame:
    pivot = panel.groupby(["ym", "concept_category"], as_index=False).agg(
        neg_hits=("neg_hits", "sum"),
        pos_hits=("pos_hits", "sum"),
    )
    cc = pivot[pivot["concept_category"] == "constitutional_conflict"][["ym", "neg_hits"]].rename(columns={"neg_hits": "cc_neg"})
    dr = pivot[pivot["concept_category"] == "democratic_reform"][["ym", "pos_hits"]].rename(columns={"pos_hits": "dr_pos"})
    sec = pivot[pivot["concept_category"] == "security_conflict"][["ym", "neg_hits"]].rename(columns={"neg_hits": "sec_neg"})
    ds = cc.merge(dr, on="ym", how="outer").merge(sec, on="ym", how="outer").fillna(0)
    ds["demo_stress"] = (ds["cc_neg"] + 0.5 * ds["sec_neg"]) / (ds["dr_pos"] + 0.01)
    return ds.sort_values("ym")


def speaker_concentration(speech: pd.DataFrame) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    counts = speech.groupby("speaker").size().sort_values()
    vals = counts.values.astype(float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, np.array([0, 1]), np.array([0, 1])
    cum = np.cumsum(vals)
    lorenz_y = np.concatenate([[0.0], cum / cum[-1]])
    lorenz_x = np.linspace(0, 1, len(vals) + 1)
    area = np.trapezoid(lorenz_y, lorenz_x)
    gini = 1 - 2 * area
    desc = counts.sort_values(ascending=False)
    top10 = desc.head(max(1, math.ceil(len(desc) * 0.10))).sum() / desc.sum()
    top20 = desc.head(max(1, math.ceil(len(desc) * 0.20))).sum() / desc.sum()
    return gini, top10, top20, lorenz_x, lorenz_y


def _window_months(panel: pd.DataFrame, event: str, pre_window: int, post_window: int) -> tuple[list[pd.Period], list[pd.Period]]:
    months = sorted(panel["ym"].unique())
    event_period = pd.Period(event, "M")
    pre = [m for m in months if m < event_period][-pre_window:]
    post = [m for m in months if m >= event_period][:post_window]
    return pre, post


def event_test(panel: pd.DataFrame, spec: TestSpec) -> dict:
    sub = panel[(panel["party"] == spec.party) & (panel["concept_category"] == spec.category)].copy()
    pre, post = _window_months(panel, spec.event, spec.pre_window, spec.post_window)
    pre_df = sub[sub["ym"].isin(pre)]
    post_df = sub[sub["ym"].isin(post)]
    if pre_df.empty or post_df.empty:
        return {"ok": False, "label": spec.label, "reason": "empty-window"}

    if spec.metric == "agenda_share":
        c1 = float(pre_df["mention_count"].sum())
        e1 = float(pre_df["party_total"].sum())
        c2 = float(post_df["mention_count"].sum())
        e2 = float(post_df["party_total"].sum())
    elif spec.metric == "pos_rate":
        c1 = float(pre_df["pos_hits"].sum())
        e1 = float(pre_df["mention_count"].sum())
        c2 = float(post_df["pos_hits"].sum())
        e2 = float(post_df["mention_count"].sum())
    elif spec.metric == "neg_rate":
        c1 = float(pre_df["neg_hits"].sum())
        e1 = float(pre_df["mention_count"].sum())
        c2 = float(post_df["neg_hits"].sum())
        e2 = float(post_df["mention_count"].sum())
    else:
        raise ValueError(f"Unknown metric: {spec.metric}")

    if min(c1, c2, e1, e2) <= 0:
        return {"ok": False, "label": spec.label, "reason": "non-positive"}

    pre_rate = c1 / e1
    post_rate = c2 / e2
    rr = post_rate / pre_rate
    se = math.sqrt((1 / c1) + (1 / c2))
    z = math.log(rr) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return {
        "ok": True,
        "label": spec.label,
        "event": spec.event,
        "party": spec.party,
        "category": spec.category,
        "metric": spec.metric,
        "pre_rate": pre_rate,
        "post_rate": post_rate,
        "rr": rr,
        "log_rr": math.log(rr),
        "z": z,
        "p": p,
        "pre_months": pre,
        "post_months": post,
    }


def normalized_event_series(panel: pd.DataFrame, party: str, category: str, metric: str, event: str, window: int = 4):
    sub = panel[(panel["party"] == party) & (panel["concept_category"] == category)].sort_values("ym").copy()
    if sub.empty:
        return None
    event_period = pd.Period(event, "M")
    months = list(sub["ym"].unique())
    if event_period not in months:
        return None
    i0 = months.index(event_period)
    lo = max(0, i0 - window)
    hi = min(len(months) - 1, i0 + window)
    selected = months[lo:hi + 1]
    s = sub[sub["ym"].isin(selected)][["ym", metric]].drop_duplicates("ym").copy()
    s["rel"] = s["ym"].apply(lambda x: months.index(x) - i0)
    pre_mean = s[s["rel"] < 0][metric].mean()
    if pd.isna(pre_mean) or pre_mean == 0:
        pre_mean = 0.001
    s["norm"] = s[metric] / pre_mean
    return s.sort_values("rel")


def yearly_party_dominance(panel: pd.DataFrame, category: str) -> pd.DataFrame:
    df = panel[panel["concept_category"] == category].copy()
    df["year"] = df["ym"].dt.year
    out = (
        df.groupby(["year", "party"], as_index=False)["mention_count"]
        .sum()
        .rename(columns={"mention_count": "mentions"})
    )
    total = out.groupby("year", as_index=False)["mentions"].sum().rename(columns={"mentions": "year_total"})
    out = out.merge(total, on="year", how="left")
    out["share"] = out["mentions"] / out["year_total"].clip(lower=1)
    return out


def monthly_party_share(panel: pd.DataFrame, category: str) -> pd.DataFrame:
    df = panel[panel["concept_category"] == category].copy()
    out = (
        df.groupby(["ym", "party"], as_index=False)["mention_count"]
        .sum()
        .rename(columns={"mention_count": "mentions"})
    )
    total = out.groupby("ym", as_index=False)["mentions"].sum().rename(columns={"mentions": "month_total"})
    out = out.merge(total, on="ym", how="left")
    out["share"] = out["mentions"] / out["month_total"].clip(lower=1)
    return out


def add_event_lines(ax, events: list[tuple[str, str, str]]):
    for ym, label, color in events:
        ts = pd.Period(ym, "M").to_timestamp()
        ax.axvline(ts, color=color, lw=1.2, ls="--", alpha=0.75)
        ymin, ymax = ax.get_ylim()
        ypos = ymin + 0.9 * (ymax - ymin)
        ax.text(
            ts,
            ypos,
            label,
            ha="center",
            va="center",
            fontsize=7,
            color=color,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2),
        )


def format_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=30, labelsize=8)


def plot_causal(term: int, panel: pd.DataFrame, speech: pd.DataFrame, meta: dict, fig_path: Path):
    tests = [event_test(panel, spec) for spec in meta["tests"]]
    tests = [t for t in tests if t["ok"]]
    agg = aggregate_series(panel)
    ds = demo_stress_series(panel)
    x_agg = agg["ym"].dt.to_timestamp()
    x_ds = ds["ym"].dt.to_timestamp()

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"{meta['label']} — Causal Identification", fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25,
                           left=0.06, right=0.97, top=0.92, bottom=0.08)

    ax = fig.add_subplot(gs[0, 0])
    setup_ax(ax)
    for cat in ["macro_economy", "constitutional_conflict", "democratic_reform", "security_conflict"]:
        if cat in agg.columns:
            ax.plot(x_agg, agg[cat], lw=2.0, color=CAT_COLORS[cat], label=CAT_LABELS[cat])
    add_event_lines(ax, meta["events"])
    ax.set_title("A. Aggregate agenda reallocation across core frames", fontsize=11, fontweight="bold")
    ax.set_ylabel("Share of all mentions")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    format_time_axis(ax)

    ax = fig.add_subplot(gs[0, 1])
    setup_ax(ax)
    sorted_tests = sorted(tests, key=lambda x: abs(x["log_rr"]), reverse=True)
    labels = [t["label"] for t in sorted_tests]
    vals = [t["log_rr"] for t in sorted_tests]
    colors = ["#C0392B" if v > 0 else "#1F77B4" for v in vals]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_text(lbl, 38) for lbl in labels], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Log rate ratio (post / pre)")
    ax.set_title("B. Strongest event-specific shifts", fontsize=11, fontweight="bold")
    for i, t in enumerate(sorted_tests):
        ax.text(
            vals[i] + (0.03 if vals[i] >= 0 else -0.03),
            i,
            f"RR={t['rr']:.2f}, p={t['p']:.3g}",
            va="center",
            ha="left" if vals[i] >= 0 else "right",
            fontsize=8,
            fontweight="bold",
        )

    ax = fig.add_subplot(gs[1, 0])
    setup_ax(ax)
    any_series = False
    for party, category, metric, label in meta["event_window_specs"]:
        s = normalized_event_series(panel, party, category, metric, meta["event_window_event"], window=4)
        if s is None or s.empty:
            continue
        color = PARTY_COLORS.get(party, "#555555")
        ax.plot(s["rel"], s["norm"], marker="o", ms=5, lw=2, color=color, label=label)
        any_series = True
    if any_series:
        ax.axvline(0, color="black", lw=1.3, ls="--")
        ax.axhline(1, color="gray", lw=1, ls=":")
        ax.set_xlabel("Months relative to event")
        ax.set_ylabel("Normalized to pre-event mean")
    ax.set_title(f"C. Event window around {meta['event_window_event']}", fontsize=11, fontweight="bold")
    if any_series:
        ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    setup_ax(ax)
    ax.plot(x_ds, ds["demo_stress"], color="#8E44AD", lw=2.2)
    ax.fill_between(x_ds, ds["demo_stress"], color="#8E44AD", alpha=0.18)
    add_event_lines(ax, meta["events"])
    peak_i = ds["demo_stress"].idxmax()
    peak_row = ds.loc[peak_i]
    peak_x = peak_row["ym"].to_timestamp()
    peak_y = peak_row["demo_stress"]
    ax.scatter([peak_x], [peak_y], s=60, color="#C0392B", zorder=5)
    ax.annotate(
        f"Peak = {peak_row['ym']}\n{peak_y:.2f}",
        xy=(peak_x, peak_y),
        xytext=(peak_x, peak_y * 0.8 if peak_y > 0 else peak_y + 0.5),
        fontsize=8,
        bbox=dict(fc="white", ec="gray", alpha=0.8, pad=2),
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.set_title("D. Democratic stress index", fontsize=11, fontweight="bold")
    ax.set_ylabel("(constitutional neg + 0.5×security neg) / democratic pos")
    format_time_axis(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fig_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return tests


def plot_dynamics(term: int, panel: pd.DataFrame, speech: pd.DataFrame, meta: dict, fig_path: Path):
    gini, top10, top20, lx, ly = speaker_concentration(speech)
    agg = aggregate_series(panel)
    x_agg = agg["ym"].dt.to_timestamp()
    dominance = yearly_party_dominance(panel, meta["dominance_category"])
    monthly_dom = monthly_party_share(panel, meta["dominance_category"])

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"{meta['label']} — Political Dynamics", fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25,
                           left=0.06, right=0.97, top=0.92, bottom=0.08)

    ax = fig.add_subplot(gs[0, 0])
    setup_ax(ax)
    for party, category, metric, label in meta["trajectory_specs"]:
        sub = panel[(panel["party"] == party) & (panel["concept_category"] == category)].sort_values("ym")
        if sub.empty:
            continue
        ax.plot(sub["ym"].dt.to_timestamp(), sub[metric], lw=2, label=label, color=PARTY_COLORS.get(party, "#444"))
    add_event_lines(ax, meta["events"])
    ax.set_title("A. Rival narrative trajectories", fontsize=11, fontweight="bold")
    ax.set_ylabel("Monthly metric value")
    ax.legend(fontsize=8, loc="best")
    format_time_axis(ax)

    ax = fig.add_subplot(gs[0, 1])
    setup_ax(ax)
    if term == 25:
        months = sorted(monthly_dom["ym"].unique())
        bottom = np.zeros(len(months))
        top_parties = monthly_dom.groupby("party")["mentions"].sum().sort_values(ascending=False).head(4).index.tolist()
        x_months = [m.to_timestamp() for m in months]
        for party in top_parties:
            sub = monthly_dom[monthly_dom["party"] == party].set_index("ym").reindex(months, fill_value=0)
            vals = sub["share"].values
            ax.bar(x_months, vals, bottom=bottom, color=PARTY_COLORS.get(party, "#999"),
                   label=short_party(party), width=20)
            bottom += vals
        ax.set_title(f"B. Monthly dominance in {CAT_LABELS[meta['dominance_category']]}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Share of category mentions")
        format_time_axis(ax)
        ax.legend(fontsize=8, ncol=2, loc="upper left")
    else:
        years = sorted(dominance["year"].unique())
        bottom = np.zeros(len(years))
        top_parties = dominance.groupby("party")["mentions"].sum().sort_values(ascending=False).head(5).index.tolist()
        for party in top_parties:
            sub = dominance[dominance["party"] == party].set_index("year").reindex(years, fill_value=0)
            vals = sub["share"].values
            ax.bar(years, vals, bottom=bottom, color=PARTY_COLORS.get(party, "#999"), label=short_party(party), width=0.7)
            bottom += vals
        ax.set_title(f"B. Yearly dominance in {CAT_LABELS[meta['dominance_category']]}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Share of category mentions")
        ax.set_xticks(years)
        ax.legend(fontsize=8, ncol=2, loc="upper left")

    ax = fig.add_subplot(gs[1, 0])
    setup_ax(ax)
    ax.plot(lx, ly, color="#2196F3", lw=2.2, label=f"Lorenz (Gini={gini:.3f})")
    ax.fill_between(lx, ly, lx, color="#2196F3", alpha=0.22)
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", alpha=0.6, label="Equality")
    ax.text(
        0.05,
        0.75,
        f"Top 10% speakers: {top10:.1%}\nTop 20% speakers: {top20:.1%}",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        bbox=dict(fc="white", ec="gray", alpha=0.8, pad=3),
    )
    ax.set_title("C. Speaker concentration in parliamentary speech", fontsize=11, fontweight="bold")
    ax.set_xlabel("Cumulative speaker share")
    ax.set_ylabel("Cumulative speech share")
    ax.legend(fontsize=8, loc="lower right")

    ax = fig.add_subplot(gs[1, 1])
    setup_ax(ax)
    ax.axis("off")
    headline = meta["paper_claim"]
    target = meta["targets"]
    summary_lines = [
        meta["title"],
        "",
        wrap_text(headline, 92),
        "",
        wrap_text(f"Best current target outlets: {target}", 92),
        "",
        "Figure logic:",
        "1. Identify the strongest event-driven shifts.",
        "2. Show who owns the dominant frame each year.",
        "3. Keep speaker concentration in view so the network story stays central.",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        fontsize=10,
        linespacing=1.5,
        bbox=dict(fc="white", ec="#BBBBBB", alpha=0.95, pad=8),
    )
    ax.set_title("D. Standalone paper framing", fontsize=11, fontweight="bold", loc="left")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fig_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def write_brief(term: int, meta: dict, tests: list[dict], notes_dir: Path):
    lines = [
        f"{meta['label']} — Publication Brief",
        "=" * 72,
        "",
        f"Core claim: {meta['paper_claim']}",
        "",
        f"Target outlets: {meta['targets']}",
        "",
        "Strongest event findings:",
    ]
    for t in sorted(tests, key=lambda x: abs(x["log_rr"]), reverse=True):
        direction = "up" if t["rr"] > 1 else "down"
        lines.append(
            f"- {t['label']}: {direction} (RR={t['rr']:.2f}, p={t['p']:.3g}, "
            f"{t['pre_rate']:.3f}->{t['post_rate']:.3f}; event={t['event']})."
        )
    lines.extend(
        [
            "",
            "Active figures to use first:",
            f"- Figures/term_{term}_causal_identification.png",
            f"- Figures/term_{term}_political_dynamics.png",
            f"- Figures/term_{term}_comprehensive_analysis.png",
            f"- Figures/term_{term}_institutional_conflict.png",
            "",
            "Auxiliary visuals kept active:",
            f"- Figures/term_{term}_layered_bipartite_networks.png",
            f"- Figures/term_{term}_negative_yearly_networks.png",
            f"- Figures/term_{term}_positive_yearly_networks.png",
            "",
            "Notes:",
            "- `neg_rate` / `pos_rate` are sentiment-hit intensities over concept mentions.",
            "- `agenda_share` is the concept's share within a party's total monthly agenda.",
            "- Term 25 is intentionally framed as a short interregnum note, not a long-window panel paper.",
        ]
    )
    out = notes_dir / f"term_{term}_publication_brief.txt"
    out.write_text("\n".join(lines), encoding="utf-8")


def cleanup_term(term: int):
    notes_dir = BASE / f"Term_{term}" / "Notes"
    figs_dir = BASE / f"Term_{term}" / "Figures"
    arch_dir = figs_dir / "archive"
    arch_dir.mkdir(parents=True, exist_ok=True)

    for note_name in ["term_22_output_index.txt", "term_22_stage1_stage2_summary.txt"]:
        p = notes_dir / note_name
        if p.exists():
            p.unlink()

    metric_dash = figs_dir / f"term_{term}_layered_metric_dashboard.png"
    if metric_dash.exists():
        shutil.move(str(metric_dash), str(arch_dir / metric_dash.name))


def build_term(term: int):
    meta = TERM_META[term]
    concept, speech = load_term_data(term)
    panel = build_monthly_panel(concept)
    figs_dir = BASE / f"Term_{term}" / "Figures"
    notes_dir = BASE / f"Term_{term}" / "Notes"
    figs_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    tests = plot_causal(term, panel, speech, meta, figs_dir / f"term_{term}_causal_identification.png")
    plot_dynamics(term, panel, speech, meta, figs_dir / f"term_{term}_political_dynamics.png")
    write_brief(term, meta, tests, notes_dir)
    cleanup_term(term)
    return tests


def main():
    for term in [23, 24, 25, 26, 27, 28]:
        tests = build_term(term)
        print(f"Term {term}: wrote upgraded figures and brief ({len(tests)} event findings).")


if __name__ == "__main__":
    main()
