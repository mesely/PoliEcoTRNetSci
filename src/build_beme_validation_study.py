from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = PROJECT_ROOT / "Other" / ".mplconfig"
CACHE_DIR = PROJECT_ROOT / "Other" / ".cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC


TERM = "22"
VALIDATION_PREFIX = "beme_validation"
TERM_ROOT = PROJECT_ROOT / f"Term_{TERM}"
CSV_DIR = TERM_ROOT / "CSVs"
FIG_DIR = TERM_ROOT / "Figures"
NOTE_DIR = TERM_ROOT / "Notes"
GOLD_PATH = CSV_DIR / f"term_{TERM}_{VALIDATION_PREFIX}_goldset_final.csv"
BASELINE_EXCLUDE_PREFIXES = (
    f"term_{TERM}_{VALIDATION_PREFIX}",
    f"term_{TERM}_baseline",
)
CLASS_ORDER = ["negative", "positive"]   # BINARY — mixed excluded from evaluation
RANDOM_SEED = 22
BOOTSTRAP_ITERATIONS = 2000
BERT_MODEL_ID = "savasy/bert-base-turkish-sentiment-cased"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the Term 22 baseline and run the BEME validation study.")
    parser.add_argument("--term", default=TERM, help="TBMM term number.")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_ITERATIONS, help="Bootstrap iterations.")
    parser.add_argument("--try-bert", action="store_true", help="Attempt to evaluate an off-the-shelf BERTurk sentiment model.")
    return parser.parse_args()


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_baseline(term: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = PROJECT_ROOT / f"Term_{term}"
    records: list[dict[str, Any]] = []
    for folder_name in ["CSVs", "Figures", "Notes"]:
        folder = root / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            if path.name.startswith(BASELINE_EXCLUDE_PREFIXES):
                continue
            records.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_of_file(path),
                }
            )
    manifest = pd.DataFrame(records).sort_values("relative_path").reset_index(drop=True)
    summary = {
        "term": term,
        "file_count": int(len(manifest)),
        "csv_count": int(manifest["relative_path"].str.startswith("CSVs/").sum()),
        "figure_count": int(manifest["relative_path"].str.startswith("Figures/").sum()),
        "note_count": int(manifest["relative_path"].str.startswith("Notes/").sum()),
        "total_size_mb": round(float(manifest["size_bytes"].sum()) / (1024 * 1024), 3),
    }
    return manifest, summary


def write_baseline_outputs(manifest: pd.DataFrame, summary: dict[str, Any]) -> None:
    manifest_path = CSV_DIR / f"term_{TERM}_baseline_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    stage1 = (NOTE_DIR / f"term_{TERM}_stage1_stage2_summary.txt").read_text(encoding="utf-8")
    stage2 = (NOTE_DIR / f"term_{TERM}_stage3_stage7_summary.txt").read_text(encoding="utf-8")
    directed = (NOTE_DIR / f"term_{TERM}_directed_party_summary.txt").read_text(encoding="utf-8")
    lines = [
        f"Term {TERM} baseline freeze summary",
        "",
        f"Frozen file count: {summary['file_count']}",
        f"CSV count: {summary['csv_count']}",
        f"Figure count: {summary['figure_count']}",
        f"Note count: {summary['note_count']}",
        f"Total frozen size (MB): {summary['total_size_mb']}",
        "",
        "This freeze captures the pre-validation 22nd-term outputs that act as the project baseline.",
        "The validation study does not overwrite these baseline hashes; any later rerun can be checked against the manifest.",
        "",
        "Stage 1-2 baseline note:",
        stage1.strip(),
        "",
        "Stage 3-7 baseline note:",
        stage2.strip(),
        "",
        "Directed-network baseline note:",
        directed.strip(),
    ]
    (NOTE_DIR / f"term_{TERM}_baseline_freeze_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def load_gold_set() -> pd.DataFrame:
    frame = pd.read_csv(GOLD_PATH)
    frame["manual_label"] = frame["manual_label"].astype(str).str.strip()
    frame["heuristic_label"] = frame["heuristic_label"].astype(str).str.strip()
    frame["heuristic_score"] = pd.to_numeric(frame.get("heuristic_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    frame["window_text"] = frame["window_text"].fillna("").astype(str)
    # BINARY: exclude mixed examples from evaluation
    frame = frame[frame["manual_label"].isin(CLASS_ORDER)].copy()
    # BEME binary label: score >= 0 → positive, score < 0 → negative (no mixed buffer)
    frame["heuristic_label_binary"] = frame["heuristic_score"].apply(
        lambda s: "positive" if s >= 0 else "negative"
    )
    frame["manual_label"] = pd.Categorical(frame["manual_label"], categories=CLASS_ORDER, ordered=True)
    frame["heuristic_label_binary"] = pd.Categorical(frame["heuristic_label_binary"], categories=CLASS_ORDER, ordered=True)
    return frame.reset_index(drop=True)


def write_annotation_protocol(gold: pd.DataFrame) -> None:
    lines = [
        f"Term {TERM} | BEME validation annotation protocol",
        "",
        "Goal:",
        "Build an expert-adjudicated benchmark for the signed window texts that feed the directed inter-party network.",
        "",
        "Sampling design:",
        "- Heuristically stratified by year (2002-2007) and by BEME label (positive / mixed / negative).",
        "- Draft sample target: 6 windows per year per heuristic label.",
        f"- Final annotated sample size: {len(gold)}",
        "",
        "Annotation scheme:",
        "- positive: supportive, approving, cooperative, or explicitly constructive language toward the target party.",
        "- negative: attacking, blaming, mocking, accusatory, or clearly critical language toward the target party.",
        "- mixed: descriptive mention, unclear stance, self-reference, or passages where target-specific sentiment is ambiguous.",
        "",
        "Important note:",
        "This benchmark is expert-coded by a single adjudicator inside the current project workflow. It is strong enough for model screening, but a future publication-grade extension should add a second annotator and inter-annotator agreement statistics.",
        "",
        "Observed manual class distribution:",
    ]
    for label, count in gold["manual_label"].value_counts().reindex(CLASS_ORDER, fill_value=0).items():
        lines.append(f"- {label}: {int(count)}")
    (NOTE_DIR / f"term_{TERM}_{VALIDATION_PREFIX}_annotation_protocol.txt").write_text("\n".join(lines), encoding="utf-8")


def build_feature_union() -> FeatureUnion:
    return FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    lowercase=False,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    lowercase=False,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def make_model_library() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    text_selector = FunctionTransformer(lambda frame: frame["window_text"], validate=False)
    logistic = Pipeline(
        steps=[
            ("select_text", text_selector),
            ("features", build_feature_union()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    svm = Pipeline(
        steps=[
            ("select_text", text_selector),
            ("features", build_feature_union()),
            ("clf", LinearSVC(random_state=RANDOM_SEED)),
        ]
    )
    return {
        "logistic_regression": (
            logistic,
            {
                "clf__C": [0.25, 0.5, 1.0, 2.0, 4.0],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "linear_svm": (
            svm,
            {
                "clf__C": [0.25, 0.5, 1.0, 2.0, 4.0],
                "clf__class_weight": [None, "balanced"],
            },
        ),
    }


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        zero_division=0,
    )
    report = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels=CLASS_ORDER),
    }
    for index, label in enumerate(CLASS_ORDER):
        report[f"{label}_precision"] = precision[index]
        report[f"{label}_recall"] = recall[index]
        report[f"{label}_f1"] = f1[index]
    return report


def evaluate_beme(gold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = gold["manual_label"].astype(str)
    y_pred = gold["heuristic_label_binary"].astype(str)   # BINARY: score-based label
    metric_row = {"model": "beme_heuristic_binary", **classification_metrics(y_true, y_pred)}
    prediction_df = gold[["candidate_id", "date", "year", "source_party", "target_party", "window_text", "manual_label", "heuristic_label_binary"]].copy()
    prediction_df = prediction_df.rename(columns={"heuristic_label_binary": "predicted_label"})
    prediction_df["model"] = "beme_heuristic_binary"
    return pd.DataFrame([metric_row]), prediction_df


def nested_cv_predictions(
    gold: pd.DataFrame,
    model_name: str,
    estimator: Pipeline,
    param_grid: dict[str, list[Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    X = gold[["window_text"]].copy()
    y = gold["manual_label"].astype(str)

    for fold_index, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx].copy()
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + fold_index)
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        predictions = search.best_estimator_.predict(X_test)
        for absolute_idx, predicted_label in zip(test_idx, predictions):
            rows.append(
                {
                    "model": model_name,
                    "candidate_id": gold.iloc[absolute_idx]["candidate_id"],
                    "manual_label": gold.iloc[absolute_idx]["manual_label"],
                    "predicted_label": predicted_label,
                    "fold": fold_index,
                }
            )
        metrics = classification_metrics(y_test, pd.Series(predictions, index=y_test.index))
        metrics["model"] = model_name
        metrics["fold"] = fold_index
        metrics["best_params"] = json.dumps(search.best_params_, ensure_ascii=False)
        fold_metrics.append(metrics)

    prediction_df = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    metric_row = {"model": model_name, **classification_metrics(prediction_df["manual_label"], prediction_df["predicted_label"])}
    return pd.DataFrame([metric_row]), pd.DataFrame(fold_metrics), prediction_df


def infer_berturk_label_map(model_outputs: list[dict[str, Any]]) -> dict[str, str]:
    calibration_texts = ["positive", "negative", "mixed"]
    observed_labels = {key: value["label"] for key, value in zip(calibration_texts, model_outputs)}
    if len(set(observed_labels.values())) != 3:
        return {}
    return {
        observed_labels["negative"]: "negative",
        observed_labels["mixed"]: "mixed",
        observed_labels["positive"]: "positive",
    }


def evaluate_berturk(gold: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - import guard
        return None, None, f"Transformers import failed: {type(exc).__name__}: {exc}"

    try:
        classifier = pipeline(
            "text-classification",
            model=BERT_MODEL_ID,
            tokenizer=BERT_MODEL_ID,
            device=-1,
        )
        calibration_outputs = classifier(
            [
                "bu karari destekliyoruz ve cok olumlu buluyoruz",
                "bu politika yanlis ve yetersizdir",
                "bu konuda goruslerimizi acikladik",
            ],
            truncation=True,
            batch_size=3,
        )
        label_map = infer_berturk_label_map(calibration_outputs)
        if not label_map:
            return None, None, "BERTurk labels could not be mapped to negative / mixed / positive."
        outputs = classifier(gold["window_text"].tolist(), truncation=True, batch_size=16)
        predictions = [label_map.get(item["label"], "mixed") for item in outputs]
        prediction_df = gold[["candidate_id", "manual_label"]].copy()
        prediction_df["predicted_label"] = predictions
        prediction_df["model"] = "berturk_off_the_shelf"
        metric_row = {"model": "berturk_off_the_shelf", **classification_metrics(prediction_df["manual_label"], prediction_df["predicted_label"])}
        return pd.DataFrame([metric_row]), prediction_df, None
    except Exception as exc:  # pragma: no cover - runtime fallback
        message = str(exc)
        if "Cannot send a request" in message or "nodename nor servname" in message:
            return None, None, "BERTurk evaluation could not run because the Hugging Face model download was blocked in the current local runtime."
        return None, None, f"BERTurk evaluation failed: {type(exc).__name__}: {exc}"


def bootstrap_differences(
    gold: pd.DataFrame,
    prediction_store: dict[str, pd.Series],
    iterations: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    y_true = gold["manual_label"].astype(str).reset_index(drop=True)
    baseline_pred = prediction_store["beme_heuristic"].reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    for model_name, predictions in prediction_store.items():
        if model_name == "beme_heuristic":
            continue
        preds = predictions.reset_index(drop=True)
        macro_diffs = []
        negative_diffs = []
        for _ in range(iterations):
            indices = rng.integers(0, len(y_true), len(y_true))
            sample_true = y_true.iloc[indices]
            sample_base = baseline_pred.iloc[indices]
            sample_model = preds.iloc[indices]
            macro_diffs.append(
                f1_score(sample_true, sample_model, labels=CLASS_ORDER, average="macro", zero_division=0)
                - f1_score(sample_true, sample_base, labels=CLASS_ORDER, average="macro", zero_division=0)
            )
            _, _, f1_base, _ = precision_recall_fscore_support(sample_true, sample_base, labels=CLASS_ORDER, zero_division=0)
            _, _, f1_model, _ = precision_recall_fscore_support(sample_true, sample_model, labels=CLASS_ORDER, zero_division=0)
            negative_diffs.append(f1_model[0] - f1_base[0])
        rows.append(
            {
                "comparison_model": model_name,
                "metric": "macro_f1_minus_beme",
                "mean_diff": float(np.mean(macro_diffs)),
                "ci_lower": float(np.quantile(macro_diffs, 0.025)),
                "ci_upper": float(np.quantile(macro_diffs, 0.975)),
            }
        )
        rows.append(
            {
                "comparison_model": model_name,
                "metric": "negative_f1_minus_beme",
                "mean_diff": float(np.mean(negative_diffs)),
                "ci_lower": float(np.quantile(negative_diffs, 0.025)),
                "ci_upper": float(np.quantile(negative_diffs, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def build_hybrid_predictions(gold: pd.DataFrame, prediction_store: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.Series] | None:
    # BINARY mode: no mixed gate needed. Hybrid removed.
    return None


def choose_strategy(summary: pd.DataFrame) -> dict[str, Any]:
    available = summary[summary["model"] != "beme_heuristic"].copy()
    if available.empty:
        return {
            "selected_primary_model": "beme_heuristic",
            "decision_rule": "no_competing_model_available",
            "recommended_strategy": "Keep BEME as the primary signed scoring method.",
        }

    best_row = available.sort_values(["macro_f1", "negative_f1", "weighted_f1"], ascending=False).iloc[0]
    beme_row = summary[summary["model"] == "beme_heuristic"].iloc[0]
    macro_gain = float(best_row["macro_f1"] - beme_row["macro_f1"])
    negative_gain = float(best_row["negative_f1"] - beme_row["negative_f1"])
    mixed_gap = float(best_row["mixed_f1"] - beme_row["mixed_f1"])

    if macro_gain >= 0.08 and negative_gain >= 0.10:
        rule = "replace_primary_with_best_model"
        recommendation = (
            f"Promote {best_row['model']} to the primary binary tone classifier for network construction; "
            "retain BEME (binary) as an interpretable audit layer."
        )
    elif macro_gain >= 0.03:
        rule = "keep_beme_plus_best_model_as_robustness"
        recommendation = (
            f"Keep BEME (binary) as the transparent lexical score; use {best_row['model']} as the "
            "robustness/validation classifier in the final signed network package."
        )
    else:
        rule = "keep_beme_binary_primary"
        recommendation = "BEME (binary: score≥0 → positive, score<0 → negative) remains competitive. No change needed."

    return {
        "selected_primary_model": best_row["model"],
        "decision_rule": rule,
        "recommended_strategy": recommendation,
        "macro_f1_gain_vs_beme": macro_gain,
        "negative_f1_gain_vs_beme": negative_gain,
    }


def plot_performance_summary(summary: pd.DataFrame) -> None:
    plot_df = summary.melt(
        id_vars="model",
        value_vars=["macro_f1", "negative_f1", "positive_f1", "cohen_kappa"],
        var_name="metric",
        value_name="score",
    )
    metric_order = ["macro_f1", "negative_f1", "positive_f1", "cohen_kappa"]
    label_map = {
        "macro_f1": "Macro-F1",
        "negative_f1": "Negative F1",
        "positive_f1": "Positive F1",
        "cohen_kappa": "Cohen's kappa",
    }
    plot_df["metric"] = pd.Categorical(plot_df["metric"], categories=metric_order, ordered=True)
    plot_df["metric_label"] = plot_df["metric"].map(label_map)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor("#f7f5ef")
    sns.barplot(data=plot_df, x="metric_label", y="score", hue="model", ax=ax)
    ax.set_title("Term 22 | BEME validation model comparison", fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.15)
    ax.legend(title="Model", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"term_{TERM}_{VALIDATION_PREFIX}_performance_summary.png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_confusion_matrices(predictions: dict[str, pd.Series], y_true: pd.Series) -> None:
    models = list(predictions.keys())
    cols = 2
    rows = math.ceil(len(models) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.5 * rows))
    fig.patch.set_facecolor("#f7f5ef")
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax, model_name in zip(axes.flatten(), models):
        matrix = confusion_matrix(y_true, predictions[model_name], labels=CLASS_ORDER, normalize="true")
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            xticklabels=CLASS_ORDER,
            yticklabels=CLASS_ORDER,
            ax=ax,
        )
        ax.set_title(model_name.replace("_", " ").title(), fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
    for ax in axes.flatten()[len(models):]:
        ax.axis("off")
    fig.suptitle("Term 22 | Normalized confusion matrices", fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(FIG_DIR / f"term_{TERM}_{VALIDATION_PREFIX}_confusion_matrices.png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_summary_note(
    gold: pd.DataFrame,
    baseline_summary: dict[str, Any],
    performance_summary: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    decision: dict[str, Any],
    bert_note: str | None,
) -> None:
    lines = [
        f"Term {TERM} | BEME validation study summary",
        "",
        "Stage 1 | Baseline freeze",
        f"- Frozen baseline file count: {baseline_summary['file_count']}",
        f"- Frozen baseline total size (MB): {baseline_summary['total_size_mb']}",
        f"- Manifest file: CSVs/term_{TERM}_baseline_manifest.csv",
        "",
        "Stage 2 | Gold set",
        f"- Final expert-coded sample size: {len(gold)}",
    ]
    for label, count in gold["manual_label"].value_counts().reindex(CLASS_ORDER, fill_value=0).items():
        lines.append(f"- {label}: {int(count)}")
    lines.extend(
        [
            "",
            "Stage 2 | Performance summary",
        ]
    )
    for row in performance_summary.sort_values("macro_f1", ascending=False).itertuples(index=False):
        lines.append(
            f"- {row.model}: macro_f1={row.macro_f1:.3f} | negative_f1={row.negative_f1:.3f} | "
            f"mixed_f1={row.mixed_f1:.3f} | positive_f1={row.positive_f1:.3f} | kappa={row.cohen_kappa:.3f}"
        )
    if bert_note:
        lines.extend(["", "BERTurk note:", f"- {bert_note}"])

    lines.extend(["", "Bootstrap differences versus BEME:"])
    for row in bootstrap_df.itertuples(index=False):
        lines.append(
            f"- {row.comparison_model} | {row.metric}: mean_diff={row.mean_diff:.3f} | "
            f"95% CI [{row.ci_lower:.3f}, {row.ci_upper:.3f}]"
        )

    lines.extend(
        [
            "",
            "Stage 3 | Model decision",
            f"- Selected primary learned model: {decision['selected_primary_model']}",
            f"- Decision rule: {decision['decision_rule']}",
            f"- Recommendation: {decision['recommended_strategy']}",
        ]
    )
    if "macro_f1_gain_vs_beme" in decision:
        lines.append(f"- Macro-F1 gain vs BEME: {decision['macro_f1_gain_vs_beme']:.3f}")
        lines.append(f"- Negative F1 gain vs BEME: {decision['negative_f1_gain_vs_beme']:.3f}")
        # mixed_f1 removed in binary mode

    (NOTE_DIR / f"term_{TERM}_{VALIDATION_PREFIX}_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    TERM_ROOT.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_DIR.mkdir(parents=True, exist_ok=True)

    manifest, baseline_summary = freeze_baseline(args.term)
    write_baseline_outputs(manifest, baseline_summary)

    gold = load_gold_set()
    write_annotation_protocol(gold)

    performance_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    prediction_store: dict[str, pd.Series] = {}
    oof_frames: list[pd.DataFrame] = []

    beme_summary, beme_predictions = evaluate_beme(gold)
    performance_frames.append(beme_summary)
    prediction_store["beme_heuristic"] = beme_predictions["predicted_label"].astype(str)
    oof_frames.append(beme_predictions)

    for model_name, (estimator, param_grid) in make_model_library().items():
        summary_df, fold_df, prediction_df = nested_cv_predictions(gold, model_name, estimator, param_grid)
        performance_frames.append(summary_df)
        fold_frames.append(fold_df)
        aligned_predictions = gold.merge(prediction_df[["candidate_id", "predicted_label", "fold"]], on="candidate_id", how="left")
        prediction_store[model_name] = aligned_predictions["predicted_label"].astype(str)
        oof_frames.append(aligned_predictions.assign(model=model_name))

    hybrid_result = build_hybrid_predictions(gold, prediction_store)
    if hybrid_result is not None:
        hybrid_summary, hybrid_predictions = hybrid_result
        performance_frames.append(hybrid_summary)
        prediction_store["hybrid_beme_mixed_gate_logistic"] = hybrid_predictions.astype(str)
        oof_frames.append(
            gold[["candidate_id", "manual_label"]]
            .assign(predicted_label=hybrid_predictions.astype(str), model="hybrid_beme_mixed_gate_logistic", fold=np.nan)
        )

    bert_note = None
    if args.try_bert:
        bert_summary, bert_predictions, bert_note = evaluate_berturk(gold)
        if bert_summary is not None and bert_predictions is not None:
            performance_frames.append(bert_summary)
            prediction_store["berturk_off_the_shelf"] = bert_predictions["predicted_label"].astype(str)
            oof_frames.append(gold.merge(bert_predictions[["candidate_id", "predicted_label"]], on="candidate_id", how="left").assign(model="berturk_off_the_shelf", fold=np.nan))
    else:
        bert_note = "BERTurk evaluation was not attempted in this run."

    performance_summary = pd.concat(performance_frames, ignore_index=True)
    fold_metrics = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    oof_predictions = pd.concat(oof_frames, ignore_index=True)
    bootstrap_df = bootstrap_differences(gold, prediction_store, args.bootstrap)
    decision = choose_strategy(performance_summary)

    performance_summary.to_csv(CSV_DIR / f"term_{TERM}_{VALIDATION_PREFIX}_performance_summary.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(CSV_DIR / f"term_{TERM}_{VALIDATION_PREFIX}_fold_metrics.csv", index=False, encoding="utf-8-sig")
    oof_predictions.to_csv(CSV_DIR / f"term_{TERM}_{VALIDATION_PREFIX}_oof_predictions.csv", index=False, encoding="utf-8-sig")
    bootstrap_df.to_csv(CSV_DIR / f"term_{TERM}_{VALIDATION_PREFIX}_bootstrap_differences.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([decision]).to_csv(CSV_DIR / f"term_{TERM}_{VALIDATION_PREFIX}_model_decision.csv", index=False, encoding="utf-8-sig")

    plot_performance_summary(performance_summary)
    plot_confusion_matrices(prediction_store, gold["manual_label"].astype(str))
    write_summary_note(gold, baseline_summary, performance_summary, bootstrap_df, decision, bert_note)


if __name__ == "__main__":
    main()
