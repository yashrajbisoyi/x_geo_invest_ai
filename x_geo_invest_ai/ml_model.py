from pathlib import Path
import json
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH_CANDIDATES = [
    PROCESSED_DIR / "runtime_combined_news.csv",
    PROCESSED_DIR / "final_combined_news.csv",
]


def first_existing_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DATA_PATH = first_existing_path(DATA_PATH_CANDIDATES)


def build_features(df):
    model_df = df.copy()
    model_df["text"] = (
        model_df["title"].fillna("").astype(str) + " " + model_df["content"].fillna("").astype(str)
    )
    # Add missing columns with defaults if not present
    if "dataset_type" not in model_df.columns:
        model_df["dataset_type"] = "combined"
    if "is_geopolitical" not in model_df.columns:
        model_df["is_geopolitical"] = True
    if "is_financial" not in model_df.columns:
        model_df["is_financial"] = True
    return model_df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=2000, ngram_range=(1, 2)), "text"),
            (
                "structured",
                OneHotEncoder(handle_unknown="ignore"),
                [
                    "source",
                    "dataset_type",
                    "geo_risk_level",
                    "sentiment",
                    "is_geopolitical",
                    "is_financial",
                ],
            ),
        ]
    )


def candidate_models():
    """5 models for thorough comparison."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "complement_naive_bayes": ComplementNB(),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
        "decision_tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        # LinearSVC wrapped for predict_proba support
        "svm_linear": CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42)),
    }


def cross_validate_model(name, estimator, x_df, y, cv=5):
    """Run StratifiedKFold cross-validation and return mean ± std."""
    pipeline = Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("model", estimator)]
    )
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(pipeline, x_df, y, cv=skf, scoring="f1_weighted", n_jobs=1)
        return {
            "cv_f1_mean": round(float(scores.mean()), 4),
            "cv_f1_std": round(float(scores.std()), 4),
            "cv_folds": cv,
        }
    except Exception as exc:
        print(f"  Cross-validation failed for {name}: {exc}")
        return {"cv_f1_mean": None, "cv_f1_std": None, "cv_folds": cv}


def evaluate_model(name, estimator, x_train, x_test, y_train, y_test):
    pipeline = Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("model", estimator)]
    )

    train_start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_time = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    predictions = pipeline.predict(x_test)
    predict_time = time.perf_counter() - predict_start

    return {
        "name": name,
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, predictions),
        "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "training_time_seconds": train_time,
        "prediction_time_seconds": predict_time,
        "predictions": predictions,
        "report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }


def save_confusion_matrix(y_test, predictions, labels, model_name):
    """Save confusion matrix as PNG for display on the performance page."""
    cm = confusion_matrix(y_test, predictions, labels=labels)
    ui_bg = "#17132f"
    panel_bg = "#221b45"
    text_color = "#ffffff"
    grid_color = (1, 1, 1, 0.08)
    cmap = sns.blend_palette([panel_bg, "#6d5efc", "#cfc8ff"], as_cmap=True)

    # Shorten long labels for readability
    short_labels = []
    for label in labels:
        if len(label) > 22:
            parts = label.split("|")
            short_labels.append(parts[0].strip()[:22] + "…")
        else:
            short_labels.append(label)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=ui_bg)
    ax.set_facecolor(panel_bg)
    heatmap = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=short_labels,
        yticklabels=short_labels,
        ax=ax,
        linewidths=0.5,
        linecolor=grid_color,
        annot_kws={"color": text_color, "fontsize": 11},
        cbar_kws={"shrink": 0.9},
    )
    ax.set_title(f"Confusion Matrix — {model_name.replace('_', ' ').title()}", fontsize=13, pad=14)
    ax.title.set_color(text_color)
    ax.set_xlabel("Predicted label", fontsize=10, color=text_color)
    ax.set_ylabel("True label", fontsize=10, color=text_color)
    ax.tick_params(axis="x", colors=text_color, labelsize=8, rotation=30)
    ax.tick_params(axis="y", colors=text_color, labelsize=8, rotation=0)
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(colorbar.ax.get_yticklabels(), color=text_color)
    colorbar.outline.set_edgecolor(grid_color)
    plt.tight_layout()

    output_path = ARTIFACTS_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Confusion matrix saved -> {output_path}")
    return output_path


def save_model_comparison_chart(comparison):
    """Save a horizontal bar chart comparing all model accuracies."""
    names = [c["name"].replace("_", " ").title() for c in comparison]
    accuracies = [c["accuracy"] for c in comparison]
    f1s = [c["f1_weighted"] for c in comparison]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.barh(x + width / 2, accuracies, width, label="Accuracy", color="#6d5efc", alpha=0.85)
    bars2 = ax.barh(x - width / 2, f1s, width, label="F1 Weighted", color="#ff4f8b", alpha=0.85)

    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Score", fontsize=10)
    ax.set_title("Model Comparison — Accuracy vs F1 Weighted", fontsize=12, pad=12)
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9)
    ax.axvline(x=0.8, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    for bar in bars1:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", fontsize=7.5)
    for bar in bars2:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", fontsize=7.5)

    plt.tight_layout()
    output_path = ARTIFACTS_DIR / "model_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Model comparison chart saved -> {output_path}")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing training dataset: {DATA_PATH}")

    df = build_features(pd.read_csv(DATA_PATH))
    if df["investment_recommendation"].nunique() < 2:
        raise RuntimeError("Training requires at least two recommendation classes.")

    feature_cols = [
        "text", "source", "dataset_type", "geo_risk_level",
        "sentiment", "is_geopolitical", "is_financial",
    ]

    x_all = df[feature_cols]
    y_all = df["investment_recommendation"]

    stratify_target = y_all if y_all.value_counts().min() > 1 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all,
        test_size=0.25,
        random_state=42,
        stratify=stratify_target,
    )

    print(f"\nDataset: {len(df)} rows | Train: {len(x_train)} | Test: {len(x_test)}")
    print(f"Classes: {y_all.nunique()} unique recommendations\n")

    models = candidate_models()
    results = []

    for name, estimator in models.items():
        print(f"Training: {name}...")

        # Cross-validation first (on full dataset for reliable estimate)
        print(f"  Running 5-fold cross-validation...")
        cv_result = cross_validate_model(name, estimator, x_all, y_all, cv=5)
        if cv_result["cv_f1_mean"]:
            print(f"  CV F1: {cv_result['cv_f1_mean']:.4f} ± {cv_result['cv_f1_std']:.4f}")

        # Train/test evaluation
        try:
            result = evaluate_model(name, estimator, x_train, x_test, y_train, y_test)
            result.update(cv_result)
            results.append(result)
            print(f"  Accuracy: {result['accuracy']:.4f} | F1: {result['f1_weighted']:.4f} | "
                  f"Train: {result['training_time_seconds']:.3f}s\n")
        except Exception as exc:
            print(f"  Model failed during train/test evaluation: {exc}\n")

    if not results:
        raise RuntimeError("No model completed training successfully.")

    # Pick best model by CV F1 (more reliable), fall back to test F1
    best = max(results, key=lambda r: (
        r["cv_f1_mean"] if r["cv_f1_mean"] else 0,
        r["f1_weighted"],
        r["accuracy"]
    ))

    print(f"Best model: {best['name']}")

    # Save best model
    with open(ARTIFACTS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best["pipeline"], f)

    # Save confusion matrix for best model
    all_labels = sorted(y_all.unique().tolist())
    save_confusion_matrix(y_test, best["predictions"], all_labels, best["name"])

    # Build comparison table
    comparison = [
        {
            "name": r["name"],
            "accuracy": round(r["accuracy"], 4),
            "f1_weighted": round(r["f1_weighted"], 4),
            "cv_f1_mean": r.get("cv_f1_mean"),
            "cv_f1_std": r.get("cv_f1_std"),
            "training_time_seconds": round(r["training_time_seconds"], 4),
            "prediction_time_seconds": round(r["prediction_time_seconds"], 6),
        }
        for r in results
    ]

    # Save comparison chart
    save_model_comparison_chart(comparison)

    evaluation_payload = {
        "dataset_path": str(DATA_PATH),
        "dataset_rows": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "best_model": best["name"],
        "accuracy": round(best["accuracy"], 4),
        "f1_weighted": round(best["f1_weighted"], 4),
        "cv_f1_mean": best.get("cv_f1_mean"),
        "cv_f1_std": best.get("cv_f1_std"),
        "training_time_seconds": round(best["training_time_seconds"], 4),
        "prediction_time_seconds": round(best["prediction_time_seconds"], 6),
        "model_comparison": comparison,
        "classification_report": best["report"],
        "improvement_notes": [
            "Increase verified-source coverage and historical depth beyond 90 days.",
            "Replace rule-generated labels with expert-annotated market truth targets.",
            "Evaluate transformer-based encoders (FinBERT) for richer geopolitical context.",
            "Add deep learning models (LSTM, BiLSTM) for sequential news pattern detection.",
            "Incorporate real-time macro indicators (CPI, interest rates) as structured features.",
        ],
    }

    with open(ARTIFACTS_DIR / "model_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_payload, f, indent=2)

    prediction_samples = x_test.copy()
    prediction_samples["actual"] = y_test.values
    prediction_samples["predicted"] = best["predictions"]
    prediction_samples.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)

    print("\nModel training and evaluation completed.")
    print(f"Best model: {best['name']}")
    print(f"Accuracy: {best['accuracy']:.4f} | F1: {best['f1_weighted']:.4f}")
    if best.get("cv_f1_mean"):
        print(f"5-Fold CV F1: {best['cv_f1_mean']:.4f} ± {best['cv_f1_std']:.4f}")


if __name__ == "__main__":
    main()
