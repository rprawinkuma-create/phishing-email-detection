"""Train and evaluate Scikit-learn models for Phishing Email Detection."""

from __future__ import annotations

from pathlib import Path
import joblib
import matplotlib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.dataset import DEFAULT_DATA_PATH, load_dataset
from src.features import FEATURE_NAMES, EmailFeatureExtractor, build_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "phishing_detector.joblib"
DEFAULT_PLOT_PATH = PROJECT_ROOT / "outputs" / "confusion_matrix.png"
LABELS = ["Safe", "Phishing"]


def train_and_evaluate(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    plot_path: Path = DEFAULT_PLOT_PATH,
) -> dict[str, object]:
    """Train multiple Scikit-learn classifiers, evaluate them, and save the best."""
    df = load_dataset(data_path)
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "LinearSVM": CalibratedClassifierCV(
            SVC(kernel="linear", random_state=42), ensemble=False
        ),
    }

    best_name = ""
    best_accuracy = -1.0
    best_pipeline = None
    best_predictions = None
    results: dict[str, float] = {}

    print("=== Feature Extraction Summary (Train Set Mean) ===")
    extractor = EmailFeatureExtractor()
    train_stats = pd.DataFrame(extractor.transform(X_train), columns=FEATURE_NAMES)
    train_stats["label"] = y_train.values
    print(train_stats.groupby("label").mean().round(3).to_string())
    print()

    print("=== Model Comparison ===")
    for name, clf in candidates.items():
        pipeline = build_pipeline(clf)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"{name:20s} | Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        if acc > best_accuracy:
            best_accuracy = acc
            best_name = name
            best_pipeline = pipeline
            best_predictions = preds

    assert best_pipeline is not None and best_predictions is not None

    cm = confusion_matrix(y_test, best_predictions, labels=LABELS)
    print(f"\n=== Best Model: {best_name} ===")
    print(f"Accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)\n")
    print("Confusion Matrix (rows=Actual, cols=Predicted):")
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in LABELS], columns=[f"Pred {l}" for l in LABELS])
    print(cm_df.to_string())
    print("\nClassification Report:")
    print(classification_report(y_test, best_predictions, target_names=LABELS, digits=4))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)
    print(f"Saved trained model to: {model_path}")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {best_name} (Acc: {best_accuracy * 100:.2f}%)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix plot to: {plot_path}")

    return {
        "best_model_name": best_name,
        "best_accuracy": best_accuracy,
        "confusion_matrix": cm,
        "results": results,
        "model_path": model_path,
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    train_and_evaluate()
