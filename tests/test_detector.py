"""Unit tests for the Phishing Email Detection Model."""

from __future__ import annotations

from pathlib import Path
from src.dataset import generate_dataset, load_dataset
from src.features import FEATURE_NAMES, EmailFeatureExtractor, extract_single_email_features
from src.predict import predict_email
from src.train import train_and_evaluate


def test_feature_extraction_detects_urls_and_keywords() -> None:
    text = (
        "URGENT: Verify your password immediately! "
        "Visit http://192.168.1.10/login or http://bit.ly/verify"
    )
    feats = extract_single_email_features(text)
    assert feats["url_count"] == 2
    assert feats["http_only_count"] == 2
    assert feats["ip_url_count"] == 1
    assert feats["suspicious_tld_count"] == 1
    assert feats["urgent_keyword_count"] >= 3
    assert feats["exclamation_count"] == 1

    transformer = EmailFeatureExtractor()
    matrix = transformer.transform([text])
    assert matrix.shape == (1, len(FEATURE_NAMES))


def test_dataset_generation_and_loading(tmp_path: Path) -> None:
    csv_path = tmp_path / "emails.csv"
    df = generate_dataset(output_path=csv_path, num_samples=40, seed=7)
    assert len(df) == 40
    assert set(df["label"].unique()) == {"Phishing", "Safe"}

    loaded = load_dataset(csv_path)
    assert "text" in loaded.columns
    assert len(loaded) == 40


def test_training_and_prediction_end_to_end(tmp_path: Path) -> None:
    data_path = tmp_path / "emails.csv"
    model_path = tmp_path / "model.joblib"
    plot_path = tmp_path / "cm.png"

    generate_dataset(output_path=data_path, num_samples=120, seed=42)
    metrics = train_and_evaluate(
        data_path=data_path,
        model_path=model_path,
        plot_path=plot_path,
    )

    assert metrics["best_accuracy"] >= 0.90
    assert model_path.exists()
    assert plot_path.exists()

    phish_res = predict_email(
        subject="URGENT: Account Suspended",
        body="Click here http://192.168.0.5/login to verify your password immediately!",
        model_path=model_path,
    )
    assert phish_res["prediction"] == "Phishing"

    safe_res = predict_email(
        subject="Project Standup Notes",
        body="Hi team, here are the notes from today's meeting at https://docs.google.com/document/d/notes.",
        model_path=model_path,
    )
    assert safe_res["prediction"] == "Safe"
