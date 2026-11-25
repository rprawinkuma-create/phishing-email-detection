"""Classify emails as Phishing or Safe using the trained Scikit-learn model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import joblib

from src.features import extract_single_email_features
from src.train import DEFAULT_MODEL_PATH, train_and_evaluate


def predict_email(
    subject: str,
    body: str,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Predict whether an email is 'Phishing' or 'Safe' with feature breakdown."""
    if not model_path.exists():
        train_and_evaluate(model_path=model_path)

    pipeline = joblib.load(model_path)
    text = f"{subject}\n{body}"
    prediction = str(pipeline.predict([text])[0])

    confidence = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([text])[0]
        classes = list(pipeline.classes_)
        confidence = float(probs[classes.index(prediction)])

    features = extract_single_email_features(text)
    return {
        "prediction": prediction,
        "confidence": confidence,
        "features": features,
    }


def _print_result(title: str, subject: str, result: dict[str, Any]) -> None:
    feats = result["features"]
    conf_str = (
        f" ({result['confidence'] * 100:.2f}% confidence)"
        if result["confidence"] is not None
        else ""
    )
    print(f"\n--- {title} ---")
    print(f"Subject          : {subject}")
    print(f"Classification   : {result['prediction']}{conf_str}")
    print(f"URLs detected    : {feats['urls'] or 'None'}")
    print(f"Matched keywords : {feats['matched_keywords'] or 'None'}")
    print(
        f"Feature summary  : url_count={feats['url_count']}, "
        f"http_only={feats['http_only_count']}, ip_url={feats['ip_url_count']}, "
        f"suspicious_tld={feats['suspicious_tld_count']}, "
        f"urgent_keywords={feats['urgent_keyword_count']}, "
        f"exclamations={feats['exclamation_count']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phishing Email Detection CLI")
    parser.add_argument("--subject", type=str, help="Email subject line")
    parser.add_argument("--body", type=str, help="Email body content")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run predictions on built-in sample Phishing and Safe emails",
    )
    args = parser.parse_args()

    if args.demo or (args.subject is None and args.body is None):
        samples = [
            (
                "Sample Phishing Email",
                "URGENT: Your PayPal account has been suspended!",
                "Dear customer, unauthorized login detected. Verify your password immediately at http://192.168.1.50/verify or click here http://bit.ly/3xVerifyNow!",
            ),
            (
                "Sample Safe Email",
                "Weekly Team Sync - Agenda for Thursday",
                "Hi team, Please find the meeting agenda and sprint notes at https://docs.google.com/document/d/notes. See you at 10 AM.",
            ),
        ]
        for title, subj, body in samples:
            res = predict_email(subj, body)
            _print_result(title, subj, res)
    else:
        subj = args.subject or ""
        body = args.body or ""
        res = predict_email(subj, body)
        _print_result("Email Classification Result", subj, res)


if __name__ == "__main__":
    main()
