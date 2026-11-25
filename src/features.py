"""Email feature extraction and Scikit-learn pipeline builder."""

from __future__ import annotations

import re
from typing import Iterable, Any
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
IP_URL_REGEX = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}", re.IGNORECASE)
SUSPICIOUS_TLD_REGEX = re.compile(
    r"(bit\.ly|tinyurl\.com|t\.co|\.(?:xyz|top|ru|tk|info|club)(?:/|\b))",
    re.IGNORECASE,
)

PHISHING_KEYWORDS = [
    "urgent",
    "verify",
    "suspended",
    "locked",
    "password",
    "confirm",
    "immediately",
    "expire",
    "unauthorized",
    "click here",
    "prize",
    "winner",
    "claim",
    "bank",
    "security alert",
    "update your",
    "billing",
    "overdue",
    "ssn",
    "deactivation",
]

FEATURE_NAMES = [
    "url_count",
    "http_only_count",
    "ip_url_count",
    "suspicious_tld_count",
    "at_symbol_in_url",
    "urgent_keyword_count",
    "exclamation_count",
    "uppercase_ratio",
    "digit_count",
    "char_length",
]


def extract_single_email_features(text: str) -> dict[str, Any]:
    """Extract URL, keyword, and stylistic features from a single email text."""
    urls = URL_REGEX.findall(text)
    http_only_count = sum(1 for u in urls if u.lower().startswith("http://"))
    ip_url_count = sum(1 for u in urls if IP_URL_REGEX.search(u))
    suspicious_tld_count = sum(1 for u in urls if SUSPICIOUS_TLD_REGEX.search(u))
    at_symbol_in_url = sum(1 for u in urls if "@" in u)

    lowered = text.lower()
    matched_keywords = [kw for kw in PHISHING_KEYWORDS if kw in lowered]
    urgent_keyword_count = len(matched_keywords)
    exclamation_count = text.count("!")

    alpha_chars = [c for c in text if c.isalpha()]
    uppercase_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars
        else 0.0
    )
    digit_count = sum(1 for c in text if c.isdigit())
    char_length = len(text)

    return {
        "url_count": len(urls),
        "http_only_count": http_only_count,
        "ip_url_count": ip_url_count,
        "suspicious_tld_count": suspicious_tld_count,
        "at_symbol_in_url": at_symbol_in_url,
        "urgent_keyword_count": urgent_keyword_count,
        "exclamation_count": exclamation_count,
        "uppercase_ratio": round(uppercase_ratio, 4),
        "digit_count": digit_count,
        "char_length": char_length,
        "urls": urls,
        "matched_keywords": matched_keywords,
    }


class EmailFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer extracting numerical URL and keyword features."""

    def fit(self, X: Iterable[str], y: Any = None) -> "EmailFeatureExtractor":
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        rows = []
        for text in X:
            feats = extract_single_email_features(str(text))
            rows.append([feats[name] for name in FEATURE_NAMES])
        return np.asarray(rows, dtype=np.float64)


def build_pipeline(classifier: BaseEstimator | None = None) -> Pipeline:
    """Build a Scikit-learn Pipeline combining TF-IDF and handcrafted email features."""
    if classifier is None:
        classifier = LogisticRegression(max_iter=1000, random_state=42)

    features = FeatureUnion(
        transformer_list=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=3000,
                    stop_words="english",
                ),
            ),
            (
                "email_stats",
                Pipeline(
                    steps=[
                        ("extractor", EmailFeatureExtractor()),
                        ("scaler", StandardScaler()),
                    ]
                ),
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("features", features),
            ("classifier", classifier),
        ]
    )
