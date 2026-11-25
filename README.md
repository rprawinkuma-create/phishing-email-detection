# Phishing Email Detection Model

A Scikit-learn machine learning project that classifies emails as **Phishing** or **Safe** using textual content (TF-IDF) and extracted email/URL features.

## Features
- **Dataset**: Balanced dataset of 400 phishing and legitimate emails (`data/emails.csv`).
- **Feature Extraction (`src/features.py`)**:
  - **Textual features**: TF-IDF unigrams and bigrams (`TfidfVectorizer`).
  - **URL & Metadata features**: URL count, insecure `http://` count, raw IP URLs, suspicious TLDs/shorteners (`bit.ly`, `.xyz`, `.top`, `.ru`, etc.), `@` in URLs, urgent/credential keywords, exclamation marks, uppercase ratio, digit count, and character length.
- **Model Training & Evaluation (`src/train.py`)**:
  - Compares `LogisticRegression`, `RandomForestClassifier`, and `LinearSVC` (`SVC(kernel="linear")`).
  - Displays test accuracy, classification report, and confusion matrix, and saves `outputs/confusion_matrix.png` and `models/phishing_detector.joblib`.
- **Prediction CLI (`src/predict.py`)**:
  - Classifies new emails as `"Phishing"` or `"Safe"` with confidence score and extracted URL/keyword breakdown.

## Setup & Usage

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 1. Generate dataset (optional - auto-generated during training if missing)
python -m src.dataset

# 2. Train & evaluate models (prints accuracy & confusion matrix, saves model & plot)
python -m src.train

# 3. Run demo predictions
python -m src.predict --demo

# 4. Classify a custom email
python -m src.predict --subject "URGENT: Verify Account" --body "Click http://192.168.1.5/login now!"

# 5. Run unit tests
pytest
```
