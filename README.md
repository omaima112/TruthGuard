# TruthGuard

TruthGuard is a small Flask web application that demonstrates a simple AI-based "fake news" / claim detector. It uses a TF-IDF vectorizer + Logistic Regression model trained on a small curated dataset of example real and fake headlines.

## What I changed (competition sprint)
- Improved preprocessing: added NLTK lemmatization and ensured necessary corpora are downloaded.
- Added basic logging and better error handling in `app.py`.
- Added automatic training fallback: if the model files are missing the app will attempt to run `train_model.py` (useful during demos).
- Added `/model_info` endpoint so the front-end can show model metadata.
- Front-end: shows model info and a short "Why:" explanation returned by the model pipeline.
- Added a small pytest (`tests/test_api.py`) to sanity-check the `/analyze` endpoint.
- Added this README and pinned `pytest` in `requirements.txt`.

## Quick run (Windows PowerShell)
1. Create and activate a venv:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train the model (creates `models/truthguard_model.pkl` and `models/vectorizer.pkl`):

```powershell
python train_model.py
```

4. Run the app:

```powershell
python app.py
```

Open http://localhost:5000 in your browser.

## For the competition
- Inputs/outputs: POST JSON `{ "claim": "..." }` to `/analyze` → returns verdict, confidence, real/fake probabilities.
- Strengths: fast TF-IDF + Logistic Regression, easy to explain and reproduce, small footprint for demo.
- Limitations: small synthetic training data, possible false positives/negatives, no provenance checking or external fact-check lookup.
- Next steps: connect to an external fact-check API (Google Fact Check Tools API), expand training corpus, add more robust NLP (BERT embedding), and add logging/analytics for predictions.

-----
