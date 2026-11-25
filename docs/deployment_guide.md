# Deployment Guide

## Local / server
```bash
git clone https://github.com/bestbout/NLP-Portfolio_KhalidaBestani
cd NLP-Portfolio_KhalidaBestani/ITAI2373-NewsBot-Final
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pytest -q tests          # 14 tests
python -m src.newsbot    # terminal chat
```
The dataset is already in `data/raw/BBC News Train.csv`; to refresh it: `kaggle competitions download -c learn-ai-bbc && unzip learn-ai-bbc.zip -d data/raw/` (needs `~/.kaggle/kaggle.json`).

## Configuration
Edit `config/settings.py` (paths, split, TF-IDF params, C grid, topics, summary length, embeddings, spaCy model). No API keys are required; `config/api_keys_template.txt` shows where they would go — never commit real keys.

## Performance tips
- `NewsBot2(df).train(tune=False)` skips grid search for a faster start.
- Limit NER with `entity_table(df, limit=400)`.
- Save the trained classifier with `NewsClassifier.save("data/models/clf.joblib")` and reload with `load()`.

## Web deployment (bonus)
The team's Streamlit app is deployed on Streamlit Community Cloud from https://github.com/Mari-XR/NewsBot2-WebApp (`app.py`, branch `main`, Python 3.12): https://newsbot2-webapp.streamlit.app/. The public version uses 300 attributed Wikinews excerpts (CC BY 2.5) instead of the BBC CSV.

## Known operational limits
Translation depends on free external services (rate limits, internet). The chatbot is rule-based.
