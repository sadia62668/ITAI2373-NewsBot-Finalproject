# User Guide – NewsBot 2.0

## Option 1: Google Colab (recommended)
1. Open `notebooks/FP_NewsBot2_ITAI2373.ipynb` in Colab (File → Open notebook → GitHub, or upload).
2. Run the first cell and upload your `kaggle.json` when asked (the notebook downloads `learn-ai-bbc`). Alternatively upload `data/raw/BBC News Train.csv` and skip the Kaggle cells.
3. Runtime → Run all. Training takes about 1–2 minutes on a Colab CPU.
4. At the end, uncomment the "Try it yourself!" cell to chat interactively.

## Option 2: Local terminal
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m src.newsbot        # from the ITAI2373-NewsBot-Final folder
```

## Things you can ask the chatbot
| Example | What it does |
|---|---|
| `show positive sports articles` | list articles of a category, optional tone filter |
| `more` / `next` | next 5 results of the last list |
| `summarize article 5` | TextRank summary |
| `what's the sentiment of business news` | average VADER score |
| `find articles about elections` | semantic search with query expansion |
| `who is tony blair` | entity lookup |
| `what are the main topics` | NMF topics |
| `translate article 3 to spanish` | summary translated (needs internet) |
| `classify: <text>` | category + confidence + sentiment |
| `export report` | writes `newsbot_report.md` |

Multi-part: `show tech articles and summarize article 3`.

## Web app (bonus, by Marisabel Morales)
Live: https://newsbot2-webapp.streamlit.app/ — source: https://github.com/Mari-XR/NewsBot2-WebApp. The public demo uses 300 Wikinews excerpts (CC BY 2.5), so its behaviour differs from the BBC notebook metrics.
