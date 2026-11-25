# 🤖 NewsBot Intelligence System 2.0

**ITAI 2373 – Natural Language Processing · Final Project**
**Team Purple Berries:** Khalida Bestani · Marisabel Morales · Sadea Saeed
**Portfolio owner:** Khalida Bestani – [NLP-Portfolio_KhalidaBestani](https://github.com/bestbout/NLP-Portfolio_KhalidaBestani)

NewsBot 2.0 is an NLP system that analyses BBC news articles: it classifies them into five categories with confidence scores, discovers topics, measures sentiment, maps entity relationships, writes extractive summaries, searches articles semantically, works across languages, and answers questions through a conversational interface. It extends our [midterm NewsBot](../ITAI2373-NewsBot-Midterm/).

## Project Video Link
[Watch the NewsBot Midterm Video](https://youtu.be/watch?v=ssH_tT2zNqY&t=4s) 

🌐 **Live web app (bonus):** https://newsbot2-webapp.streamlit.app/ (source: [Mari-XR/NewsBot2-WebApp](https://github.com/Mari-XR/NewsBot2-WebApp))

---

## ✨ Features
| Module | What it does | Code |
|---|---|---|
| **A. Advanced Content Analysis** | Tuned TF-IDF + Logistic Regression pipeline with confidence and top-2 categories; LDA & NMF topic modeling; topic "evolution" and KMeans clustering; VADER sentiment + evolution; entity relationship mapping (spaCy) | `src/analysis/` |
| **B. Language Understanding & Generation** | TextRank extractive summaries (vs. Lead-3), semantic search with query expansion, enhanced article cards, automatic insights | `src/language_models/` |
| **C. Multilingual Intelligence** | Language detection (langdetect), translation (deep-translator, Google → MyMemory fallback), cross-language coverage comparison | `src/multilingual/` |
| **D. Conversational Interface** | Rule-based intent classifier (12 intents), multi-part questions, conversation memory ("more"), response generator | `src/conversation/` |
| **Integration** | `NewsBot2` class wiring all modules; Markdown report export | `src/newsbot.py`, `src/utils/` |

## 📊 Results (from the notebook / technical report)
Dataset: Kaggle *learn-ai-bbc* `BBC News Train.csv`, 1,490 articles → **1,440 after removing 50 exact duplicates**. 80/20 stratified split, `random_state=42`.

| Model | Midterm (1,490, with duplicates) | NewsBot 2.0 (1,440, no duplicates) |
|---|---|---|
| Naive Bayes | 96.98% | 94.79% |
| Logistic Regression | 97.65% | 95.83% |
| SVM (midterm RBF / final Linear) | 94.30% | 96.18% |
| **Tuned Logistic Regression (C=10)** | 97.65% | **96.53%** (macro F1 0.96) |

| Other result | Value |
|---|---|
| Topic purity | LDA 0.943 · NMF 0.910 |
| Mean sentiment (VADER compound) | entertainment 0.59, sport 0.55, tech 0.53, business 0.24, politics 0.05 |
| Summary length / category kept (TextRank vs Lead-3) | 23.4% of article · 99.0% vs 98.5% |
| Search precision@5 (plain / expanded) | 0.96 / 0.94 |
| Language detection | 5/5 (en, es, fr, de, it) · round-trip EN→ES→EN 5/5 category kept |
| Intent accuracy | 20/20 on our own test questions |

The lower accuracy than the midterm is due to removing duplicate articles that leaked between train and test — 96.5% is the more honest number. Full details: [docs/technical_documentation.md](docs/technical_documentation.md) and [reports/technical_report.pdf](reports/technical_report.pdf).

**Limitations:** the BBC data has no dates (topic/sentiment "evolution" is simulated by article_id order); old British English-only data (2004–05); VADER misreads some finance news; translation needs internet; the chatbot is rule-based.

## 📁 Repository structure
```
ITAI2373-NewsBot-Final/
├── README.md
├── requirements.txt
├── config/            settings.py, api_keys_template.txt (no real keys)
├── src/
│   ├── data_processing/   text_preprocessor.py, feature_extractor.py, data_validator.py
│   ├── analysis/          classifier.py, sentiment_analyzer.py, ner_extractor.py, topic_modeler.py
│   ├── language_models/   summarizer.py, generator.py, embeddings.py
│   ├── multilingual/      translator.py, language_detector.py, cross_lingual_analyzer.py
│   ├── conversation/      query_processor.py, intent_classifier.py, response_generator.py
│   ├── utils/             visualization.py, evaluation.py, export.py
│   └── newsbot.py         NewsBot2 – connects everything
├── notebooks/         01_Data_Exploration … 07_System_Integration + full FP_NewsBot2_ksm_GroupName_ITAI2373.ipynb
├── tests/             test_preprocessing, test_classification, test_topic_modeling, test_integration
├── data/              raw/ (BBC News Train.csv), processed/, models/, results/
├── docs/              technical_documentation, user_guide, api_reference, deployment_guide,
│                      deployment_instructions, individual_contributions
└── reports/           executive_summary.pdf, technical_report.pdf, presentation_slides.pdf,
                       FP_* originals, FP_ReflectiveJournal_PurpleBerries_ITAI2373.pdf
```
The numbered notebooks are sections split out of the full notebook (outputs preserved); the full notebook `FP_NewsBot2_ksm_GroupName_ITAI2373.ipynb` is the one that runs end to end.

## ⚙️ Setup
```bash
git clone https://github.com/bestbout/NLP-Portfolio_KhalidaBestani
cd NLP-Portfolio_KhalidaBestani/ITAI2373-NewsBot-Final
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pytest -q tests          # 14 tests
python -m src.newsbot    # chat with NewsBot in the terminal
```
Python usage:
```python
from config import settings
from src.data_processing.data_validator import load_bbc
from src.newsbot import NewsBot2

bot = NewsBot2(load_bbc(settings.RAW_DATA)).train(tune=True)   # ~96.5% test accuracy
print(bot.ask("show positive tech articles and summarize article 3"))
```

## ☁️ Running in Google Colab
1. Open `notebooks/FP_NewsBot2_ksm_GroupName_ITAI2373.ipynb` in Colab (File → Open notebook → GitHub → paste this repo URL).
2. Run the first cell and upload `kaggle.json` when prompted (the notebook downloads the `learn-ai-bbc` dataset). Without Kaggle: upload `data/raw/BBC News Train.csv` to the Colab file panel and skip the Kaggle cells.
3. **Runtime → Run all.** Everything trains in about 1–2 minutes on CPU. Translation cells need internet.
4. The interactive chat loop at the end is commented out so "Run all" does not block — uncomment to try it.

## 👩‍💻 Individual contribution summary – Khalida Bestani
Per the team's technical documentation, Khalida led **Module A – Advanced Content Analysis** (classification with tuning and confidence scores, LDA/NMF topic modeling, sentiment, entity relationships) and shared **Module D** (intent processing, query handling, system integration). Presenter for slides 3–8.

- **My key work:** TODO (Khalida: list the specific parts you implemented)
- **Challenges I solved:** TODO
- **What I learned:** TODO

More detail: [docs/individual_contributions.md](docs/individual_contributions.md). Team-level reflection: [reports/FP_ReflectiveJournal_PurpleBerries_ITAI2373.pdf](reports/FP_ReflectiveJournal_PurpleBerries_ITAI2373.pdf).

## 📄 Reports
- [Executive summary](reports/executive_summary.pdf)
- [Technical report](reports/technical_report.pdf)
- [Presentation slides (PDF)](reports/presentation_slides.pdf) · [PPTX](reports/FP_Presentation_PurpleBerries_ITAI2373.pptx)
- [Reflective journal](reports/FP_ReflectiveJournal_PurpleBerries_ITAI2373.pdf)

## 🤝 AI assistance attribution
AI tools were used to help organise this repository: refactoring the notebook's code into the `src/` module files, writing docstrings, tests and documentation (summarised from our technical report), and drafting this README. All NLP methods, experiments and results come from our team's notebook; the code was reviewed and the tests were run before committing. TODO (Khalida): add any other AI tools the team used during the project, as required by the course policy.

## 📚 Data & references
BBC dataset: Greene & Cunningham (2006), via Kaggle *learn-ai-bbc*. Methods: LDA (Blei et al., 2003), NMF (Lee & Seung, 1999), TextRank (Mihalcea & Tarau, 2004), VADER (Hutto & Gilbert, 2014); libraries scikit-learn, NLTK, spaCy, langdetect, deep-translator.
