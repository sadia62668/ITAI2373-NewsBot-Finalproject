# Technical Documentation – NewsBot Intelligence System 2.0

Summary of `reports/technical_report.pdf` (ITAI 2373 Final Project, team Purple Berries: Khalida Bestani, Marisabel Morales, Sadea Saeed). The PDF is the authoritative version.

## 1. Introduction
The midterm NewsBot 1.0 cleaned BBC articles, built TF-IDF features, analysed POS tags and syntax, scored sentiment with VADER, extracted entities with spaCy and classified 5 categories (best: Logistic Regression, 97.65%). NewsBot 2.0 turns that notebook into a small system with four modules (A–D) and a Python package, deliberately using methods taught in class (TF-IDF, LDA/NMF, TextRank, rules) so every member can explain each part.

## 2. Architecture
1. `data_processing` loads the Kaggle CSV, checks quality (missing, empty, duplicate articles) and cleans text.
2. Module A (`analysis`) trains the classifier and topic model and scores sentiment.
3. Module B (`language_models`) builds the search index and makes summaries.
4. Module C (`multilingual`) detects language, translates to English, sends text to Module A.
5. Module D (`conversation`) reads the question, finds the intent, calls A, B or C.
6. `src/newsbot.py` (`NewsBot2`) owns one copy of every component.

## 3. Data
- Kaggle `learn-ai-bbc`, `BBC News Train.csv`: 1,490 articles — sport 346, business 336, politics 274, entertainment 273, tech 261.
- `data_validator.validate()` found **50 exact duplicates** → 1,440 articles (removes train/test leakage).
- Preprocessing: lowercase; strip HTML/URLs/emails/numbers/symbols; NLTK stop words; WordNet lemmatization; drop words ≤ 2 letters; sentence splitter for summaries.

## 4. Modules and methods
### Module A – Advanced Content Analysis
| Feature | Method | File |
|---|---|---|
| Classification | Pipeline: TF-IDF (5,000 features, 1–2-grams, sublinear_tf) + Logistic Regression; GridSearchCV over C ∈ {1,5,10,20}, 3-fold | `analysis/classifier.py` |
| Confidence / multi-level | `predict_proba` → top-1 confidence, top-2 categories | `analysis/classifier.py` |
| Topic discovery | LDA on counts, NMF on TF-IDF; 5 topics, 3,000 words | `analysis/topic_modeler.py` |
| Topic evolution & clustering | dominant-topic share per period; KMeans on topic mix | `analysis/topic_modeler.py` |
| Sentiment + evolution | VADER compound (≥0.05 pos, ≤−0.05 neg); mean per period/category | `analysis/sentiment_analyzer.py` |
| Entity relationships | spaCy NER; co-occurring PERSON/ORG/GPE/NORP/EVENT pairs | `analysis/ner_extractor.py` |

> The BBC data has no publish dates; "evolution" uses 5 periods by `article_id` order — a demonstration, not a real trend.

### Module B – Language Understanding and Generation
- **Summarization**: TextRank (sentence TF-IDF → cosine graph → PageRank, damping 0.85, 50 iterations → top 3 in original order). Baseline Lead-3.
- **Semantic search**: TF-IDF (1–2-grams) + cosine; optional sentence-transformers `all-MiniLM-L6-v2`.
- **Query expansion**: pseudo-relevance feedback (top 5 results → 3 strongest centroid words).
- **Content enhancement**: `enhance_article()` article card. **Insights**: `generate_insights()`.

### Module C – Multilingual Intelligence
- `langdetect` (seeded) → code + probability.
- `deep-translator`: Google, then MyMemory backup; on failure returns original text and `ok=False`.
- `MultilingualAnalyzer`: detect → translate → English classifier + sentiment; `coverage_comparison()`.

### Module D – Conversational Interface
- `intent_classifier.py`: keyword + regex rules, 12 intents; synonyms (sports→sport, technology→tech).
- `split_multi_part()` handles "show tech articles and summarize article 3".
- `query_processor.py`: context dict (last results, offset, last category/article) so "more" works.
- `response_generator.py`: calls the module, formats the answer, catches errors.

## 5. Results
80/20 stratified split, `random_state=42`.

| Model | Midterm (1,490, with duplicates) | NewsBot 2.0 (1,440, no duplicates) |
|---|---|---|
| Naive Bayes | 96.98% | 94.79% |
| Logistic Regression | 97.65% | 95.83% |
| SVM (midterm RBF SVC / final Linear SVM) | 94.30% | 96.18% |
| Tuned Logistic Regression (C=10) | 97.65% (same pipeline on midterm data) | **96.53%**, macro F1 0.96 |

Per-class F1: business 0.96, entertainment 0.97, politics 0.95, sport 0.99, tech 0.95. The drop vs. the midterm reflects removed duplicates, not a worse model.

**Topics** — purity LDA 0.943, NMF 0.910. NMF topics: economy/market; film/award; labour/election; game/england; mobile/phone.

**Sentiment** (mean compound): entertainment 0.59, sport 0.55, tech 0.53, business 0.24, politics 0.05. Entity pairs (400 articles): blair–tony blair (16), michael howard–tory (15), britain–tony blair (13).

| Test | Result |
|---|---|
| Summary length | 23.4% of article (200 articles) |
| Summary keeps category (TextRank / Lead-3) | 99.0% / 98.5% |
| Search precision@5 (plain / expanded) | 0.96 / 0.94 (10 queries) |
| Language detection | 5/5 (en, es, fr, de, it) |
| Translation round-trip EN→ES→EN | 5/5 kept category |
| Intent accuracy | 20/20 (own test set – optimistic) |

No ROUGE: the dataset has no reference summaries.

## 6. Limitations and ethics
No dates (simulated evolution); old British English-only data (2004–05); VADER misreads finance news ("shares fell after the bank cut its profit forecast" → positive); translation needs internet; rule-based chatbot; extractive summaries can be choppy. Automatic labels/tone should not be used alone to judge journalists or sources.

## 7. Future work
Transformer models (DistilBERT, BART), dated multilingual data, improved Streamlit app.

## 8. References
Greene & Cunningham (2006) – BBC dataset; Blei, Ng & Jordan (2003) – LDA; Lee & Seung (1999) – NMF; Mihalcea & Tarau (2004) – TextRank; Hutto & Gilbert (2014) – VADER; scikit-learn, NLTK, spaCy, langdetect, deep-translator docs.
