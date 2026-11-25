"""Central configuration for NewsBot 2.0.

Values are the ones used in the project notebook
(notebooks/FP_NewsBot2_ITAI2373.ipynb) and listed in the
technical documentation, section 7 "Configuration Manual".
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA = DATA_DIR / "raw" / "BBC News Train.csv"
PROCESSED_DATA = DATA_DIR / "processed" / "newsbot_dataset.csv"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Classifier features (TF-IDF) and grid search
TFIDF_PARAMS = dict(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
LOGREG_C_GRID = [1, 5, 10, 20]

# Topic modeling
N_TOPICS = 5
TOPIC_METHOD = "nmf"  # "nmf" or "lda"

# Summarization
SUMMARY_SENTENCES = 3

# Semantic search
USE_EMBEDDINGS = False
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Multilingual
TRANSLATION_TARGET = "en"

# NER
SPACY_MODEL = "en_core_web_sm"
NER_ARTICLE_LIMIT = 400
