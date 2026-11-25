"""Feature extraction: TF-IDF, per-category top terms, POS and syntactic features,
plus the combined midterm feature matrix.

Source: midterm part of the notebook (Modules 3-5 and 7 cells). Sentence
embeddings live in ``src/language_models/embeddings.py``.
"""
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MIDTERM_TFIDF_PARAMS = dict(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8)


def build_tfidf(processed_texts, **params):
    """Fit a TF-IDF vectorizer (midterm settings by default).

    Returns:
        (vectorizer, sparse matrix, feature_names)
    """
    vec = TfidfVectorizer(**{**MIDTERM_TFIDF_PARAMS, **params})
    matrix = vec.fit_transform(processed_texts)
    return vec, matrix, vec.get_feature_names_out()


def tfidf_dataframe(matrix, feature_names, categories):
    """Dense TF-IDF DataFrame with an extra ``category`` column."""
    tfidf_df = pd.DataFrame(matrix.toarray(), columns=feature_names)
    tfidf_df["category"] = list(categories)
    return tfidf_df


def get_top_tfidf_terms(tfidf_df, category, n_terms=10):
    """Top-N terms by mean TF-IDF score for one category."""
    category_data = tfidf_df[tfidf_df["category"] == category]
    mean_scores = category_data.drop("category", axis=1).mean().sort_values(ascending=False)
    return mean_scores.head(n_terms)


def analyze_pos_patterns(text):
    """Proportion of each Penn Treebank POS tag in ``text`` (NLTK tagger)."""
    from nltk import pos_tag, word_tokenize
    if not text or pd.isna(text):
        return {}
    pos_tags = pos_tag(word_tokenize(str(text)))
    total = len(pos_tags)
    if total == 0:
        return {}
    counts = Counter(tag for _, tag in pos_tags)
    return {pos: c / total for pos, c in counts.items()}


def extract_syntactic_features(text, nlp):
    """spaCy dependency features: sentence/token counts, noun phrases, subjects, objects.

    Args:
        text: article text.
        nlp: a loaded spaCy pipeline (e.g. ``en_core_web_sm``).
    """
    if not text or pd.isna(text):
        return {}
    doc = nlp(str(text))
    features = {"num_sentences": len(list(doc.sents)), "num_tokens": len(doc),
                "dependency_relations": [], "noun_phrases": [], "subjects": [], "objects": []}
    for token in doc:
        if not token.is_space and not token.is_punct:
            features["dependency_relations"].append(token.dep_)
    for chunk in doc.noun_chunks:
        features["noun_phrases"].append(chunk.text.lower())
    for token in doc:
        if token.dep_ in ["nsubj", "nsubjpass"]:
            features["subjects"].append(token.text.lower())
        elif token.dep_ in ["dobj", "iobj", "pobj"]:
            features["objects"].append(token.text.lower())
    features["dependency_counts"] = dict(Counter(features["dependency_relations"]))
    return features


def combined_features(tfidf_matrix, sentiment_df, texts):
    """Midterm feature matrix: TF-IDF + 4 VADER scores + character length + word count.

    ``sentiment_df`` needs columns content_sentiment, pos_score, neu_score, neg_score.
    """
    texts = pd.Series(list(texts))
    sentiment = sentiment_df[["content_sentiment", "pos_score", "neu_score", "neg_score"]].values
    lengths = np.array([texts.str.len(), texts.str.split().str.len()]).T
    return np.hstack([tfidf_matrix.toarray(), sentiment, lengths])
