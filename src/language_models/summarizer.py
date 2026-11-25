"""Module B: extractive summarization (TF-IDF scoring + TextRank) and Lead-3 baseline.

Source: notebook section B1.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_processing.text_preprocessor import split_sentences


def summarize(text, n_sentences=3, method="textrank"):
    """Pick the ``n_sentences`` most central sentences, kept in original order.

    method="textrank": PageRank (damping 0.85, 50 iterations) over the TF-IDF
    cosine-similarity graph of sentences. method="tfidf": average TF-IDF weight.
    """
    sents = split_sentences(text)
    if len(sents) <= n_sentences:
        return " ".join(sents)
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(sents)
    if method == "tfidf":
        scores = np.asarray(tfidf.sum(axis=1)).ravel() / (np.asarray((tfidf > 0).sum(axis=1)).ravel() + 1)
    else:
        sim = cosine_similarity(tfidf)
        np.fill_diagonal(sim, 0)
        row = sim.sum(axis=1, keepdims=True)
        row[row == 0] = 1
        m = sim / row
        scores = np.ones(len(sents)) / len(sents)
        for _ in range(50):
            scores = 0.15 / len(sents) + 0.85 * m.T.dot(scores)
    top = sorted(np.argsort(scores)[::-1][:n_sentences])
    return " ".join(sents[i] for i in top)


def lead_baseline(text, n_sentences=3):
    """Baseline summary: the first ``n_sentences`` sentences (Lead-3)."""
    return " ".join(split_sentences(text)[:n_sentences])
