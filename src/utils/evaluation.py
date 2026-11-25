"""Evaluation utilities refactored from the notebook's evaluation cells
(search precision@5, summary label preservation, intent accuracy).
Topic purity lives in ``src/analysis/topic_modeler.py``.
"""
import numpy as np
import pandas as pd

from src.conversation.intent_classifier import detect_intent
from src.data_processing.text_preprocessor import preprocess_text
from src.language_models.summarizer import lead_baseline, summarize


def search_precision_at_k(searcher, df, queries, k=5, expand=False):
    """Mean share of top-k results whose category matches the expected one.

    ``queries`` = {query: expected_category}.
    """
    scores = []
    for q, cat in queries.items():
        qq = searcher.expand_query(q) if expand else q
        scores.append(np.mean([df.category[i] == cat for i, _ in searcher.search(qq, k)]))
    return float(np.mean(scores))


def summary_label_preservation(news_clf, df, n=200, random_state=1):
    """Classify TextRank and Lead-3 summaries; report compression and label kept rates."""
    rows = []
    for _, r in df.sample(n, random_state=random_state).iterrows():
        s_tr, s_lead = summarize(r["content"], 3), lead_baseline(r["content"], 3)
        rows.append({
            "compression": len(s_tr.split()) / len(r["content"].split()),
            "textrank_keeps_label": news_clf.predict([preprocess_text(s_tr)])[0] == r["category"],
            "lead3_keeps_label": news_clf.predict([preprocess_text(s_lead)])[0] == r["category"],
        })
    return pd.DataFrame(rows).mean().to_dict()


def intent_accuracy(tests):
    """Accuracy of ``detect_intent`` on [(query, expected_intent)]; returns (acc, wrong list)."""
    pred = [detect_intent(q)[0] for q, _ in tests]
    wrong = [(q, p, t) for (q, t), p in zip(tests, pred) if p != t]
    return float(np.mean([p == t for p, (_, t) in zip(pred, tests)])), wrong
