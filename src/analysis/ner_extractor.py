"""Module A: named entity recognition (spaCy, like midterm) + entity relationship mapping.

Source: notebook section A4.
"""
from collections import Counter
from itertools import combinations

import pandas as pd

_nlp = None
KEEP = {"PERSON", "ORG", "GPE", "NORP", "EVENT", "LOC", "PRODUCT"}


def get_nlp(model="en_core_web_sm"):
    """Load (and cache) the spaCy model, downloading it if missing."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load(model)
        except OSError:
            from spacy.cli import download
            download(model)
            _nlp = spacy.load(model)
    return _nlp


def extract_entities(text):
    """Return a list of (entity_text, label)."""
    doc = get_nlp()(str(text))
    return [(e.text.strip(), e.label_) for e in doc.ents]


def entity_table(df, text_col="content", limit=None):
    """One row per entity: doc index, category, text, label. ``limit`` keeps it fast on Colab."""
    rows = []
    sub = df if limit is None else df.head(limit)
    docs = get_nlp().pipe(sub[text_col].astype(str), batch_size=50)
    for (i, row), doc in zip(sub.iterrows(), docs):
        for e in doc.ents:
            rows.append({"doc": i, "category": row["category"], "text": e.text.strip(), "label": e.label_})
    return pd.DataFrame(rows)


def entity_relationships(ent_df, min_count=3, top_n=20):
    """Count how often two entities (PERSON/ORG/GPE...) appear in the same article."""
    pairs = Counter()
    for _, g in ent_df[ent_df["label"].isin(KEEP)].groupby("doc"):
        ents = sorted(set(g["text"].str.lower()))
        pairs.update(combinations(ents, 2))
    rel = pd.DataFrame([(a, b, c) for (a, b), c in pairs.items() if c >= min_count],
                       columns=["entity_1", "entity_2", "articles_together"])
    return rel.sort_values("articles_together", ascending=False).head(top_n).reset_index(drop=True)
