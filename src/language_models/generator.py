"""Module B: content enhancement and insight generation (template-based, no LLM needed).

Source: notebook section B3.
"""
from collections import Counter


def enhance_article(text, summary, category, confidence, sentiment, entities, related):
    """Build an 'enhanced' article card: summary + context (entities, related articles).

    Args:
        entities: list of (text, label) from ``ner_extractor.extract_entities``.
        related: list of (doc_index, score) from ``SemanticSearch.similar_to``.
    """
    people = [e for e, l in entities if l == "PERSON"]
    orgs = [e for e, l in entities if l == "ORG"]
    places = [e for e, l in entities if l == "GPE"]

    def top(xs):
        return ", ".join(w for w, _ in Counter(xs).most_common(3)) or "none found"

    return (f"Category: {category} ({confidence:.0%} confident) | Tone: {sentiment}\n"
            f"Summary: {summary}\n"
            f"Key people: {top(people)} | Organizations: {top(orgs)} | Places: {top(places)}\n"
            f"Related articles: {', '.join(str(i) for i, _ in related) or 'none'}")


def generate_insights(df, sent_df=None, topic_words=None):
    """Turn numbers into short plain-English findings."""
    out = []
    counts = df["category"].value_counts()
    out.append(f"The biggest category is {counts.index[0]} with {counts.iloc[0]} articles; "
               f"the smallest is {counts.index[-1]} with {counts.iloc[-1]}.")
    if sent_df is not None:
        m = sent_df.groupby("category")["compound"].mean().sort_values()
        out.append(f"The most negative category is {m.index[0]} (avg {m.iloc[0]:.2f}); "
                   f"the most positive is {m.index[-1]} (avg {m.iloc[-1]:.2f}).")
    if topic_words:
        for k, words in topic_words.items():
            out.append(f"{k} is about: {', '.join(words[:5])}.")
    return out
