"""Module D: rule-based intent classification (keywords + regex).

Source: notebook section D. 12 intents: greeting, help, show_category, follow_up,
summarize, sentiment, search, entity, topics, translate, classify, export.
"""
import re

CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]
SYNONYMS = {"sports": "sport", "technology": "tech", "political": "politics",
            "finance": "business", "economy": "business", "movies": "entertainment",
            "music": "entertainment", "film": "entertainment"}


def find_category(text):
    """Return the first category (or synonym) mentioned in ``text``, else None."""
    words = re.findall(r"[a-z]+", text.lower())
    for w in words:
        if w in CATEGORIES:
            return w
        if w in SYNONYMS:
            return SYNONYMS[w]
    return None


def find_sentiment_filter(text):
    """Return "positive"/"negative" if the query asks for a tone filter, else None."""
    q = text.lower()
    if re.search(r"\b(positive|good|happy)\b", q):
        return "positive"
    if re.search(r"\b(negative|bad|sad)\b", q):
        return "negative"
    return None


def detect_intent(query):
    """Return (intent, slots) for one query using ordered keyword/regex rules."""
    q = query.lower().strip()
    num = re.search(r"\b(\d+)\b", q)
    if re.search(r"\b(hi|hello|hey)\b", q):
        return "greeting", {}
    if re.search(r"\b(help|what can you do)\b", q):
        return "help", {}
    if "summar" in q and num:
        return "summarize", {"index": int(num.group(1))}
    if "sentiment" in q or "mood" in q or "feel" in q:
        return "sentiment", {"category": find_category(q), "index": int(num.group(1)) if num else None}
    if "topic" in q:
        return "topics", {}
    if re.search(r"\b(translate)\b", q):
        m = re.search(r"translate\s+(?:article\s+)?(\d+)\s+(?:to|into)\s+(\w+)", q)
        if m:
            return "translate", {"index": int(m.group(1)), "lang": m.group(2)}
    if re.search(r"\b(classify|categor)", q):
        text = re.sub(r"^.*?(classify|categorize)\s*:?\s*", "", query, flags=re.I)
        return "classify", {"text": text}
    m = re.search(r"\b(?:who is|tell me about|news about|mentions of)\s+(.+)$", q)
    if m and not re.search(r"\barticles?\b", q):
        return "entity", {"name": m.group(1).strip(" ?.")}
    if re.search(r"\b(more|another|next|show more)\b", q) and len(q.split()) <= 4:
        return "follow_up", {}
    if re.search(r"\b(report|export)\b", q):
        return "export", {}
    if re.search(r"\b(find|search|about|related)\b", q):
        m = re.search(r"(?:about|for|on|related to)\s+(.+)$", q)
        return "search", {"query": m.group(1) if m else q}
    if re.search(r"\b(show|list|give)\b", q) and find_category(q):
        return "show_category", {"category": find_category(q), "sentiment": find_sentiment_filter(q)}
    if find_category(q):
        return "show_category", {"category": find_category(q), "sentiment": find_sentiment_filter(q)}
    return "unknown", {}


def split_multi_part(query):
    """Handle multi-part questions like 'show tech articles and summarize article 3'."""
    parts = re.split(r"\s+(?:and then|and also|then|;)\s+|\s+and\s+(?=(?:show|summar|find|what|translate|classify|list))", query, flags=re.I)
    return [p for p in parts if p and p.strip()]
