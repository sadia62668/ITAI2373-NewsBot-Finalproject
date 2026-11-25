"""Module C: cross-language analysis.

Pipeline: detect language -> translate to English -> run our English models -> compare.
Source: notebook section C.
"""
import pandas as pd

from src.data_processing.text_preprocessor import preprocess_text
from src.multilingual.language_detector import LANG_NAMES, detect_language
from src.multilingual.translator import translate


class RawTextClassifier:
    """Small wrapper so the multilingual module can send raw English text to a NewsClassifier."""

    def __init__(self, news_clf):
        self.news_clf = news_clf

    def predict_with_confidence(self, text):
        return self.news_clf.predict_with_confidence(preprocess_text(text))


class MultilingualAnalyzer:
    """Detect language -> translate to English -> run the English NewsBot models."""

    def __init__(self, classifier=None, sentiment=None):
        self.classifier = classifier
        self.sentiment = sentiment

    def analyze(self, text):
        """Return language, confidence, translated flag, English text, and optional category/sentiment."""
        lang, prob = detect_language(text)
        english, translated = (text, False) if lang == "en" else translate(
            text, "en", source=lang if lang != "unknown" else "auto")
        result = {"language": lang, "language_name": LANG_NAMES.get(lang, lang),
                  "confidence": prob, "translated": translated, "english_text": english}
        if self.classifier is not None:
            result["category"], result["category_confidence"] = self.classifier.predict_with_confidence(english)
        if self.sentiment is not None:
            s = self.sentiment.analyze(english)
            result["sentiment"], result["sentiment_score"] = s["label"], s["compound"]
        return result

    def compare(self, texts):
        """Cross-language comparison: analyze a list of texts (same story in different languages)."""
        rows = []
        for t in texts:
            r = self.analyze(t)
            rows.append({k: r.get(k) for k in ["language", "translated", "category",
                                               "category_confidence", "sentiment", "sentiment_score"]})
        return pd.DataFrame(rows)

    def coverage_comparison(self, stories):
        """stories = {"source/lang name": text}; compare category and tone per source."""
        df = self.compare(list(stories.values()))
        df.insert(0, "source", list(stories.keys()))
        return df
