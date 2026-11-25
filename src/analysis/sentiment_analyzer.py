"""Module A: sentiment analysis (VADER, same as midterm) + sentiment evolution.

Source: notebook section A3.
"""
import nltk

try:
    nltk.download("vader_lexicon", quiet=True)
except Exception:
    pass
from nltk.sentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """VADER sentiment with a label: compound >= 0.05 positive, <= -0.05 negative, else neutral."""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze(self, text):
        """Return VADER scores dict (neg, neu, pos, compound) plus ``label``."""
        s = self.sia.polarity_scores(str(text))
        if s["compound"] >= 0.05:
            s["label"] = "positive"
        elif s["compound"] <= -0.05:
            s["label"] = "negative"
        else:
            s["label"] = "neutral"
        return s

    def analyze_df(self, df, text_col="content"):
        """Score every row; returns article_id, category, compound, label."""
        scores = df[text_col].map(self.analyze)
        out = df[["article_id", "category"]].copy()
        out["compound"] = scores.map(lambda s: s["compound"])
        out["label"] = scores.map(lambda s: s["label"])
        return out

    def sentiment_evolution(self, sent_df, time_col="period"):
        """Average sentiment per time period and category.

        NOTE: the BBC dataset has no dates, so the notebook uses article_id order
        split into periods as a stand-in for time. With a dated dataset, pass a
        real date column.
        """
        return sent_df.groupby([time_col, "category"])["compound"].mean().unstack()
