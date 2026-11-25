from src.analysis.classifier import NewsClassifier
from src.analysis.sentiment_analyzer import SentimentAnalyzer

TEXTS = ["football match goal team win cup player"] * 6 + ["bank market share profit company economy"] * 6
LABELS = ["sport"] * 6 + ["business"] * 6


def test_classifier_fit_predict():
    clf = NewsClassifier().fit(TEXTS, LABELS, tune=False)
    cat, conf = clf.predict_with_confidence("team win the cup match")
    assert cat == "sport" and 0 <= conf <= 1
    assert len(clf.predict_top_k("bank profit", k=2)) == 2


def test_sentiment_labels():
    sa = SentimentAnalyzer()
    assert sa.analyze("This is a wonderful, great success")["label"] == "positive"
    assert sa.analyze("This is a terrible, awful disaster")["label"] == "negative"
