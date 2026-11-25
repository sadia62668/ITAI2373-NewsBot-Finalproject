"""Module A: enhanced classification with confidence scores.

Source: notebook section A1. TF-IDF + model in one scikit-learn Pipeline,
grid search over C for Logistic Regression, confidence and top-k output.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TFIDF_PARAMS = dict(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)


def make_pipeline(model):
    """TF-IDF (final-project settings) followed by ``model``."""
    return Pipeline([("tfidf", TfidfVectorizer(**TFIDF_PARAMS)), ("clf", model)])


class NewsClassifier:
    """TF-IDF + Logistic Regression, tuned with a small grid search.

    Inputs are *preprocessed* texts (see ``text_preprocessor.preprocess_text``).
    """

    def __init__(self):
        self.model = None
        self.results = {}
        self.best_params = {}

    def compare_models(self, X_train, y_train, X_test, y_test):
        """Train Naive Bayes, Logistic Regression and Linear SVM; return accuracy and macro F1."""
        models = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Linear SVM": LinearSVC(),
        }
        for name, m in models.items():
            pipe = make_pipeline(m).fit(X_train, y_train)
            pred = pipe.predict(X_test)
            self.results[name] = {"accuracy": accuracy_score(y_test, pred),
                                  "macro_f1": f1_score(y_test, pred, average="macro")}
        return self.results

    def fit(self, X_train, y_train, tune=True, c_grid=(1, 5, 10, 20)):
        """Fit Logistic Regression; with ``tune`` run 3-fold GridSearchCV over C."""
        pipe = make_pipeline(LogisticRegression(max_iter=2000))
        if tune:
            grid = GridSearchCV(pipe, {"clf__C": list(c_grid)}, cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            self.best_params = grid.best_params_
        else:
            self.model = pipe.fit(X_train, y_train)
            self.best_params = {}
        return self

    def evaluate(self, X_test, y_test):
        """Return (accuracy, classification_report string)."""
        pred = self.model.predict(X_test)
        return accuracy_score(y_test, pred), classification_report(y_test, pred)

    def predict(self, texts):
        """Predict categories for a list of preprocessed texts."""
        return self.model.predict(texts)

    def predict_with_confidence(self, text):
        """Return (category, probability) for one preprocessed text."""
        probs = self.model.predict_proba([text])[0]
        i = int(np.argmax(probs))
        return self.model.classes_[i], float(probs[i])

    def predict_top_k(self, text, k=2):
        """Multi-level output: the top-k categories with probabilities."""
        probs = self.model.predict_proba([text])[0]
        order = np.argsort(probs)[::-1][:k]
        return [(self.model.classes_[i], float(probs[i])) for i in order]

    def save(self, path):
        """Save the fitted pipeline with joblib."""
        import joblib
        joblib.dump(self.model, path)

    def load(self, path):
        """Load a pipeline saved with :meth:`save`."""
        import joblib
        self.model = joblib.load(path)
        return self
