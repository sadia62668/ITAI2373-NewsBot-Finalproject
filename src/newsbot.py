"""NewsBot 2.0: connects every module (A-D) into one system.

In the notebook ``NewsBot2`` reused objects trained in earlier cells (globals);
here ``train()`` performs those same notebook steps so the class is
self-contained: validate -> preprocess -> split -> tuned classifier ->
VADER -> NMF topics -> search index -> multilingual -> chat.

Run ``python -m src.newsbot`` from the project root to chat in the terminal.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split  # noqa: E402

from config import settings  # noqa: E402
from src.analysis.classifier import NewsClassifier  # noqa: E402
from src.analysis.sentiment_analyzer import SentimentAnalyzer  # noqa: E402
from src.analysis.topic_modeler import TopicModeler  # noqa: E402
from src.conversation.query_processor import QueryProcessor  # noqa: E402
from src.data_processing.data_validator import load_bbc  # noqa: E402
from src.data_processing.text_preprocessor import preprocess_text  # noqa: E402
from src.language_models.embeddings import SemanticSearch  # noqa: E402
from src.language_models.generator import generate_insights  # noqa: E402
from src.language_models.summarizer import summarize  # noqa: E402
from src.multilingual.cross_lingual_analyzer import MultilingualAnalyzer, RawTextClassifier  # noqa: E402
from src.utils.export import export_report  # noqa: E402


class NewsBot2:
    """NewsBot 2.0 system. ``df`` needs columns article_id, content, category."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.trained = False

    def train(self, tune=True):
        """Train all components (same settings as the notebook). Returns self."""
        df = self.df
        df["processed"] = df["content"].apply(preprocess_text)
        X_train, X_test, y_train, y_test = train_test_split(
            df["processed"], df["category"], test_size=settings.TEST_SIZE,
            random_state=settings.RANDOM_STATE, stratify=df["category"])
        self.classifier = NewsClassifier().fit(X_train, y_train, tune=tune, c_grid=settings.LOGREG_C_GRID)
        self.test_accuracy, self.test_report = self.classifier.evaluate(X_test, y_test)
        self.sentiment = SentimentAnalyzer()
        self.sent_df = self.sentiment.analyze_df(df)
        self.sentiment_scores = self.sent_df["compound"].set_axis(df.index)
        self.sentiment_labels = self.sent_df["label"].set_axis(df.index)
        self.topic_model = TopicModeler(n_topics=settings.N_TOPICS, method=settings.TOPIC_METHOD).fit(df["processed"])
        self.searcher = SemanticSearch(use_embeddings=settings.USE_EMBEDDINGS).fit(df["content"])
        self.multi = MultilingualAnalyzer(RawTextClassifier(self.classifier), self.sentiment)
        self.chat = QueryProcessor(self)
        self.trained = True
        return self

    def analyze_article(self, text):
        """Category, confidence, sentiment label and 2-sentence summary for raw text."""
        cat, conf = self.classifier.predict_with_confidence(preprocess_text(text))
        return {"category": cat, "confidence": conf, "sentiment": self.sentiment.analyze(text)["label"],
                "summary": summarize(text, 2)}

    def summarize(self, text, n=3):
        return summarize(text, n)

    def search(self, query, top_k=5):
        return self.searcher.search(query, top_k)

    def topics(self, n=10):
        return self.topic_model.top_words(n)

    def insights(self):
        return generate_insights(self.df, self.sent_df, self.topics())

    def export_report(self, path="newsbot_report.md"):
        return export_report(self, path)

    def ask(self, query):
        """Answer a natural-language question via the conversational interface."""
        return self.chat.process(query)


if __name__ == "__main__":
    bot = NewsBot2(load_bbc(settings.RAW_DATA)).train(tune=False)
    print(f"NewsBot 2.0 ready ({len(bot.df)} articles, accuracy {bot.test_accuracy:.4f}). Type 'quit' to stop.")
    while True:
        q = input("You: ")
        if q.lower() in ("quit", "exit"):
            break
        print("NewsBot:", bot.ask(q))
