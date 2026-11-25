"""Module A: topic discovery with LDA and NMF, topic evolution and clustering.

Source: notebook sections A2-A3.
"""
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TopicModeler:
    """LDA (on word counts) or NMF (on TF-IDF) with 3,000-word vocabulary."""

    def __init__(self, n_topics=5, method="nmf", random_state=42):
        self.n_topics = n_topics
        self.method = method
        if method == "lda":
            self.vectorizer = CountVectorizer(max_features=3000, min_df=5, max_df=0.7)
            self.model = LatentDirichletAllocation(n_components=n_topics, random_state=random_state,
                                                   learning_method="batch")
        else:
            self.vectorizer = TfidfVectorizer(max_features=3000, min_df=5, max_df=0.7)
            self.model = NMF(n_components=n_topics, random_state=random_state, init="nndsvd", max_iter=400)

    def fit(self, processed_texts):
        """Fit on preprocessed texts; stores ``dtm`` and ``doc_topics``."""
        self.dtm = self.vectorizer.fit_transform(processed_texts)
        self.doc_topics = self.model.fit_transform(self.dtm)
        return self

    def fit_transform(self, documents):
        """Train topic model and return the document-topic matrix."""
        return self.fit(documents).doc_topics

    def transform(self, processed_texts):
        """Topic distribution for new texts."""
        return self.model.transform(self.vectorizer.transform(processed_texts))

    def top_words(self, n=10):
        """{"Topic k": [top n words]} for every topic."""
        words = self.vectorizer.get_feature_names_out()
        return {f"Topic {k}": [words[i] for i in comp.argsort()[::-1][:n]]
                for k, comp in enumerate(self.model.components_)}

    def get_topic_words(self, topic_id, n_words=10):
        """Top words of one topic."""
        return self.top_words(n_words)[f"Topic {topic_id}"]

    def dominant_topic(self):
        """Index of the strongest topic for each training document."""
        return self.doc_topics.argmax(axis=1)

    def topic_vs_category(self, categories):
        """Crosstab of dominant topic vs. true category."""
        return pd.crosstab(pd.Series(self.dominant_topic(), name="topic"),
                           pd.Series(list(categories), name="category"))

    def topic_evolution(self, periods):
        """Share of each dominant topic per period (topic trend over 'time')."""
        return pd.crosstab(pd.Series(list(periods), name="period"),
                           pd.Series(self.dominant_topic(), name="topic"), normalize="index")

    def cluster_articles(self, n_clusters=5, random_state=42):
        """Group similar articles using their topic distributions (KMeans)."""
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(self.doc_topics)

    def visualize_topics(self, n_words=8, save_path=None):
        """Bar charts of the top words for each topic."""
        import matplotlib.pyplot as plt
        words = self.vectorizer.get_feature_names_out()
        fig, axes = plt.subplots(1, self.n_topics, figsize=(3.2 * self.n_topics, 4))
        for k, ax in enumerate(axes):
            comp = self.model.components_[k]
            idx = comp.argsort()[::-1][:n_words]
            ax.barh([words[i] for i in idx][::-1], comp[idx][::-1], color="#4C72B0")
            ax.set_title(f"Topic {k}")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120)
        return fig


def topic_purity(crosstab):
    """Share of articles that belong to the majority category of their topic."""
    return float(crosstab.max(axis=1).sum() / crosstab.values.sum())
