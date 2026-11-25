"""Module B: semantic search with TF-IDF cosine similarity and query expansion
(optional: sentence-transformers embeddings if installed).

Source: notebook section B2.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_processing.text_preprocessor import preprocess_text


class SemanticSearch:
    """Article search index. ``use_embeddings=True`` tries all-MiniLM-L6-v2, else TF-IDF."""

    def __init__(self, use_embeddings=False):
        self.use_embeddings = use_embeddings
        self.encoder = None

    def fit(self, texts):
        """Index raw article texts."""
        self.texts = list(texts)
        if self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
                self.matrix = self.encoder.encode(self.texts, show_progress_bar=False)
                return self
            except Exception as e:
                print("Embeddings not available, using TF-IDF instead:", e)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([preprocess_text(t) for t in self.texts])
        return self

    def search(self, query, top_k=5):
        """Return [(doc_index, cosine_score)] for the best matches (score > 0)."""
        if self.encoder is not None:
            q = self.encoder.encode([query])
        else:
            q = self.vectorizer.transform([preprocess_text(query)])
        sims = cosine_similarity(q, self.matrix).ravel()
        idx = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in idx if sims[i] > 0]

    def similar_to(self, doc_index, top_k=5):
        """Most similar articles to an indexed article (excluding itself)."""
        sims = cosine_similarity(self.matrix[doc_index:doc_index + 1] if self.encoder is None
                                 else self.matrix[[doc_index]], self.matrix).ravel()
        sims[doc_index] = -1
        idx = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in idx]

    def expand_query(self, query, n_terms=3):
        """Query expansion: add words that often appear in the top results (pseudo-relevance feedback)."""
        if self.encoder is not None:
            return query
        hits = self.search(query, top_k=5)
        if not hits:
            return query
        names = self.vectorizer.get_feature_names_out()
        centroid = np.asarray(self.matrix[[i for i, _ in hits]].mean(axis=0)).ravel()
        q_words = set(preprocess_text(query).split())
        extra = [names[i] for i in centroid.argsort()[::-1]
                 if " " not in names[i] and names[i] not in q_words][:n_terms]
        return query + " " + " ".join(extra)
