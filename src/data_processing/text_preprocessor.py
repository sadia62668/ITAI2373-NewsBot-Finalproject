"""Text cleaning and preprocessing (enhanced from our midterm).

Source: final-project notebook, "Preprocessing functions" cell. Same steps as the
midterm (lowercase, strip HTML/URLs/emails/numbers/symbols, stop-word removal,
WordNet lemmatization, drop words of <= 2 letters) plus a sentence splitter used
by the summarizer.
"""
import re

import nltk

for _pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "vader_lexicon"]:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:  # offline: rely on already-installed data
        pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def clean_text(text):
    """Lowercase, remove HTML, URLs, emails, numbers and symbols.

    Args:
        text: raw article text (any type; ``None`` returns "").
    Returns:
        str: cleaned text with single spaces.
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text):
    """Clean + remove stop words + lemmatize + drop very short words.

    Returns:
        str: space-joined processed tokens (input for TF-IDF models).
    """
    tokens = clean_text(text).split()
    tokens = [_lemmatizer.lemmatize(t) for t in tokens if t not in _stop_words]
    return " ".join(t for t in tokens if len(t) > 2)


def split_sentences(text):
    """Split an article into sentences (BBC text uses '. ' between sentences).

    Uses NLTK ``sent_tokenize`` with a regex fallback; sentences with fewer than
    4 words are dropped.
    """
    try:
        sents = nltk.sent_tokenize(str(text))
    except Exception:
        sents = re.split(r"(?<=[.!?])\s+", str(text))
    return [s.strip() for s in sents if len(s.split()) >= 4]
