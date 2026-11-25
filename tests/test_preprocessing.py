import pandas as pd
import pytest

from src.data_processing.data_validator import validate
from src.data_processing.text_preprocessor import clean_text, preprocess_text, split_sentences


def test_clean_text_removes_urls_and_symbols():
    out = clean_text("Apple Inc. announced record earnings! Visit https://apple.com #TechNews")
    assert "http" not in out and "#" not in out and out == out.lower()


def test_preprocess_removes_stopwords_and_short_words():
    out = preprocess_text("The companies are in the market")
    assert "the" not in out.split() and "company" in out.split()


def test_split_sentences():
    s = split_sentences("This is the first sentence. This is the second sentence here. Hi.")
    assert len(s) == 2


def test_validate_drops_duplicates():
    df = pd.DataFrame({"article_id": [1, 2, 3], "content": ["a b c", "a b c", "d e f"],
                       "category": ["x", "x", "y"]})
    out = validate(df)
    assert len(out) == 2 and out.attrs["validation_report"]["duplicates_removed"] == 1


def test_validate_missing_column():
    with pytest.raises(ValueError):
        validate(pd.DataFrame({"content": ["a"]}))
