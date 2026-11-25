"""Data quality checks before analysis.

Source: notebook cell "Data quality check (new)". Found 50 exact duplicate
articles in the Kaggle BBC training file (1,490 -> 1,440 articles).
"""
import logging

import pandas as pd

log = logging.getLogger(__name__)
REQUIRED = ["article_id", "content", "category"]


def load_bbc(path):
    """Load the Kaggle 'BBC News Train.csv', rename columns like the midterm, and validate.

    Args:
        path: path to the CSV (columns ArticleId, Text, Category).
    Returns:
        pandas.DataFrame with columns article_id, content, category (validated).
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"ArticleId": "article_id", "Text": "content", "Category": "category"})
    return validate(df)


def validate(df):
    """Drop missing/empty/duplicate articles and attach a short report.

    The report is stored in ``df.attrs["validation_report"]`` with keys
    rows_in, rows_out, duplicates_removed, categories.

    Raises:
        ValueError: if a required column is missing.
    """
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    before = len(df)
    df = df.dropna(subset=["content", "category"])
    df = df[df["content"].str.strip().str.len() > 0]
    n_dup = int(df.duplicated(subset="content").sum())
    df = df.drop_duplicates(subset="content").reset_index(drop=True)
    report = {"rows_in": before, "rows_out": len(df), "duplicates_removed": n_dup,
              "categories": df["category"].value_counts().to_dict()}
    log.info("Data validation: %s", report)
    df.attrs["validation_report"] = report
    return df
