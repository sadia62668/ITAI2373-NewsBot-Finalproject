"""Plotting helpers, refactored from the plotting code in the notebook
(category distribution, confusion matrix, POS heatmap, topic/sentiment evolution).
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_category_distribution(df, save_path=None):
    """Bar chart of articles per category (midterm data exploration)."""
    counts = df["category"].value_counts()
    fig = plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="category", order=counts.index)
    plt.title("Distribution of News Categories")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return fig


def plot_confusion_matrix(cm, labels, title="NewsBot 2.0 - Confusion Matrix", save_path=None):
    """Annotated heatmap of a confusion matrix."""
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return fig


def plot_pos_heatmap(pos_summary, save_path=None):
    """Heatmap of POS tag proportions by category (midterm Module 4)."""
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(pos_summary.T, annot=True, cmap="YlOrRd", fmt=".3f")
    plt.title("POS Tag Proportions by News Category")
    plt.xlabel("Category"); plt.ylabel("POS Tag"); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return fig


def plot_topic_evolution(evo, save_path=None):
    """Stacked bar of topic share per period (period = article_id order, NOT real dates)."""
    ax = evo.plot(kind="bar", stacked=True, figsize=(8, 4), colormap="tab10")
    plt.title("Topic share per period (period = article_id order, NOT real dates)")
    plt.ylabel("Share of articles"); plt.legend(title="NMF topic", bbox_to_anchor=(1.02, 1)); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return ax.figure


def plot_sentiment_evolution(evo, save_path=None):
    """Line plot of average VADER compound per period and category."""
    ax = evo.plot(marker="o", figsize=(8, 4))
    plt.title("Sentiment evolution (period = article_id order, NOT real dates)")
    plt.ylabel("Average VADER compound"); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return ax.figure
