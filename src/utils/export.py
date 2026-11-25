"""Report generation (Markdown), from the notebook's ``export_report`` function."""


def export_report(bot, path="newsbot_report.md"):
    """Write a short Markdown report (size, accuracy, topics, insights) and return its path."""
    lines = ["# NewsBot 2.0 Report", f"Articles: {len(bot.df)}",
             f"Classifier accuracy: {bot.test_accuracy:.4f}", "", "## Topics"]
    lines += [f'- {k}: {", ".join(v[:8])}' for k, v in bot.topics().items()] + ["", "## Insights"]
    lines += [f"- {x}" for x in bot.insights()]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
