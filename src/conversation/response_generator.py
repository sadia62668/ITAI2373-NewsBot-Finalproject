"""Module D: turns an intent + slots into a friendly answer using the NewsBot components.

Source: notebook section D.
"""
from src.multilingual.translator import translate

HELP_TEXT = """I can help with:
  - "show positive sports articles"
  - "summarize article 5"
  - "what's the sentiment of business news"
  - "find articles about elections"
  - "who is tony blair"           (entity lookup)
  - "what are the main topics"
  - "translate article 3 to spanish"
  - "classify: <some news text>"
  - "more" / "next"               (follow-up on the last list)
  - "export report"
You can also combine: "show tech articles and summarize article 3"."""

LANG_CODES = {"spanish": "es", "french": "fr", "german": "de", "italian": "it",
              "portuguese": "pt", "arabic": "ar", "chinese": "zh-CN"}


class ResponseGenerator:
    """Dispatch an intent to the right NewsBot2 component and format the answer."""

    def __init__(self, bot):
        self.bot = bot

    def _list(self, sub, title):
        lines = [f"[{i}] {t[:85]}..." for i, t in zip(sub.index, sub["content"])]
        return title + "\n" + "\n".join(lines)

    def respond(self, intent, s, ctx):
        """Answer one intent. ``s`` = slots, ``ctx`` = conversation context dict (mutated)."""
        b, df = self.bot, self.bot.df
        if intent == "greeting":
            return "Hi! I'm NewsBot 2.0. Type 'help' to see what I can do."
        if intent == "help":
            return HELP_TEXT
        if intent == "show_category":
            sub = df[df["category"] == s["category"]]
            if s.get("sentiment"):
                sub = sub[b.sentiment_labels.loc[sub.index] == s["sentiment"]]
            ctx["last_results"], ctx["offset"] = list(sub.index), 5
            ctx["last_category"] = s["category"]
            mood = f"{s['sentiment']} " if s.get("sentiment") else ""
            return self._list(sub.head(5), f"Here are 5 of {len(sub)} {mood}{s['category']} articles:")
        if intent == "follow_up":
            res, off = ctx.get("last_results", []), ctx.get("offset", 0)
            if not res:
                return "There is nothing to continue yet. Try 'show tech articles' first."
            ctx["offset"] = off + 5
            nxt = res[off:off + 5]
            return self._list(df.loc[nxt], "Next results:") if nxt else "No more results."
        if intent == "summarize":
            i = s["index"]
            if i not in df.index:
                return f"Sorry, article {i} does not exist (0-{len(df) - 1})."
            ctx["last_article"] = i
            return f"Summary of article {i} ({df.loc[i, 'category']}):\n" + b.summarize(df.loc[i, "content"])
        if intent == "sentiment":
            if s.get("index") is not None and s["index"] in df.index:
                r = b.sentiment.analyze(df.loc[s["index"], "content"])
                return f"Article {s['index']} is {r['label']} (compound={r['compound']:.2f})."
            cat = s.get("category") or ctx.get("last_category")
            mask = df["category"] == cat if cat else df["category"].notna()
            scores = b.sentiment_scores[mask]
            return (f"Sentiment of {cat or 'all'} news: average compound {scores.mean():.2f}, "
                    f"{(scores >= 0.05).mean() * 100:.0f}% positive articles.")
        if intent == "search":
            q = b.searcher.expand_query(s["query"])
            hits = b.search(q, top_k=5)
            ctx["last_results"], ctx["offset"] = [i for i, _ in hits], 5
            if not hits:
                return "No matching articles found."
            return f"Top articles about '{s['query']}' (expanded query: '{q}'):\n" + "\n".join(
                f"[{i}] ({df.loc[i, 'category']}, score {sc:.2f}) {df.loc[i, 'content'][:75]}..." for i, sc in hits)
        if intent == "entity":
            name = s["name"].lower()
            mask = df["content"].str.lower().str.contains(name, regex=False)
            if not mask.any():
                return f"I could not find '{s['name']}' in the articles."
            cats = df[mask]["category"].value_counts().to_dict()
            mood = b.sentiment_scores[mask].mean()
            ctx["last_results"], ctx["offset"] = list(df[mask].index), 5
            return (f"'{s['name']}' appears in {int(mask.sum())} articles {cats}. "
                    f"Average tone of those articles: {mood:.2f}. Say 'more' to list them.")
        if intent == "topics":
            return "Main topics:\n" + "\n".join(f"{k}: {', '.join(v[:8])}" for k, v in b.topics().items())
        if intent == "translate":
            i = s["index"]
            if i not in df.index:
                return f"Sorry, article {i} does not exist."
            code = LANG_CODES.get(s["lang"], s["lang"])
            text, ok = translate(b.summarize(df.loc[i, "content"], 2), target=code, source="en")
            return text if ok else "Translation service not available right now (offline?). Original:\n" + text
        if intent == "classify":
            if not s["text"].strip():
                return "Please give me some text, e.g. 'classify: the team won the cup final'."
            r = b.analyze_article(s["text"])
            return f"Category: {r['category']} ({r['confidence']:.0%}), sentiment: {r['sentiment']}."
        if intent == "export":
            path = b.export_report()
            return f"Report saved to {path}"
        return "Sorry, I did not understand. Type 'help' for examples."
