"""Module D: main entry for natural-language questions. Keeps conversation state (context).

Source: notebook section D.
"""
import logging

from src.conversation.intent_classifier import detect_intent, split_multi_part
from src.conversation.response_generator import ResponseGenerator

log = logging.getLogger(__name__)


class QueryProcessor:
    """Split multi-part questions, detect intents, answer, and remember context/history."""

    def __init__(self, bot):
        self.responder = ResponseGenerator(bot)
        self.context = {}      # remembers last results, last category, last article
        self.history = []

    def process(self, query):
        """Answer a (possibly multi-part) question; errors become a friendly message."""
        answers = []
        for part in split_multi_part(query):
            intent, slots = detect_intent(part)
            log.info("query=%r intent=%s slots=%s", part, intent, slots)
            try:
                ans = self.responder.respond(intent, slots, self.context)
            except Exception as e:  # never crash the chat
                ans = f"Something went wrong ({type(e).__name__}). Please try another question."
            self.history.append({"query": part, "intent": intent, "answer": ans})
            answers.append(ans)
        return "\n\n".join(answers)
