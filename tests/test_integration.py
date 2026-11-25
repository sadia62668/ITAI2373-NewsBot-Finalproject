import pytest

from config import settings
from src.conversation.intent_classifier import detect_intent, split_multi_part
from src.language_models.summarizer import summarize
from src.multilingual.language_detector import HAVE_LANGDETECT, detect_language


def test_intents():
    assert detect_intent("summarize article 5") == ("summarize", {"index": 5})
    assert detect_intent("show sports articles")[0] == "show_category"
    assert detect_intent("who is tony blair")[0] == "entity"


def test_multi_part_split():
    assert len(split_multi_part("show tech articles and summarize article 3")) == 2


def test_summary_length():
    text = " ".join(f"Sentence number {i} talks about the economy and markets." for i in range(10))
    assert summarize(text, 3).count("Sentence number") == 3


@pytest.mark.skipif(not HAVE_LANGDETECT, reason="langdetect not installed")
def test_language_detection():
    assert detect_language("El gobierno anunció nuevos impuestos.")[0] == "es"


@pytest.mark.skipif(not settings.RAW_DATA.exists(), reason="dataset not available")
def test_full_system_on_real_data():
    from src.data_processing.data_validator import load_bbc
    from src.newsbot import NewsBot2
    bot = NewsBot2(load_bbc(settings.RAW_DATA)).train(tune=False)
    assert len(bot.df) == 1440
    assert bot.test_accuracy > 0.9
    assert "sport" in bot.ask("show sports articles")
