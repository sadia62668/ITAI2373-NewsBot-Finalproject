"""Module C: language detection with langdetect (seeded for repeatable results).

Source: notebook section C.
"""
try:
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 0
    HAVE_LANGDETECT = True
except ImportError:
    HAVE_LANGDETECT = False

LANG_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "de": "German",
              "it": "Italian", "pt": "Portuguese", "ar": "Arabic", "zh-cn": "Chinese"}


def detect_language(text):
    """Return (language_code, probability); ("unknown", 0.0) if empty or unavailable."""
    if not HAVE_LANGDETECT or not str(text).strip():
        return "unknown", 0.0
    try:
        best = detect_langs(str(text))[0]
        return best.lang, round(best.prob, 3)
    except Exception:
        return "unknown", 0.0
