"""Module C: translation with deep-translator.

We try Google first, then MyMemory as a backup. If both fail (no internet or
rate limit), the original text is returned so the pipeline never crashes.
Source: notebook section C.
"""
import time

try:
    from deep_translator import GoogleTranslator, MyMemoryTranslator
    HAVE_TRANSLATOR = True
except ImportError:
    HAVE_TRANSLATOR = False

# MyMemory needs full locale codes
MYMEMORY_CODES = {"en": "en-GB", "es": "es-ES", "fr": "fr-FR", "de": "de-DE", "it": "it-IT",
                  "pt": "pt-PT", "ar": "ar-SA", "zh-CN": "zh-CN", "zh-cn": "zh-CN", "nl": "nl-NL"}


def translate(text, target="en", source="auto"):
    """Translate text. Returns (translated_text, ok_flag); ok_flag False means fallback to original."""
    if not HAVE_TRANSLATOR or not str(text).strip():
        return text, False
    text = str(text)[:4500]
    err = None
    try:
        return GoogleTranslator(source=source, target=target).translate(text), True
    except Exception as e:
        err = e
    if source != "auto":  # MyMemory cannot auto-detect
        try:
            time.sleep(0.5)
            out = MyMemoryTranslator(source=MYMEMORY_CODES.get(source, source),
                                     target=MYMEMORY_CODES.get(target, target)).translate(text[:480])
            return out, True
        except Exception as e:
            err = e
    print(f"[translation fallback] {type(err).__name__}: {str(err)[:80]}")
    return text, False
