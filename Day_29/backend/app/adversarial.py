"""
Part C extension -- a small, honest adversarial-robustness demo. Applies a
real, documented evasion technique (leetspeak substitution of common
phishing trigger words, the same trick real spam/phishing authors use to
dodge keyword filters) to the user's own text, then re-runs the same model
on the perturbed version. Both predictions are genuine model calls -- this
never fabricates a "before/after" comparison.
"""
import re

LEET_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}

TRIGGER_WORDS = [
    "verify", "click", "urgent", "account", "suspend", "suspended",
    "password", "confirm", "immediately", "bank", "security", "login",
    "update", "expire", "expires", "alert",
]


def _leetspeak(word: str) -> str:
    return "".join(LEET_MAP.get(ch.lower(), ch) for ch in word)


def perturb(text: str) -> dict:
    replaced = []

    def _sub(match):
        word = match.group(0)
        leet = _leetspeak(word)
        if leet != word.lower():
            replaced.append(word)
            return leet
        return word

    pattern = r'\b(' + '|'.join(TRIGGER_WORDS) + r')\b'
    perturbed_text = re.sub(pattern, _sub, text, flags=re.IGNORECASE)
    return dict(perturbed_text=perturbed_text, replaced_words=replaced)
