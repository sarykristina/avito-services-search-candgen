"""
Text preprocessing helpers.

The corpus is Russian, informal, service-marketplace text (titles,
descriptions, filter values). We deliberately keep preprocessing very
simple and fully local (no downloads, no external morphology models):

    - lowercase
    - normalize `ё` -> `е` (a very common informal spelling variant)
    - a small hand-written Russian stopword list (function words carry
      almost no signal for this kind of short-query retrieval and just
      add noise / inflate document length)

We do NOT stem/lemmatize. Avito service ads are full of names, brands,
professional/technical jargon and rare compound words where a generic
stemmer is more likely to merge unrelated words than to help recall.
Plain surface tokens already give strong lexical overlap for this task
(see notebook error analysis for confirmation).
"""

import re

# A short list of the most frequent Russian function words. Not exhaustive
# on purpose -- the goal is only to stop them from dominating term
# frequencies, IDF already downweights any other very common word.
RU_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него",
    "до", "вас", "нибудь", "опять", "уж", "вам", "сказал", "ведь", "там",
    "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть",
    "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб",
    "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж",
    "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем",
    "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее",
    "сейчас", "были", "куда", "зачем", "всех", "никогда", "можно", "при",
    "об", "какая", "который", "которая", "которые", "которых",
}

_TOKEN_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list:
    """Lowercase + extract alphanumeric tokens (Cyrillic/Latin/digits),
    dropping stopwords and single-character tokens."""
    if not text:
        return []
    text = text.lower().replace("ё", "е")  # ё -> е
    tokens = _TOKEN_RE.findall(text)
    return [t for t in tokens if len(t) > 1 and t not in RU_STOPWORDS]


def build_weighted_text(parts_with_weights) -> str:
    """Concatenate several text fields into one bag-of-words string,
    repeating a field's tokens `weight` times to make it count more in
    plain term-frequency scoring (a cheap stand-in for per-field BM25).

    parts_with_weights: iterable of (raw_text, weight:int)
    """
    tokens = []
    for raw_text, weight in parts_with_weights:
        toks = tokenize(raw_text)
        if weight > 1:
            toks = toks * weight
        tokens.extend(toks)
    return " ".join(tokens)
