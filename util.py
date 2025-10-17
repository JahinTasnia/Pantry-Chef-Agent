from __future__ import annotations
from typing import List, Dict, Any
import json, re, os

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None
    from difflib import SequenceMatcher

    class _FallbackFuzz:
        @staticmethod
        def token_set_ratio(a: str, b: str) -> int:
            ta = " ".join(sorted(set(a.split())))
            tb = " ".join(sorted(set(b.split())))
            return int(SequenceMatcher(None, ta, tb).ratio() * 100)

    fuzz = _FallbackFuzz()


STOPWORDS = {
    "fresh",
    "ground",
    "dried",
    "chopped",
    "sliced",
    "skinless",
    "boneless",
    "large",
    "small",
    "medium",
}


UNIT_ALIASES = {
    "tsp": ["tsp", "teaspoon", "teaspoons"],
    "tbsp": ["tbsp", "tablespoon", "tablespoons"],
    "cup": ["cup", "cups"],
    "g": ["g", "gram", "grams"],
    "kg": ["kg", "kilogram", "kilograms"],
    "ml": ["ml", "milliliter", "milliliters"],
    "l": ["l", "liter", "litre", "liters", "litres"],
}


CANONICAL_UNITS = {alias: k for k, vs in UNIT_ALIASES.items() for alias in vs}


INGR_SYNONYMS = {
    "scallion": "spring onion",
    "cilantro": "coriander",
    "capsicum": "bell pepper",
    "chili": "chili",
    "chilli": "chili",
}


def normalize_ingredient(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    parts = [w for w in t.split() if w not in STOPWORDS]
    t = " ".join(parts)
    t = INGR_SYNONYMS.get(t, t)
    return t


def jaccard_score(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def fuzzy_overlap(pantry: List[str], recipe_ingredients: List[str]) -> float:
    total, count = 0.0, 0
    for p in pantry:
        best = 0
        for r in recipe_ingredients:
            if fuzz is not None and hasattr(fuzz, "token_set_ratio"):
                score = fuzz.token_set_ratio(p, r)
            else:
                ta = " ".join(sorted(set(p.split())))
                tb = " ".join(sorted(set(r.split())))
                from difflib import SequenceMatcher

                score = int(SequenceMatcher(None, ta, tb).ratio() * 100)
            best = max(best, score)
        total += best
        count += 1
    return (total / count) / 100.0 if count else 0.0
