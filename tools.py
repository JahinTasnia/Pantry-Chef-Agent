from __future__ import annotations
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from util import normalize_ingredient, jaccard_score, fuzzy_overlap
import json
import os

RECIPES_PATH = os.environ.get("RECIPES_PATH", os.path.join("data", "recipes.json"))


def load_recipes(path: str) -> List[Dict[str, Any]]:
    """
    Load recipes from a JSON file and return a list of recipe objects.
    Accepts either a top-level list or a dict containing a "recipes" key.
    If the file is missing or invalid, returns an empty list.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    except Exception:
        return []

    if isinstance(data, list):
        return data
    if (
        isinstance(data, dict)
        and "recipes" in data
        and isinstance(data["recipes"], list)
    ):
        return data["recipes"]
    return []


RECIPES_PATH = os.environ.get("RECIPES_PATH", os.path.join("data", "recipes.json"))


class SearchRecipesInput(BaseModel):
    pantry_items: List[str] = Field(..., description="List of ingredients available")
    top_k: int = Field(5, ge=1, le=20, description="Number of top candidates to return")


# @tool("search_recipes", args_schema=SearchRecipesInput)
# def search_recipes(pantry_items: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
#     """
#     Return top-K recipe candidates that best match the pantry items.
#     Uses normalized fuzzy + Jaccard overlap scoring against a local JSON corpus.
#     """
#     recipes = load_recipes(RECIPES_PATH)
#     pantry = [normalize_ingredient(p) for p in pantry_items]
#     scored: List[Dict[str, Any]] = []

#     for r in recipes:
#         r_norm = [normalize_ingredient(x) for x in r.get("ingredients", [])]
#         score = 0.6 * jaccard_score(pantry, r_norm) + 0.4 * fuzzy_overlap(
#             pantry, r_norm
#         )
#         missing = [x for x in r_norm if x not in pantry]
#         scored.append(
#             {
#                 "id": r.get("id"),
#                 "title": r.get("title"),
#                 "score": round(float(score), 3),
#                 "missing": missing,
#                 "recipe": r,
#             }
#         )

#     scored.sort(key=lambda x: x["score"], reverse=True)
#     return scored[:top_k]


@tool("search_recipes", args_schema=SearchRecipesInput)
def search_recipes(pantry_items: List[str], top_k: int = 5) -> str:
    """
    Search local recipe database and return the best matches for the user's pantry items.
    Combines fuzzy matching and Jaccard similarity to find top recipes.
    Returns JSON-formatted list with title, score, and missing ingredients.
    """
    recipes = load_recipes(RECIPES_PATH)
    pantry = [normalize_ingredient(p) for p in pantry_items]
    scored = []
    for r in recipes:
        r_norm = [normalize_ingredient(x) for x in r.get("ingredients", [])]
        score = 0.6 * jaccard_score(pantry, r_norm) + 0.4 * fuzzy_overlap(
            pantry, r_norm
        )
        missing = [x for x in r_norm if x not in pantry]
        scored.append(
            {"title": r["title"], "score": round(score, 3), "missing": missing}
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return json.dumps(scored[:top_k], indent=2)


class ScaleRecipeInput(BaseModel):
    base_servings: int = Field(..., ge=1)
    target_servings: int = Field(..., ge=1)
    ingredients: List[str] = Field(
        ..., description="Lines like '2 tbsp oil' or '1 egg'"
    )


@tool("scale_recipe", args_schema=ScaleRecipeInput)
def scale_recipe(
    base_servings: int, target_servings: int, ingredients: List[str]
) -> str:
    """
    Scale ingredient quantities for the target number of servings.
    Returns a JSON-formatted list of scaled ingredient lines.
    """
    factor = target_servings / max(base_servings, 1)
    import re

    def scale_line(line: str) -> str:
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(\w+)?\s+(.*)", line)
        if not m:
            return line
        qty_s, unit, name = m.groups()
        try:
            qty = float(qty_s) * factor
        except ValueError:
            return line
        unit = (unit or "").strip()
        return f"{qty:.2f} {unit} {name}".strip()

    lines = [scale_line(x) for x in ingredients]
    return json.dumps(lines, indent=2)


class SubstitutionInput(BaseModel):
    missing_items: List[str]


SUBS: Dict[str, List[str]] = {
    "egg": ["flax egg (1 tbsp ground flax + 3 tbsp water)", "silken tofu"],
    "butter": ["oil", "ghee", "margarine"],
    "milk": ["evaporated milk", "plant milk", "water + milk powder"],
    "garlic": ["garlic powder", "asafoetida (pinch)"],
    "lemon": ["lime", "vinegar + pinch sugar"],
}


@tool("suggest_substitutions", args_schema=SubstitutionInput)
def suggest_substitutions(missing_items: List[str]) -> str:
    """
    Suggest practical ingredient substitutions for missing items.
    Returns a JSON dictionary mapping each missing item to substitute options.
    """
    out: Dict[str, List[str]] = {}
    for item in missing_items:
        key = item.lower().strip()
        out[item] = SUBS.get(key, [])
    return json.dumps(out, indent=2)


class ShoppingListInput(BaseModel):
    missing_items: List[str]


@tool("shopping_list", args_schema=ShoppingListInput)
def shopping_list(missing_items: List[str]) -> str:
    """
    Generate a de-duplicated shopping list from missing ingredients.
    Returns a JSON-formatted list.
    """
    seen, out = set(), []
    for m in missing_items:
        k = m.lower().strip()
        if k not in seen:
            out.append(m)
            seen.add(k)
    return json.dumps(out, indent=2)


TOOLS = [search_recipes, scale_recipe, suggest_substitutions, shopping_list]
__all__ = [
    "search_recipes",
    "scale_recipe",
    "suggest_substitutions",
    "shopping_list",
    "TOOLS",
]
