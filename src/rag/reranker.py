"""
Recipe reranker — two modes controlled by USE_LLM_RERANKER in .env

Mode 1 (USE_LLM_RERANKER=true):
    Uses GPT-4o-mini with strict typed schema validation.
    Requires OPENAI_API_KEY with credits.

Mode 2 (USE_LLM_RERANKER=false):
    Uses a free local scoring algorithm with corrected nutrition scoring.
    No API key needed, instant response.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from src.config import OPENAI_API_KEY, LLM_MODEL, RERANK_TOP_N

# ── Read switch from environment ──────────────────────────────────────────────
_use_llm = os.getenv("USE_LLM_RERANKER", "false").strip().lower() == "true"

if _use_llm:
    import openai
    _client = openai.OpenAI(api_key=OPENAI_API_KEY)


# ── Strict typed schema for LLM output ────────────────────────

@dataclass
class RankedRecipe:
    """
    Strict typed schema for a single ranked recipe.
    All fields are validated and clamped to valid ranges.
    Prevents bad LLM output from crashing the pipeline.
    """
    rank:                  int
    name:                  str
    coverage_pct:          int            # 0-100
    missing_ingredients:   list[str]      # max 3 items
    nutrition_score:       int            # 1-10
    reason:                str

    @classmethod
    def from_dict(cls, d: dict, fallback_name: str = "") -> "RankedRecipe":
        """
        Parse and validate a single LLM output dict.
        Clamps all numeric fields to valid ranges.
        Falls back to safe defaults on missing fields.
        """
        try:
            rank = int(d.get("rank", 99))
        except (ValueError, TypeError):
            rank = 99

        try:
            coverage = int(d.get("coverage_pct", 0))
            coverage = max(0, min(100, coverage))   # clamp 0-100
        except (ValueError, TypeError):
            coverage = 0

        try:
            nutrition = int(d.get("nutrition_score", 5))
            nutrition = max(1, min(10, nutrition))  # clamp 1-10
        except (ValueError, TypeError):
            nutrition = 5

        missing = d.get("missing_ingredients", [])
        if not isinstance(missing, list):
            missing = []
        missing = [str(m) for m in missing[:3]]    # max 3 items

        reason = str(d.get("reason", "Good ingredient match."))[:300]
        name   = str(d.get("name", fallback_name))

        return cls(
            rank=rank,
            name=name,
            coverage_pct=coverage,
            missing_ingredients=missing,
            nutrition_score=nutrition,
            reason=reason,
        )

    def to_dict(self) -> dict:
        return {
            "rank":                 self.rank,
            "name":                 self.name,
            "coverage_pct":         self.coverage_pct,
            "missing_ingredients":  self.missing_ingredients,
            "nutrition_score":      self.nutrition_score,
            "reason":               self.reason,
        }


def _validate_llm_output(raw: str, candidates: list[dict]) -> list[RankedRecipe]:
    """
    Strictly parse and validate LLM JSON output.

    Handles all common LLM failure modes:
        - JSON wrapped in markdown fences (```json ... ```)
        - Top-level dict instead of array ({"recipes": [...]})
        - Missing fields — filled with safe defaults
        - Out-of-range values — clamped to valid ranges
        - Completely malformed JSON — raises ValueError for fallback
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON: {e}") from e

    # Unwrap common dict wrappers
    if isinstance(parsed, dict):
        for key in ("recipes", "items", "recommendations", "results"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            raise ValueError(f"LLM returned dict with unexpected keys: {list(parsed.keys())}")

    if not isinstance(parsed, list):
        raise ValueError(f"LLM output is not a list, got {type(parsed)}")

    # Validate candidate names exist in our actual results
    valid_names = {c["name"] for c in candidates}
    validated   = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        recipe = RankedRecipe.from_dict(item)
        # Only include recipes that actually exist in our candidates
        if recipe.name not in valid_names:
            # Try to find closest match
            for name in valid_names:
                if recipe.name.lower() in name.lower() or name.lower() in recipe.name.lower():
                    recipe.name = name
                    break
        validated.append(recipe)

    return validated


# ── Enhancement 1: Fixed nutrition scoring ────────────────────────────────────

def _compute_nutrition_score(
    calories:  float,
    protein_g: float,
    fat_g:     float,
    carbs_g:   float,
) -> int:
    """
    Compute a balanced nutrition score from 1 to 10.

    Previous logic: score = protein / 4  → always 1-2, completely wrong.

    New logic uses four factors:
        1. Protein adequacy:  20-40g per serving is ideal
        2. Calorie balance:   300-700 kcal per serving is reasonable
        3. Fat balance:       5-25g per serving
        4. Carb balance:      20-60g per serving

    Each factor contributes 0-2.5 points, summing to 0-10.
    """
    score = 0.0

    # Factor 1 — Protein (0 to 2.5 points)
    if protein_g >= 30:
        score += 2.5   # excellent protein
    elif protein_g >= 20:
        score += 2.0   # good protein
    elif protein_g >= 10:
        score += 1.5   # moderate protein
    elif protein_g >= 5:
        score += 1.0   # low protein
    else:
        score += 0.5   # very low protein

    # Factor 2 — Calories (0 to 2.5 points)
    if 300 <= calories <= 600:
        score += 2.5   # ideal calorie range
    elif 200 <= calories <= 800:
        score += 2.0   # acceptable
    elif 100 <= calories <= 1000:
        score += 1.0   # borderline
    elif calories > 0:
        score += 0.5   # extreme (too low or too high)
    else:
        score += 1.0   # unknown calories — neutral

    # Factor 3 — Fat (0 to 2.5 points)
    if 5 <= fat_g <= 20:
        score += 2.5   # healthy fat range
    elif 2 <= fat_g <= 30:
        score += 1.5   # acceptable
    elif fat_g > 0:
        score += 0.5   # very low or very high
    else:
        score += 1.0   # unknown — neutral

    # Factor 4 — Carbs (0 to 2.5 points)
    if 20 <= carbs_g <= 60:
        score += 2.5   # ideal carb range
    elif 10 <= carbs_g <= 80:
        score += 1.5   # acceptable
    elif carbs_g > 0:
        score += 0.5   # extreme
    else:
        score += 1.0   # unknown — neutral

    return max(1, min(10, round(score)))


# ── Mode 1: LLM reranker (GPT-4o-mini) ───────────────────────────────────────

def _llm_rerank(
    detected_ingredients: list[str],
    candidates:           list[dict],
    top_n:                int,
    user_preferences:     str,
) -> list[dict]:
    """
    Rerank using GPT-4o-mini with strict schema validation.
    Falls back to local reranker on any failure.
    """
    # Pre-compute nutrition scores so LLM has accurate data to reason with
    for c in candidates:
        c["nutrition_score_computed"] = _compute_nutrition_score(
            calories=float(c.get("calories",  0) or 0),
            protein_g=float(c.get("protein_g", 0) or 0),
            fat_g=float(c.get("fat_g",        0) or 0),
            carbs_g=float(c.get("carbs_g",    0) or 0),
        )

    candidates_str = "\n".join([
        f"{i+1}. {c['name']} | "
        f"calories: {c['calories']:.0f} | "
        f"protein: {c['protein_g']:.1f}g | "
        f"fat: {c['fat_g']:.1f}g | "
        f"carbs: {c['carbs_g']:.1f}g | "
        f"time: {c['minutes']}min | "
        f"nutrition_score: {c['nutrition_score_computed']}/10 | "
        f"needs: {', '.join(str(x) for x in c['ingredients'][:8])} | "
        f"similarity: {c['similarity_score']:.2f}"
        for i, c in enumerate(candidates[:15])
    ])

    pref_line = f"User preferences: {user_preferences}\n" if user_preferences else ""

    prompt = f"""You are a recipe recommendation assistant.

The user has these ingredients available:
{', '.join(detected_ingredients)}

Candidate recipes (with pre-computed nutrition scores):
{candidates_str}

{pref_line}Rank the top {top_n} recipes.

IMPORTANT RULES:
- coverage_pct must be an INTEGER between 0 and 100
- nutrition_score must be an INTEGER between 1 and 10
  Use the pre-computed nutrition_score provided above as your guide.
  Do NOT set all scores to 1. Vary them based on actual nutritional content.
- missing_ingredients must be a JSON array of strings, max 3 items
- name must exactly match one of the recipe names in the list above
- reason must be a single sentence

Return ONLY a valid JSON array, no markdown fences, no extra text:
[
  {{
    "rank": 1,
    "name": "exact recipe name",
    "coverage_pct": 85,
    "missing_ingredients": ["item1"],
    "nutrition_score": 7,
    "reason": "High protein meal that uses most of your available ingredients."
  }}
]"""

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,   # lower temp = more consistent structured output
        max_tokens=900,
    )

    raw = response.choices[0].message.content.strip()

    try:
        validated = _validate_llm_output(raw, candidates)
    except ValueError as e:
        print(f"[reranker] Schema validation failed: {e} — falling back to local")
        return _local_rerank(detected_ingredients, candidates, top_n, user_preferences)

    if not validated:
        print("[reranker] No valid recipes after validation — falling back to local")
        return _local_rerank(detected_ingredients, candidates, top_n, user_preferences)

    name_to_meta = {c["name"]: c for c in candidates}
    final = []
    for recipe in validated[:top_n]:
        meta = name_to_meta.get(recipe.name, {})
        # Use computed nutrition score if LLM gave a bad one
        computed = meta.get("nutrition_score_computed", 5)
        if recipe.nutrition_score <= 1:
            recipe.nutrition_score = computed
        entry = {**meta, **recipe.to_dict()}
        entry.pop("nutrition_score_computed", None)
        final.append(entry)

    return final


# ── Mode 2: Local reranker (free, no API key needed) ─────────────────────────

def _local_rerank(
    detected_ingredients: list[str],
    candidates:           list[dict],
    top_n:                int,
    user_preferences:     str,
) -> list[dict]:
    """
    Free local reranker with corrected nutrition scoring.

    Scoring formula:
        combined = 0.4 * similarity_score
                 + 0.4 * coverage_pct
                 + 0.2 * (nutrition_score / 10)
    """
    detected_set = set(
        ing.lower().replace("_", " ")
        for ing in detected_ingredients
    )

    # Parse preference keywords for bonus scoring
    pref_keywords = set(user_preferences.lower().split()) if user_preferences else set()
    high_protein  = bool(pref_keywords & {"protein", "high-protein", "highprotein"})
    low_carb      = bool(pref_keywords & {"low-carb", "lowcarb", "keto"})
    vegetarian    = bool(pref_keywords & {"vegetarian", "vegan", "meatless"})
    quick         = bool(pref_keywords & {"quick", "fast", "easy", "simple"})

    scored = []
    for c in candidates:
        recipe_ings = [str(i).lower() for i in c.get("ingredients", [])]

        # ── Coverage score ────────────────────────────────────────────────────
        matched = sum(
            1 for ing in detected_set
            if any(ing in r or r in ing for r in recipe_ings)
        )
        total        = max(len(recipe_ings), 1)
        coverage_pct = round((matched / total) * 100)

        # Missing ingredients (max 3)
        missing = [
            ing for ing in recipe_ings[:8]
            if not any(ing in d or d in ing for d in detected_set)
        ][:3]

        # ── Nutrition score (fixed) ───────────────────────────────────────────
        nutrition_score = _compute_nutrition_score(
            calories=float(c.get("calories",  0) or 0),
            protein_g=float(c.get("protein_g", 0) or 0),
            fat_g=float(c.get("fat_g",        0) or 0),
            carbs_g=float(c.get("carbs_g",    0) or 0),
        )

        # ── Preference bonus ──────────────────────────────────────────────────
        pref_bonus = 0.0
        protein_g  = float(c.get("protein_g", 0) or 0)
        carbs_g    = float(c.get("carbs_g",   0) or 0)
        minutes    = int(c.get("minutes",     60) or 60)
        category   = str(c.get("category",    "")).lower()

        if high_protein and protein_g >= 20:
            pref_bonus += 0.1
        if low_carb and carbs_g <= 20:
            pref_bonus += 0.1
        if vegetarian and not any(
            m in " ".join(recipe_ings)
            for m in ["chicken", "beef", "pork", "lamb", "fish", "shrimp", "bacon"]
        ):
            pref_bonus += 0.1
        if quick and minutes <= 30:
            pref_bonus += 0.1

        # ── Build reason string ───────────────────────────────────────────────
        reason_parts = [f"Matches {coverage_pct}% of your ingredients"]
        if protein_g >= 20:
            reason_parts.append(f"good protein source ({protein_g:.0f}g)")
        if minutes <= 30:
            reason_parts.append(f"quick to make ({minutes} min)")
        if nutrition_score >= 7:
            reason_parts.append("well-balanced nutritionally")
        reason = " — ".join(reason_parts) + "."

        # ── Combined score ────────────────────────────────────────────────────
        combined = (
            0.40 * float(c.get("similarity_score", 0)) +
            0.40 * (coverage_pct / 100) +
            0.20 * (nutrition_score / 10) +
            pref_bonus
        )

        scored.append({
            **c,
            "coverage_pct":        coverage_pct,
            "missing_ingredients": missing,
            "nutrition_score":     nutrition_score,
            "reason":              reason,
            "_combined_score":     combined,
        })

    scored.sort(key=lambda x: -x["_combined_score"])

    final = []
    for i, item in enumerate(scored[:top_n]):
        item["rank"] = i + 1
        item.pop("_combined_score", None)
        final.append(item)

    return final


# ── Public interface ──────────────────────────────────────────────────────────

def llm_rerank(
    detected_ingredients: list[str],
    candidates:           list[dict],
    top_n:                int = RERANK_TOP_N,
    user_preferences:     str = "",
) -> list[dict]:
    """
    Main reranker entry point. Routes to LLM or local based on .env setting.

    To switch:
        USE_LLM_RERANKER=true  → GPT-4o-mini (best quality, needs API key)
        USE_LLM_RERANKER=false → Local scorer (free, instant, no key needed)

    Restart uvicorn after changing .env.
    """
    if _use_llm:
        print("[reranker] Mode: LLM (GPT-4o-mini)")
        try:
            return _llm_rerank(
                detected_ingredients, candidates, top_n, user_preferences
            )
        except Exception as e:
            print(f"[reranker] LLM failed ({e.__class__.__name__}) "
                  f"— falling back to local reranker")
            return _local_rerank(
                detected_ingredients, candidates, top_n, user_preferences
            )
    else:
        print("[reranker] Mode: Local (free, no API key needed)")
        return _local_rerank(
            detected_ingredients, candidates, top_n, user_preferences
        )