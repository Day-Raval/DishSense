"""
Recipe reranker — two modes controlled by USE_LLM_RERANKER in .env

Mode 1 (USE_LLM_RERANKER=true):
    Uses GPT-4o-mini to reason about ingredient coverage and nutrition.
    Requires a valid OPENAI_API_KEY with available credits.
    Cost: ~$0.0001 per request (very cheap).
    Quality: Best — natural language reasoning.

Mode 2 (USE_LLM_RERANKER=false):
    Uses a free local scoring algorithm.
    No API key needed, instant response.
    Quality: Good — rule-based coverage + similarity scoring.

Switch between modes by editing USE_LLM_RERANKER in your .env file.
"""

import json
import os
from src.config import OPENAI_API_KEY, LLM_MODEL, RERANK_TOP_N

# ── Read switch from environment ──────────────────────────────────────────────
# Set USE_LLM_RERANKER=true in .env to use GPT-4o-mini
# Set USE_LLM_RERANKER=false to use the free local reranker
_use_llm = os.getenv("USE_LLM_RERANKER", "false").strip().lower() == "true"

# Only import openai if LLM mode is enabled — avoids errors when key is missing
if _use_llm:
    import openai
    _client = openai.OpenAI(api_key=OPENAI_API_KEY)


# ── Mode 1: LLM reranker (GPT-4o-mini) ───────────────────────────────────────

def _llm_rerank(
    detected_ingredients: list[str],
    candidates:           list[dict],
    top_n:                int,
    user_preferences:     str,
) -> list[dict]:
    """
    Rerank recipes using GPT-4o-mini.

    Requires:
        - OPENAI_API_KEY set in .env with valid credits
        - USE_LLM_RERANKER=true in .env

    The LLM reasons about:
        1. Ingredient coverage % — how many detected ingredients are used
        2. Missing ingredients   — what still needs to be bought (max 3)
        3. Nutrition score 1-10  — how balanced the meal is
        4. Match reason          — one-sentence explanation
    """
    candidates_str = "\n".join([
        f"{i+1}. {c['name']} | "
        f"calories: {c['calories']:.0f} | "
        f"protein: {c['protein_g']:.1f}g | "
        f"time: {c['minutes']}min | "
        f"needs: {', '.join(str(x) for x in c['ingredients'][:8])} | "
        f"similarity: {c['similarity_score']:.2f}"
        for i, c in enumerate(candidates[:15])
    ])

    pref_line = f"User preferences: {user_preferences}\n" if user_preferences else ""

    prompt = f"""You are a recipe recommendation assistant.

The user has these ingredients available in their fridge:
{', '.join(detected_ingredients)}

Candidate recipes:
{candidates_str}

{pref_line}Rank the top {top_n} recipes. For each recipe provide:
1. coverage_pct: what percentage of the needed ingredients the user already has (integer 0-100)
2. missing_ingredients: key items still needed to buy, max 3 (empty list if none)
3. nutrition_score: balanced meal quality rating from 1 to 10 (integer)
4. reason: one sentence explaining why this is a good match

Return ONLY a valid JSON array, no markdown, no extra text:
[
  {{
    "rank": 1,
    "name": "exact recipe name from list above",
    "coverage_pct": 85,
    "missing_ingredients": ["item1"],
    "nutrition_score": 8,
    "reason": "Uses most of your available ingredients and is high in protein."
  }}
]"""

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=900,
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        ranked_list = (
            parsed
            if isinstance(parsed, list)
            else parsed.get("recipes",
                 parsed.get("items",
                 parsed.get("recommendations", [])))
        )
    except json.JSONDecodeError:
        # LLM returned malformed JSON — fall back to local reranker
        print("[reranker] LLM returned malformed JSON — falling back to local reranker")
        return _local_rerank(detected_ingredients, candidates, top_n, user_preferences)

    name_to_meta = {c["name"]: c for c in candidates}
    final = []
    for item in ranked_list[:top_n]:
        full = name_to_meta.get(item.get("name", ""), {})
        final.append({**full, **item})

    return final


# ── Mode 2: Local reranker (free, no API key needed) ─────────────────────────

def _local_rerank(
    detected_ingredients: list[str],
    candidates:           list[dict],
    top_n:                int,
    user_preferences:     str,
) -> list[dict]:
    """
    Rerank recipes using a free local scoring algorithm.

    Does NOT require an OpenAI API key.
    Set USE_LLM_RERANKER=false in .env to use this mode.

    Scoring formula:
        combined_score = 0.5 * similarity_score + 0.5 * coverage_pct
        - similarity_score: semantic similarity from ChromaDB vector search
        - coverage_pct:     % of recipe ingredients the user already has

    Nutrition score is estimated from protein content:
        - 0-10g protein  → score 1-3  (low)
        - 10-30g protein → score 4-6  (medium)
        - 30g+ protein   → score 7-10 (high)
    """
    detected_set = set(
        ing.lower().replace("_", " ")
        for ing in detected_ingredients
    )

    scored = []
    for c in candidates:
        recipe_ings = [str(i).lower() for i in c.get("ingredients", [])]

        # Count matched ingredients (partial match allowed)
        matched = sum(
            1 for ing in detected_set
            if any(ing in r or r in ing for r in recipe_ings)
        )

        total        = max(len(recipe_ings), 1)
        coverage_pct = round((matched / total) * 100)

        # Find key missing ingredients (max 3)
        missing = [
            ing for ing in recipe_ings[:8]
            if not any(ing in d or d in ing for d in detected_set)
        ][:3]

        # Estimate nutrition score from protein content
        protein       = float(c.get("protein_g", 0))
        nutrition_score = min(10, max(1, round(protein / 4)))

        # Build match reason based on user preferences
        reason_parts = [f"Matches {coverage_pct}% of your ingredients"]
        if protein > 20:
            reason_parts.append(f"high protein ({protein:.0f}g)")
        if c.get("minutes", 99) <= 30:
            reason_parts.append(f"quick to make ({c.get('minutes')} min)")
        if user_preferences:
            reason_parts.append(f"aligns with '{user_preferences}'")

        reason = " — ".join(reason_parts) + "."

        # Combined score: equal weight to semantic similarity and coverage
        combined = (
            0.5 * float(c.get("similarity_score", 0)) +
            0.5 * (coverage_pct / 100)
        )

        scored.append({
            **c,
            "coverage_pct":        coverage_pct,
            "missing_ingredients": missing,
            "nutrition_score":     nutrition_score,
            "reason":              reason,
            "_combined_score":     combined,
        })

    # Sort by combined score descending
    scored.sort(key=lambda x: -x["_combined_score"])

    # Assign ranks and clean up internal scoring field
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
    Main reranker entry point.

    Automatically routes to LLM or local reranker based on
    USE_LLM_RERANKER setting in your .env file.

    To switch modes:
        LLM mode (GPT-4o-mini):  set USE_LLM_RERANKER=true  in .env
        Local mode (free):       set USE_LLM_RERANKER=false in .env

    Args:
        detected_ingredients: Confirmed ingredients from vision ensemble
        candidates:           Retrieved recipes from ChromaDB
        top_n:                Number of final recipes to return
        user_preferences:     Free-text dietary preferences

    Returns:
        Top-N ranked recipe dicts with scores merged in
    """
    if _use_llm:
        print("[reranker] Mode: LLM (GPT-4o-mini)")
        try:
            return _llm_rerank(
                detected_ingredients, candidates, top_n, user_preferences
            )
        except Exception as e:
            # Graceful fallback if LLM fails (rate limit, quota, network)
            print(f"[reranker] LLM failed ({e.__class__.__name__}: {e}) "
                  f"— falling back to local reranker")
            return _local_rerank(
                detected_ingredients, candidates, top_n, user_preferences
            )
    else:
        print("[reranker] Mode: Local (free, no API key needed)")
        return _local_rerank(
            detected_ingredients, candidates, top_n, user_preferences
        )