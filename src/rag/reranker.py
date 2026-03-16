import json
import openai # type: ignore
from src.config import OPENAI_API_KEY, LLM_MODEL, RERANK_TOP_N

client = openai.OpenAI(api_key=OPENAI_API_KEY)


def llm_rerank(
    detected_ingredients: list[str],
    candidates:           list[dict],
    top_n:                int = RERANK_TOP_N,
    user_preferences:     str = "",
) -> list[dict]:
    """
    Use GPT-4o-mini to re-rank retrieved recipes beyond pure vector similarity.

    Why LLM re-ranking over similarity alone?
        Vector similarity finds semantically related recipes but cannot reason
        about whether you actually have enough of the right ingredients,
        or how the nutritional profile matches your dietary goals.

    The LLM scores each candidate on:
        - Ingredient coverage % — what percentage you already have
        - Missing ingredients   — what you still need to buy (max 3)
        - Nutrition score 1-10  — balanced meal quality assessment
        - Reason                — one-sentence explanation of the match

    Args:
        detected_ingredients: Confirmed ingredients from the vision ensemble
        candidates:           Retrieved recipes from ChromaDB
        top_n:                Number of final recipes to return
        user_preferences:     Free-text dietary preferences

    Returns:
        Top-N ranked recipe dicts with LLM scores merged into metadata
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

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=900,
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed      = json.loads(raw)
        ranked_list = (
            parsed
            if isinstance(parsed, list)
            else parsed.get("recipes",
                 parsed.get("items",
                 parsed.get("recommendations", [])))
        )
    except json.JSONDecodeError:
        # Fallback: return top-N by similarity score
        return candidates[:top_n]

    name_to_meta = {c["name"]: c for c in candidates}
    final        = []
    for item in ranked_list[:top_n]:
        full = name_to_meta.get(item.get("name", ""), {})
        final.append({**full, **item})

    return final