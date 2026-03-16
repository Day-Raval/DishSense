from src.vision.yolo_detector import detect_yolo
from src.vision.detr_detector import detect_detr
from src.vision.clip_detector import detect_clip
from src.vision.ensemble      import fuse_detections
from src.rag.retriever        import retrieve_recipes
from src.rag.reranker         import llm_rerank


def recommend_from_photo(
    image_path:       str,
    user_preferences: str         = "",
    max_calories:     float | None = None,
    max_minutes:      int   | None = None,
    top_n:            int          = 5,
) -> dict:
    """
    Full end-to-end FridgeRAG pipeline:

        image path
            → [YOLOv8 | DETR | CLIP]   (three vision models in parallel)
            → ensemble fusion           (deduplicate + confidence boost)
            → ChromaDB semantic search  (top-20 candidate recipes)
            → GPT-4o-mini re-ranking    (ingredient coverage + nutrition)
            → ranked recipe list

    Args:
        image_path:       Absolute or relative path to the fridge photo
        user_preferences: Free-text e.g. "high protein, low carb, vegetarian"
        max_calories:     Pre-filter recipes above this calorie count
        max_minutes:      Pre-filter recipes above this cook time
        top_n:            Number of final recommendations to return

    Returns:
        {
            detected_ingredients: list[str],
            recommendations:      list[dict],
            model_sources:        {yolo_count, detr_count, clip_count},
            error:                str | None
        }
    """
    # ── Stage 1: Vision ───────────────────────────────────────────────────────
    print("[1/3] Running vision ensemble...")
    yolo_hits = detect_yolo(image_path)
    detr_hits = detect_detr(image_path)
    clip_hits = detect_clip(image_path)

    ingredients = fuse_detections(yolo_hits, detr_hits, clip_hits)

    if not ingredients:
        return {
            "error":                "No ingredients detected. Try a clearer photo.",
            "detected_ingredients": [],
            "recommendations":      [],
            "model_sources":        {},
        }

    print(f"    Detected {len(ingredients)} ingredients: {', '.join(ingredients[:10])}")

    # ── Stage 2: RAG retrieval ────────────────────────────────────────────────
    print("[2/3] Retrieving recipe candidates from ChromaDB...")
    candidates = retrieve_recipes(
        ingredients,
        max_calories=max_calories,
        max_minutes=max_minutes,
    )

    if not candidates:
        return {
            "detected_ingredients": ingredients,
            "recommendations":      [],
            "model_sources":        {},
            "error":                "No matching recipes found. Try relaxing filters.",
        }

    print(f"    Retrieved {len(candidates)} candidates.")

    # ── Stage 3: LLM re-ranking ───────────────────────────────────────────────
    print("[3/3] Re-ranking with GPT-4o-mini...")
    ranked = llm_rerank(
        detected_ingredients=ingredients,
        candidates=candidates,
        top_n=top_n,
        user_preferences=user_preferences,
    )

    print(f"    Done. Returning top {len(ranked)} recipes.")

    return {
        "detected_ingredients": ingredients,
        "recommendations":      ranked,
        "model_sources": {
            "yolo_count": len(yolo_hits),
            "detr_count": len(detr_hits),
            "clip_count": len(clip_hits),
        },
        "error": None,
    }