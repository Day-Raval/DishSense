"""
One-time ingestion: embed all recipes and store in ChromaDB.
Run via: python scripts/build_vectordb.py --limit 10000
"""

import json
import pandas as pd
import chromadb # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from src.config import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL

embedder = SentenceTransformer(EMBED_MODEL)


def _parse_ingredients(raw) -> list:
    """
    Safely parse RecipeIngredientParts.
    Food.com stores ingredients as R-style vectors: c("item1", "item2", ...)
    This handles that format plus plain JSON lists and NaN values.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    try:
        raw_str = str(raw).strip()
        if pd.isna(raw):
            return []
    except Exception:
        pass
    try:
        # Strip R-style c(...) wrapper
        if raw_str.startswith("c("):
            raw_str = raw_str[2:-1]
        parsed = json.loads(raw_str)
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except Exception:
        pass
    try:
        # Fallback: manually split R vector string
        cleaned = raw_str.strip('c(")')
        parts   = cleaned.split('","')
        return [p.strip().strip('"') for p in parts if p.strip()]
    except Exception:
        return []


def _parse_time(raw) -> int:
    """
    Convert ISO 8601 duration string to minutes.
    Food.com uses formats like: PT30M, PT1H30M, PT2H
    """
    if raw is None:
        return 30
    if isinstance(raw, (int, float)):
        return int(raw) if not pd.isna(raw) else 30
    try:
        raw_str = str(raw).upper().replace("PT", "")
        minutes = 0
        if "H" in raw_str:
            parts    = raw_str.split("H")
            minutes += int(parts[0]) * 60
            raw_str  = parts[1] if len(parts) > 1 else ""
        if "M" in raw_str:
            minutes += int(raw_str.replace("M", "").strip())
        return minutes if minutes > 0 else 30
    except Exception:
        return 30


def _build_recipe_document(row: pd.Series) -> str:
    """
    Build a rich natural language string for embedding.
    More context in the document = better semantic retrieval at query time.
    """
    ingredients     = _parse_ingredients(row.get("RecipeIngredientParts", ""))
    ingredients_str = ", ".join(str(i) for i in ingredients[:20])
    keywords        = str(row.get("Keywords", ""))[:200]

    return (
        f"Recipe: {row['Name']}. "
        f"Ingredients needed: {ingredients_str}. "
        f"Category: {row.get('RecipeCategory', 'general')}. "
        f"Keywords: {keywords}."
    )


def ingest_recipes(csv_path: str, batch_size: int = 256, limit: int = None):
    """
    Embed all recipes and persist into ChromaDB.

    Args:
        csv_path:   Path to Food.com recipes.csv
        batch_size: Rows processed per embedding batch
        limit:      Cap row count — e.g. 10000 for fast dev builds
    """
    print(f"\nLoading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows loaded:  {len(df):,}")

    df = df.dropna(subset=["Name", "RecipeIngredientParts"])
    print(f"Rows after dropna:  {len(df):,}")

    if limit:
        df = df.head(limit)
        print(f"Rows after limit:   {len(df):,}")

    print(f"\nConnecting to ChromaDB at: {CHROMA_PATH}")
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Collection '{COLLECTION_NAME}' ready.")
    print(f"\nStarting ingestion in batches of {batch_size}...\n")

    for i in range(0, len(df), batch_size):
        batch  = df.iloc[i: i + batch_size]
        docs   = [_build_recipe_document(row) for _, row in batch.iterrows()]
        embeds = embedder.encode(docs, show_progress_bar=False).tolist()
        ids    = [
            str(row.get("RecipeId", f"{i}_{j}"))
            for j, (_, row) in enumerate(batch.iterrows())
        ]

        metadatas = []
        for _, row in batch.iterrows():
            ings = _parse_ingredients(row.get("RecipeIngredientParts", ""))
            metadatas.append({
                "name":        str(row["Name"]),
                "calories":    float(row.get("Calories",            0) or 0),
                "protein_g":   float(row.get("ProteinContent",      0) or 0),
                "fat_g":       float(row.get("FatContent",          0) or 0),
                "carbs_g":     float(row.get("CarbohydrateContent", 0) or 0),
                "minutes":     _parse_time(row.get("TotalTime")),
                "category":    str(row.get("RecipeCategory", "")),
                "ingredients": json.dumps(ings[:20]),
            })

        collection.add(
            documents=docs,
            embeddings=embeds,
            ids=ids,
            metadatas=metadatas,
        )
        print(f"  Ingested {min(i + batch_size, len(df)):,} / {len(df):,} recipes")

    final_count = collection.count()
    print(f"\nDone. {final_count:,} recipes stored in ChromaDB at '{CHROMA_PATH}'")