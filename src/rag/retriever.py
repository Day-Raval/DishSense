import json
import chromadb # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from src.config import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL, RETRIEVE_TOP_K

embedder    = SentenceTransformer(EMBED_MODEL)
_client     = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client     = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve_recipes(
    ingredients:  list[str],
    top_k:        int        = RETRIEVE_TOP_K,
    max_calories: float | None = None,
    max_minutes:  int   | None = None,
) -> list[dict]:
    """
    Embed the detected ingredient list as a semantic query and
    retrieve the top-K most similar recipes from ChromaDB.

    Why pre-filter instead of post-filter?
        Applying the WHERE clause inside ChromaDB before vector search
        is much cheaper than retrieving all results and filtering after.
        Always filter early when possible.

    Args:
        ingredients:  Detected ingredient list from the vision ensemble
        top_k:        Number of candidates to retrieve
        max_calories: Optional calorie ceiling for pre-filtering
        max_minutes:  Optional cook time ceiling for pre-filtering

    Returns:
        List of candidate recipe dicts with similarity_score appended
    """
    collection  = _get_collection()
    query       = f"Recipe using: {', '.join(ingredients)}"
    query_embed = embedder.encode(query).tolist()

    where = {}
    if max_calories is not None:
        where["calories"] = {"$lte": float(max_calories)}
    if max_minutes is not None:
        where["minutes"]  = {"$lte": int(max_minutes)}

    results = collection.query(
        query_embeddings=[query_embed],
        n_results=top_k,
        where=where if where else None,
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        candidates.append({
            **meta,
            "ingredients":      json.loads(meta.get("ingredients", "[]")),
            "similarity_score": round(1 - dist, 4),
            "document":         doc,
        })

    return candidates