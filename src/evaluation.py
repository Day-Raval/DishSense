"""
Retrieval and ranking evaluation harness.

Metrics computed:
    - Precision@K  (P@K):   fraction of top-K results that are relevant
    - Recall@K:             fraction of all relevant items in top-K
    - nDCG@K:               normalized Discounted Cumulative Gain

Usage:
    python src/evaluation.py
    python src/evaluation.py --k 5 --k 10 --k 20
"""

import math
import json
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("."))


# ── Metric functions ──────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Precision@K = |relevant ∩ top-K retrieved| / K

    Answers: Of the top-K recipes returned, what fraction are actually good?
    Range: 0.0 (all bad) to 1.0 (all good)

    Example:
        retrieved = ["pasta", "pizza", "salad", "soup", "stew"]
        relevant  = {"pasta", "salad", "soup"}
        P@3 = 2/3 = 0.667  (pasta + salad in top 3, pizza is not relevant)
    """
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits  = sum(1 for r in top_k if r in relevant)
    return round(hits / k, 4)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Recall@K = |relevant ∩ top-K retrieved| / |relevant|

    Answers: Of all good recipes, how many did we find in the top-K?
    Range: 0.0 (found none) to 1.0 (found all)

    Example:
        retrieved = ["pasta", "pizza", "salad", "soup", "stew"]
        relevant  = {"pasta", "salad", "soup"}
        R@3 = 2/3 = 0.667  (found pasta + salad but missed soup in top 3)
        R@4 = 3/3 = 1.0    (found pasta + salad + soup in top 4)
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits  = sum(1 for r in top_k if r in relevant)
    return round(hits / len(relevant), 4)


def dcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Discounted Cumulative Gain@K

    Rewards relevant results appearing earlier in the ranking.
    Position 1 gets full credit, position 2 gets credit/log2(2),
    position 3 gets credit/log2(3), etc.

    Formula: DCG@K = sum(rel_i / log2(i+1)) for i in 1..K
    """
    dcg = 0.0
    for i, recipe in enumerate(retrieved[:k], start=1):
        if recipe in relevant:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Normalized DCG@K = DCG@K / IDCG@K

    IDCG (Ideal DCG) = DCG of a perfect ranking where all relevant
    items appear first.

    Range: 0.0 (worst possible ranking) to 1.0 (perfect ranking)

    This is the most informative single metric for ranking quality
    because it accounts for both relevance AND position.

    Example:
        retrieved = ["pasta", "pizza", "salad"]
        relevant  = {"pasta", "salad"}
        DCG@3  = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
        IDCG@3 = 1/log2(2) + 1/log2(3)     = 1.0 + 0.63 = 1.63
        nDCG@3 = 1.5 / 1.63 = 0.92
    """
    if not relevant:
        return 0.0
    # Ideal: all relevant items appear first
    ideal     = list(relevant)[:k]
    idcg      = dcg_at_k(ideal, relevant, k)
    actual    = dcg_at_k(retrieved, relevant, k)
    if idcg == 0:
        return 0.0
    return round(actual / idcg, 4)


# ── Test queries ──────────────────────────────────────────────────────────────

# Ground truth: for each query, which recipe names are considered relevant?
# In a real evaluation you would have human-annotated relevance judgments.
# Here we define them manually based on ingredient overlap.
EVAL_QUERIES = [
    {
        "query_id":   "Q001",
        "ingredients": ["egg", "milk", "butter", "cheese"],
        "relevant_recipes": {
            "Scrambled Eggs",
            "Cheese Omelette",
            "French Toast",
            "Egg Fried Rice",
            "Breakfast Casserole",
            "Quiche Lorraine",
        },
        "description": "Dairy + egg heavy fridge",
    },
    {
        "query_id":   "Q002",
        "ingredients": ["chicken_breast", "garlic", "onion", "tomato", "olive_oil"],
        "relevant_recipes": {
            "Chicken Stir Fry",
            "Garlic Chicken",
            "Chicken Tomato Pasta",
            "Roasted Chicken",
            "Chicken Cacciatore",
        },
        "description": "Chicken-based dinner ingredients",
    },
    {
        "query_id":   "Q003",
        "ingredients": ["pasta", "tomato", "garlic", "olive_oil", "parmesan_cheese"],
        "relevant_recipes": {
            "Spaghetti Marinara",
            "Pasta Pomodoro",
            "Aglio e Olio",
            "Tomato Pasta",
            "Pasta Arrabiata",
        },
        "description": "Classic pasta ingredients",
    },
    {
        "query_id":   "Q004",
        "ingredients": ["salmon", "lemon", "garlic", "butter", "asparagus"],
        "relevant_recipes": {
            "Lemon Butter Salmon",
            "Baked Salmon",
            "Salmon with Asparagus",
            "Pan Seared Salmon",
        },
        "description": "Seafood dinner ingredients",
    },
    {
        "query_id":   "Q005",
        "ingredients": ["black_beans", "corn", "bell_pepper", "onion", "tortilla"],
        "relevant_recipes": {
            "Black Bean Tacos",
            "Veggie Burrito",
            "Bean Quesadilla",
            "Mexican Rice Bowl",
            "Stuffed Peppers",
        },
        "description": "Vegetarian Mexican ingredients",
    },
]


def run_evaluation(k_values: list[int] = None) -> dict:
    """
    Run the full evaluation harness over all test queries.

    For each query:
        1. Retrieve top-max(K) recipes from ChromaDB
        2. Compute P@K, Recall@K, nDCG@K for each K value
        3. Aggregate mean metrics across all queries

    Args:
        k_values: List of K cutoffs to evaluate (default: [3, 5, 10, 20])

    Returns:
        Dict with per-query results and aggregate means
    """
    from src.rag.retriever import retrieve_recipes

    if k_values is None:
        k_values = [3, 5, 10, 20]

    max_k   = max(k_values)
    results = []

    print(f"\nRunning evaluation over {len(EVAL_QUERIES)} queries...")
    print(f"K values: {k_values}")
    print("-" * 70)

    for query in EVAL_QUERIES:
        qid        = query["query_id"]
        ingredients = query["ingredients"]
        relevant    = query["relevant_recipes"]
        desc        = query["description"]

        print(f"\n{qid}: {desc}")
        print(f"  Ingredients: {', '.join(ingredients)}")

        # Retrieve from ChromaDB
        candidates = retrieve_recipes(ingredients, top_k=max_k)
        retrieved  = [c["name"] for c in candidates]

        print(f"  Retrieved {len(retrieved)} candidates")

        # Compute metrics for each K
        query_metrics = {
            "query_id":    qid,
            "description": desc,
            "n_relevant":  len(relevant),
            "n_retrieved": len(retrieved),
            "metrics":     {}
        }

        for k in k_values:
            pk     = precision_at_k(retrieved, relevant, k)
            rk     = recall_at_k(retrieved, relevant, k)
            ndcg   = ndcg_at_k(retrieved, relevant, k)
            query_metrics["metrics"][f"P@{k}"]    = pk
            query_metrics["metrics"][f"R@{k}"]    = rk
            query_metrics["metrics"][f"nDCG@{k}"] = ndcg
            print(f"  @{k:2d}: P={pk:.3f}  R={rk:.3f}  nDCG={ndcg:.3f}")

        results.append(query_metrics)

    # Aggregate means across all queries
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (mean across all queries)")
    print("=" * 70)

    aggregate = {}
    for k in k_values:
        for metric_prefix in ("P", "R", "nDCG"):
            key  = f"{metric_prefix}@{k}"
            vals = [r["metrics"][key] for r in results]
            mean = round(sum(vals) / len(vals), 4)
            aggregate[f"mean_{key}"] = mean

    # Print formatted table
    header = f"{'Metric':<12}" + "".join(f"  K={k:<4}" for k in k_values)
    print(header)
    print("-" * len(header))
    for prefix, label in [("P", "Precision"), ("R", "Recall"), ("nDCG", "nDCG")]:
        row = f"{label:<12}"
        for k in k_values:
            val  = aggregate[f"mean_{prefix}@{k}"]
            row += f"  {val:.3f} "
        print(row)

    print("\nInterpretation:")
    mean_ndcg5 = aggregate.get("mean_nDCG@5", 0)
    if mean_ndcg5 >= 0.7:
        print(f"  nDCG@5={mean_ndcg5:.3f} → GOOD ranking quality")
    elif mean_ndcg5 >= 0.4:
        print(f"  nDCG@5={mean_ndcg5:.3f} → MODERATE ranking quality")
    else:
        print(f"  nDCG@5={mean_ndcg5:.3f} → needs improvement")

    return {
        "per_query":  results,
        "aggregate":  aggregate,
        "k_values":   k_values,
        "n_queries":  len(EVAL_QUERIES),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FridgeRAG evaluation harness")
    parser.add_argument("--k", type=int, action="append", dest="k_values",
                        help="K cutoff value (can specify multiple times)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    k_values = sorted(set(args.k_values)) if args.k_values else [3, 5, 10, 20]
    results  = run_evaluation(k_values=k_values)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")