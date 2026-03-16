"""
One-time script: embed all recipes and store in ChromaDB.
Run AFTER downloading the dataset with the kaggle CLI command.

Usage:
    python scripts/build_vectordb.py                  # full 230k recipes (~5 min)
    python scripts/build_vectordb.py --limit 10000    # fast dev build (~30 sec)
    python scripts/build_vectordb.py --limit 10000 --batch-size 128
    python scripts/build_vectordb.py --csv path/to/recipes.csv
"""

import argparse
import os
import sys

# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.abspath("."))

from src.rag.ingest import ingest_recipes


def main():
    parser = argparse.ArgumentParser(
        description="Build FridgeRAG recipe vector database"
    )
    parser.add_argument(
        "--csv",
        default="data/raw/recipes.csv",
        help="Path to Food.com recipes.csv (default: data/raw/recipes.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows — e.g. 10000 for a fast dev build",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Embedding batch size (default: 256)",
    )
    args = parser.parse_args()

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not os.path.exists(args.csv):
        print(f"\nERROR: Dataset file not found at '{args.csv}'")
        print("\nDownload it first by running:")
        print(
            "  kaggle datasets download -d irkaal/foodcom-recipes-and-reviews "
            "--unzip -p data/raw/"
        )
        print("\nOr manually from:")
        print(
            "  https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews/data"
        )
        sys.exit(1)

    file_size_mb = os.path.getsize(args.csv) / (1024 * 1024)
    print(f"\nFound: {args.csv} ({file_size_mb:.1f} MB)")

    if args.limit:
        print(f"Mode:  Dev build — limiting to {args.limit:,} recipes")
    else:
        print("Mode:  Full build — all recipes (~5 min)")

    print(f"Batch: {args.batch_size} recipes per embedding batch")
    print("-" * 50)

    # ── Run ingestion ─────────────────────────────────────────────────────────
    ingest_recipes(
        csv_path=args.csv,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    print("-" * 50)
    print("\nNext steps:")
    print("  1. Start the API:       uvicorn api.main:app --reload --port 8000")
    print("  2. Start the dashboard: streamlit run dashboard/app.py")
    print("  3. Check health:        curl http://localhost:8000/health")


if __name__ == "__main__":
    main()