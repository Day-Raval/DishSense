import json
import pytest # type: ignore
from unittest.mock import patch


def test_retriever_builds_correct_query():
    with patch("src.rag.retriever.embedder") as mock_embedder, \
         patch("src.rag.retriever._get_collection") as mock_col:

        mock_embedder.encode.return_value = [0.1] * 384
        mock_col.return_value.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{
                "name":        "Egg Fried Rice",
                "calories":    350.0,
                "protein_g":   12.0,
                "fat_g":       8.0,
                "carbs_g":     45.0,
                "minutes":     15,
                "category":    "Rice",
                "ingredients": '["egg","rice","soy sauce"]',
            }]],
            "distances": [[0.15]],
        }

        from src.rag.retriever import retrieve_recipes
        results   = retrieve_recipes(["egg", "rice", "soy_sauce"])
        call_args = mock_embedder.encode.call_args[0][0]

        assert "egg"  in call_args
        assert "rice" in call_args
        assert results[0]["similarity_score"] == round(1 - 0.15, 4)
        assert results[0]["name"] == "Egg Fried Rice"


def test_retriever_applies_calorie_filter():
    with patch("src.rag.retriever.embedder") as mock_embedder, \
         patch("src.rag.retriever._get_collection") as mock_col:

        mock_embedder.encode.return_value = [0.1] * 384
        mock_col.return_value.query.return_value = {
            "documents": [[]], "metadatas": [[]], "distances": [[]]
        }

        from src.rag.retriever import retrieve_recipes
        retrieve_recipes(["egg"], max_calories=500.0)

        where_arg = mock_col.return_value.query.call_args[1]["where"]
        assert where_arg == {"calories": {"$lte": 500.0}}