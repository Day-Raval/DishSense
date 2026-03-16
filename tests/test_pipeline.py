import pytest # type: ignore
from unittest.mock import patch

FAKE_YOLO = [{"label": "egg",    "confidence": 0.92, "source": "yolo"}]
FAKE_DETR = [{"label": "milk",   "confidence": 0.80, "source": "detr"}]
FAKE_CLIP = [{"label": "butter", "confidence": 0.65, "source": "clip"}]

FAKE_CANDIDATES = [{
    "name":             "Scrambled Eggs",
    "calories":         220.0,
    "protein_g":        14.0,
    "fat_g":            16.0,
    "carbs_g":          2.0,
    "minutes":          10,
    "category":         "Breakfast",
    "similarity_score": 0.91,
    "ingredients":      ["egg", "butter", "milk"],
    "document":         "",
}]

FAKE_RANKED = [{
    "rank":                 1,
    "name":                 "Scrambled Eggs",
    "coverage_pct":         100,
    "missing_ingredients":  [],
    "nutrition_score":      7,
    "reason":               "Uses all your detected ingredients.",
    **FAKE_CANDIDATES[0],
}]


@patch("src.pipeline.detect_yolo",      return_value=FAKE_YOLO)
@patch("src.pipeline.detect_detr",      return_value=FAKE_DETR)
@patch("src.pipeline.detect_clip",      return_value=FAKE_CLIP)
@patch("src.pipeline.retrieve_recipes", return_value=FAKE_CANDIDATES)
@patch("src.pipeline.llm_rerank",       return_value=FAKE_RANKED)
def test_pipeline_happy_path(mock_rank, mock_ret, mock_clip, mock_detr, mock_yolo):
    from src.pipeline import recommend_from_photo
    result = recommend_from_photo("fake.jpg")

    assert result["error"] is None
    assert len(result["detected_ingredients"]) == 3
    assert result["recommendations"][0]["name"] == "Scrambled Eggs"
    assert result["model_sources"]["yolo_count"] == 1
    assert result["model_sources"]["detr_count"] == 1
    assert result["model_sources"]["clip_count"] == 1


@patch("src.pipeline.detect_yolo", return_value=[])
@patch("src.pipeline.detect_detr", return_value=[])
@patch("src.pipeline.detect_clip", return_value=[])
def test_pipeline_no_detections_returns_error(mock_clip, mock_detr, mock_yolo):
    from src.pipeline import recommend_from_photo
    result = recommend_from_photo("empty_fridge.jpg")

    assert "error" in result
    assert result["detected_ingredients"] == []
    assert result["recommendations"]      == []


@patch("src.pipeline.detect_yolo",      return_value=FAKE_YOLO)
@patch("src.pipeline.detect_detr",      return_value=FAKE_DETR)
@patch("src.pipeline.detect_clip",      return_value=FAKE_CLIP)
@patch("src.pipeline.retrieve_recipes", return_value=[])
def test_pipeline_no_candidates_returns_error(mock_ret, mock_clip, mock_detr, mock_yolo):
    from src.pipeline import recommend_from_photo
    result = recommend_from_photo("fake.jpg")

    assert "error" in result
    assert len(result["detected_ingredients"]) > 0
    assert result["recommendations"] == []