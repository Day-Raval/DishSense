import pytest # type: ignore
from src.vision.ensemble import fuse_detections


def test_fuse_deduplicates_across_models():
    yolo = [{"label": "apple",  "confidence": 0.90, "source": "yolo"}]
    detr = [{"label": "apple",  "confidence": 0.85, "source": "detr"}]
    clip = [{"label": "banana", "confidence": 0.60, "source": "clip"}]
    result = fuse_detections(yolo, detr, clip)
    assert result.count("apple") == 1
    assert "banana" in result


def test_fuse_multi_model_boost_keeps_item():
    yolo = [{"label": "carrot", "confidence": 0.50, "source": "yolo"}]
    detr = [{"label": "carrot", "confidence": 0.50, "source": "detr"}]
    result = fuse_detections(yolo, detr, [], min_confidence=0.35)
    assert "carrot" in result


def test_fuse_filters_low_confidence():
    clip = [{"label": "mystery_item", "confidence": 0.10, "source": "clip"}]
    result = fuse_detections([], [], clip, min_confidence=0.35)
    assert "mystery_item" not in result


def test_fuse_normalizes_spaces_to_underscores():
    clip = [{"label": "chicken breast", "confidence": 0.80, "source": "clip"}]
    result = fuse_detections([], [], clip)
    assert "chicken_breast" in result


def test_fuse_empty_inputs():
    result = fuse_detections([], [], [])
    assert result == []


def test_fuse_sorts_by_confidence_descending():
    clip = [
        {"label": "tomato", "confidence": 0.40, "source": "clip"},
        {"label": "egg",    "confidence": 0.90, "source": "clip"},
        {"label": "milk",   "confidence": 0.65, "source": "clip"},
    ]
    result = fuse_detections([], [], clip)
    assert result.index("egg")  < result.index("milk")
    assert result.index("milk") < result.index("tomato")