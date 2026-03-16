from collections import defaultdict
from src.config import ENSEMBLE_MIN_CONF, MULTI_MODEL_BOOST


def fuse_detections(
    yolo_results: list[dict],
    detr_results: list[dict],
    clip_results: list[dict],
    min_confidence: float = ENSEMBLE_MIN_CONF,
) -> list[str]:
    """
    Merge detections from all three vision models into a clean ingredient list.

    Strategy:
        1. Normalize all labels — lowercase, strip whitespace, spaces to underscores
        2. Pool confidence scores per label across all sources
        3. Average confidence and apply 15% boost for multi-model agreement
        4. Filter out anything below min_confidence threshold
        5. Return sorted highest confidence first

    Why multi-model boost matters:
        An item seen by both YOLO and CLIP is far more likely to be real
        than one seen by only CLIP at borderline confidence. The boost
        rewards consensus between architectures with different inductive biases.

    Args:
        yolo_results:   Detections from YOLOv8
        detr_results:   Detections from DETR
        clip_results:   Detections from CLIP
        min_confidence: Minimum score to include in final list

    Returns:
        Sorted list of ingredient name strings
    """
    scores: dict[str, list[float]] = defaultdict(list)

    for item in yolo_results + detr_results + clip_results:
        label = item["label"].lower().strip().replace(" ", "_")
        scores[label].append(item["confidence"])

    fused: dict[str, float] = {}
    for label, confs in scores.items():
        avg   = sum(confs) / len(confs)
        boost = MULTI_MODEL_BOOST if len(confs) > 1 else 1.0
        fused[label] = round(avg * boost, 4)

    return [
        label
        for label, score in sorted(fused.items(), key=lambda x: -x[1])
        if score >= min_confidence
    ]