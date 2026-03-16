from ultralytics import YOLO # type: ignore
from src.config import YOLO_MODEL, YOLO_CONF

_model = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(YOLO_MODEL)
    return _model


def detect_yolo(image_path: str, conf_threshold: float = YOLO_CONF) -> list[dict]:
    """
    Run YOLOv8 on a fridge photo.

    Strengths:
        - Fastest model in the ensemble
        - Highly accurate on common grocery items
        - Returns bounding boxes and class labels directly

    Args:
        image_path:     Path to the fridge image
        conf_threshold: Minimum confidence to include a detection

    Returns:
        List of dicts with keys: label, confidence, source
    """
    model   = _get_model()
    results = model(image_path, conf=conf_threshold, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            conf  = float(box.conf)
            if conf >= conf_threshold:
                detections.append({
                    "label":      label.lower(),
                    "confidence": round(conf, 4),
                    "source":     "yolo",
                })
    return detections