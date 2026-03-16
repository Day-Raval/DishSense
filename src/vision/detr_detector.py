import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection # type: ignore
from src.config import DETR_MODEL, DETR_CONF

_processor = None
_model     = None


def _load():
    global _processor, _model
    if _model is None:
        _processor = DetrImageProcessor.from_pretrained(DETR_MODEL)
        _model     = DetrForObjectDetection.from_pretrained(DETR_MODEL)
        _model.eval()


def detect_detr(image_path: str, conf_threshold: float = DETR_CONF) -> list[dict]:
    """
    Run DETR (Detection Transformer) on a fridge photo.

    Strengths:
        - Transformer encoder-decoder architecture
        - No anchor boxes, no non-maximum suppression needed
        - Catches occluded and unusual-angle items YOLO misses
        - Architecturally interesting — shows evolution beyond CNN detectors

    Args:
        image_path:     Path to the fridge image
        conf_threshold: Minimum confidence to include a detection

    Returns:
        List of dicts with keys: label, confidence, source
    """
    _load()
    image  = Image.open(image_path).convert("RGB")
    inputs = _processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = _model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results      = _processor.post_process_object_detection(
        outputs,
        threshold=conf_threshold,
        target_sizes=target_sizes,
    )[0]

    detections = []
    for score, label_id in zip(results["scores"], results["labels"]):
        label = _model.config.id2label[label_id.item()].lower()
        detections.append({
            "label":      label,
            "confidence": round(float(score), 4),
            "source":     "detr",
        })
    return detections