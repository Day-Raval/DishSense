import torch
import open_clip
from PIL import Image
from src.config import (
    CLIP_MODEL, CLIP_PRETRAINED,
    CLIP_CANDIDATES, CLIP_THRESHOLD, CLIP_TOP_K,
)

_clip_model = None
_preprocess = None
_tokenizer  = None


def _load():
    global _clip_model, _preprocess, _tokenizer
    if _clip_model is None:
        _clip_model, _, _preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        _tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        _clip_model.eval()


def detect_clip(
    image_path: str,
    candidates: list[str] = CLIP_CANDIDATES,
    top_k:      int       = CLIP_TOP_K,
    threshold:  float     = CLIP_THRESHOLD,
) -> list[dict]:
    """
    Zero-shot ingredient detection using CLIP image-text similarity.

    Strengths:
        - Handles any ingredient — no training data needed
        - Covers the long tail of items YOLO and DETR were never trained on
        - Works by comparing image embedding to text embeddings of each candidate

    How it works:
        1. Encode the image into a 512-dim embedding
        2. Encode every candidate as 'a photo of <ingredient>'
        3. Compute cosine similarity between image and all text embeddings
        4. Return candidates above the confidence threshold

    Args:
        image_path: Path to the fridge image
        candidates: List of ingredient names to match against
        top_k:      Max number of results to return
        threshold:  Minimum similarity score to include

    Returns:
        List of dicts with keys: label, confidence, source — sorted desc
    """
    _load()
    image = _preprocess(Image.open(image_path)).unsqueeze(0)
    texts = _tokenizer([f"a photo of {ing}" for ing in candidates])

    with torch.no_grad():
        img_features  = _clip_model.encode_image(image)
        text_features = _clip_model.encode_text(texts)
        img_features  /= img_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * img_features @ text_features.T).softmax(dim=-1)[0]

    results = [
        {
            "label":      ing.lower(),
            "confidence": round(float(prob), 4),
            "source":     "clip",
        }
        for ing, prob in zip(candidates, probs)
        if float(prob) >= threshold
    ]
    return sorted(results, key=lambda x: -x["confidence"])[:top_k]