from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def pretty_label(label: str) -> str:
    return label.replace("___", " — ").replace("_", " ").replace("  ", " ").strip()


def _normalise_name(value: str) -> str:
    """Normalise crop labels to match variants such as 'Pepper, bell' to 'Pepper'."""
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _split_label(label: str) -> tuple[str, str]:
    if "___" in label:
        crop, disease = label.split("___", 1)
        return crop, disease
    return "Unknown", label


def _crop_matches(label: str, crop_hint: str) -> bool:
    crop, _ = _split_label(label)
    actual = _normalise_name(crop)
    wanted = _normalise_name(crop_hint)
    # Prefix matching supports dataset labels such as "Pepper,_bell" and "Corn_(maize)".
    return bool(wanted and (actual == wanted or actual.startswith(wanted) or wanted.startswith(actual)))


class PlantDiseasePredictor:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id).eval()

    def predict(
        self,
        image: Image.Image,
        top_k: int = 3,
        crop_hint: Optional[str] = None,
    ) -> Dict:
        """Return global predictions and (optionally) crop-filtered candidates.

        Every score is the original softmax output over all model classes. We deliberately
        do not renormalise scores after filtering by crop, as that could make weak matches
        look falsely confident.
        """
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        labels = self.model.config.id2label
        k = min(max(int(top_k), 1), int(probs.numel()))
        values, indices = torch.topk(probs, k=k)

        def make_result(index: int, score: float) -> Dict:
            label = labels[int(index)]
            crop, disease = _split_label(label)
            return {
                "label": label,
                "pretty_disease": pretty_label(label),
                "crop": re.sub(r"\s+", " ", crop.replace("_", " ").replace(",", ", ")).strip(),
                "disease": disease.replace("_", " ").strip(),
                "confidence": float(score),
            }

        global_results = [make_result(int(i), float(v)) for v, i in zip(values.tolist(), indices.tolist())]
        global_top = global_results[0]

        applied_hint = crop_hint if crop_hint and crop_hint != "Auto-detect" else None
        crop_results: List[Dict] = []
        if applied_hint:
            matching_indices = [i for i, label in labels.items() if _crop_matches(str(label), applied_hint)]
            if matching_indices:
                ranked = sorted(matching_indices, key=lambda i: float(probs[int(i)]), reverse=True)[:k]
                crop_results = [make_result(i, float(probs[int(i)])) for i in ranked]

        return {
            "global_results": global_results,
            "global_top": global_top,
            "results": crop_results if applied_hint and crop_results else global_results,
            "crop_hint": applied_hint,
            "crop_hint_applied": bool(applied_hint and crop_results),
            "crop_hint_has_classes": bool(crop_results),
        }
