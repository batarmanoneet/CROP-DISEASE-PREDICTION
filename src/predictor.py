from __future__ import annotations

from typing import List, Dict

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def pretty_label(label: str):
    parts = label.replace("___", " — ").replace("_", " ").replace("  ", " ")
    return parts.strip()


class PlantDiseasePredictor:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id).eval()

    def predict(self, image: Image.Image, top_k: int = 3) -> List[Dict]:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            values, indices = torch.topk(probs, k=min(top_k, probs.numel()))

        output = []
        for value, index in zip(values.tolist(), indices.tolist()):
            label = self.model.config.id2label[int(index)]
            crop, disease = (label.split("___", 1) + ["unknown"])[:2] if "___" in label else ("Unknown", label)
            output.append(
                {
                    "label": label,
                    "pretty_disease": pretty_label(label),
                    "crop": crop.replace("_", " "),
                    "disease": disease.replace("_", " "),
                    "confidence": float(value),
                }
            )
        return output
