from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class KnowledgeBase:
    def __init__(self, path: str):
        self.path = Path(path)
        self.records = json.loads(self.path.read_text(encoding="utf-8"))

    def lookup(self, crop: str, disease: str) -> Optional[dict]:
        target = f"{crop} {disease}".lower().replace("_", " ")
        exact = []
        for r in self.records:
            hay = f"{r.get('crop','')} {r.get('disease','')}".lower().replace("_", " ")
            if crop.lower() in hay and disease.lower() in hay:
                exact.append(r)
        return exact[0] if exact else None
