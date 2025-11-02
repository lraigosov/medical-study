"""
Entidades y value objects del dominio.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ROI:
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class PatientInfo:
    id: str
    age: Optional[int] = None
    sex: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisResult:
    image_path: str
    analysis_type: str
    response_text: Optional[str]
    extra: Optional[Dict[str, Any]] = None
