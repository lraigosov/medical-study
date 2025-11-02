"""
Puerto (interfaz) para servicios de IA generativa (Gemini u otros proveedores).
Define el contrato que cualquier adaptador de IA debe cumplir.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class GenAIAnalyzerPort(Protocol):
    def analyze_medical_image(self, image_path: str, analysis_type: str = "general") -> Dict[str, Any]:
        ...

    def batch_analyze_images(self, image_paths: List[str], analysis_type: str = "general") -> List[Dict[str, Any]]:
        ...

    def compare_images(self, image_path_1: str, image_path_2: str, comparison_type: str = "temporal") -> Dict[str, Any]:
        ...

    def analyze_roi(self, image_path: str, roi_coordinates: Dict[str, int]) -> Dict[str, Any]:
        ...

    def generate_report(self, analysis_results: List[Dict[str, Any]], patient_info: Optional[Dict[str, Any]] = None) -> str:
        ...
