"""
Caso de uso: análisis de imagen médica orquestando puertos.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...ports.genai_port import GenAIAnalyzerPort


class AnalysisService:
    def __init__(self, genai: GenAIAnalyzerPort) -> None:
        self._genai = genai

    def analyze_image(self, image_path: str, analysis_type: str = "general") -> Dict[str, Any]:
        return self._genai.analyze_medical_image(image_path, analysis_type)

    def analyze_batch(self, image_paths: List[str], analysis_type: str = "general") -> List[Dict[str, Any]]:
        return self._genai.batch_analyze_images(image_paths, analysis_type)

    def compare(self, image1: str, image2: str, comparison_type: str = "temporal") -> Dict[str, Any]:
        return self._genai.compare_images(image1, image2, comparison_type)
