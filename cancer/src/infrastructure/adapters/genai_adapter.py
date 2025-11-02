"""
Adaptador que implementa GenAIAnalyzerPort usando utils.gemini_analyzer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...ports.genai_port import GenAIAnalyzerPort
from ...utils.gemini_analyzer import GeminiAnalyzer


class GenAIGeminiAdapter(GenAIAnalyzerPort):
    def __init__(self, config_path: Optional[str] = None) -> None:
        self._impl = GeminiAnalyzer(config_path=config_path)

    def analyze_medical_image(self, image_path: str, analysis_type: str = "general") -> Dict[str, Any]:
        return self._impl.analyze_medical_image(image_path, analysis_type)

    def batch_analyze_images(self, image_paths: List[str], analysis_type: str = "general") -> List[Dict[str, Any]]:
        return self._impl.batch_analyze_images(image_paths, analysis_type)

    def compare_images(self, image_path_1: str, image_path_2: str, comparison_type: str = "temporal") -> Dict[str, Any]:
        return self._impl.compare_images(image_path_1, image_path_2, comparison_type)

    def analyze_roi(self, image_path: str, roi_coordinates: Dict[str, int]) -> Dict[str, Any]:
        return self._impl.analyze_roi(image_path, roi_coordinates)

    def generate_report(self, analysis_results: List[Dict[str, Any]], patient_info: Optional[Dict[str, Any]] = None) -> str:
        return self._impl.generate_report(analysis_results, patient_info)
