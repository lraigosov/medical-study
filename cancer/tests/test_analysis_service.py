"""
Tests unitarios mínimos para AnalysisService.
"""
from __future__ import annotations

import pytest
from typing import Any, Dict, List, Optional


class MockGenAIPort:
    """Mock del puerto GenAI para tests."""
    
    def analyze_medical_image(self, image_path: str, analysis_type: str = "general") -> Dict[str, Any]:
        return {
            "image_path": image_path,
            "analysis_type": analysis_type,
            "response_text": "Mock analysis result",
            "disclaimer_added": True,
        }
    
    def batch_analyze_images(self, image_paths: List[str], analysis_type: str = "general") -> List[Dict[str, Any]]:
        return [self.analyze_medical_image(p, analysis_type) for p in image_paths]
    
    def compare_images(self, image_path_1: str, image_path_2: str, comparison_type: str = "temporal") -> Dict[str, Any]:
        return {
            "image_1": image_path_1,
            "image_2": image_path_2,
            "comparison_type": comparison_type,
            "response_text": "Mock comparison result",
            "disclaimer_added": True,
        }
    
    def analyze_roi(self, image_path: str, roi_coordinates: Dict[str, int]) -> Dict[str, Any]:
        return {
            "image_path": image_path,
            "roi_coordinates": roi_coordinates,
            "response_text": "Mock ROI analysis",
            "disclaimer_added": True,
        }
    
    def generate_report(self, analysis_results: List[Dict[str, Any]], patient_info: Optional[Dict[str, Any]] = None) -> str:
        return "Mock report"


def test_analysis_service_analyze_image():
    """Verificar que AnalysisService invoca el puerto correctamente."""
    from src.application.services.analysis_service import AnalysisService
    
    mock_port = MockGenAIPort()
    svc = AnalysisService(genai=mock_port)
    
    result = svc.analyze_image("test.jpg", "general")
    
    assert result["image_path"] == "test.jpg"
    assert result["analysis_type"] == "general"
    assert "response_text" in result
    assert result.get("disclaimer_added") is True


def test_analysis_service_batch_analyze():
    """Verificar análisis en lote."""
    from src.application.services.analysis_service import AnalysisService
    
    mock_port = MockGenAIPort()
    svc = AnalysisService(genai=mock_port)
    
    results = svc.analyze_batch(["a.jpg", "b.jpg"], "cancer_detection")
    
    assert len(results) == 2
    assert results[0]["image_path"] == "a.jpg"
    assert results[1]["image_path"] == "b.jpg"


def test_analysis_service_compare():
    """Verificar comparación de dos imágenes."""
    from src.application.services.analysis_service import AnalysisService
    
    mock_port = MockGenAIPort()
    svc = AnalysisService(genai=mock_port)
    
    result = svc.compare("before.jpg", "after.jpg", "temporal")
    
    assert result["image_1"] == "before.jpg"
    assert result["image_2"] == "after.jpg"
    assert result["comparison_type"] == "temporal"
    assert result.get("disclaimer_added") is True
