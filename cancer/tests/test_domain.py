"""
Tests para verificar entidades del dominio.
"""
from __future__ import annotations


def test_roi_entity():
    """Verificar entidad ROI."""
    from src.domain.entities import ROI
    
    roi = ROI(x=10, y=20, width=30, height=40)
    
    assert roi.x == 10
    assert roi.y == 20
    assert roi.width == 30
    assert roi.height == 40
    
    roi_dict = roi.to_dict()
    assert roi_dict["x"] == 10
    assert roi_dict["width"] == 30


def test_patient_info_entity():
    """Verificar entidad PatientInfo."""
    from src.domain.entities import PatientInfo
    
    patient = PatientInfo(id="P001", age=45, sex="M", metadata={"study": "XYZ"})
    
    assert patient.id == "P001"
    assert patient.age == 45
    assert patient.sex == "M"
    assert patient.metadata["study"] == "XYZ"


def test_analysis_result_entity():
    """Verificar entidad AnalysisResult."""
    from src.domain.entities import AnalysisResult
    
    result = AnalysisResult(
        image_path="/tmp/test.jpg",
        analysis_type="general",
        response_text="Mock analysis",
        extra={"confidence": 0.95}
    )
    
    assert result.image_path == "/tmp/test.jpg"
    assert result.analysis_type == "general"
    assert result.response_text == "Mock analysis"
    assert result.extra["confidence"] == 0.95
