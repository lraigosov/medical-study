"""
Puerto (interfaz) para procesamiento DICOM (carga, serie, preprocesamiento).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class DicomPort(Protocol):
    def load_dicom_file(self, dicom_path: str) -> Optional[Any]:
        ...

    def load_dicom_series(self, directory: str) -> List[Any]:
        ...

    def dicom_to_array(self, dicom_data: Any) -> Optional[Any]:
        ...
