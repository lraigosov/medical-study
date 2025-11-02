"""
Puerto (interfaz) para acceso a TCIA u orígenes equivalentes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class TciaPort(Protocol):
    def get_collections(self) -> List[Dict[str, Any]]:
        ...

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        ...

    def get_patients(self, collection: str) -> List[Dict[str, Any]]:
        ...

    def get_studies(self, collection: str, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        ...

    def get_series(self, collection: str, patient_id: Optional[str] = None, study_uid: Optional[str] = None) -> List[Dict[str, Any]]:
        ...

    def download_series(self, series_uid: str, download_path: Optional[str] = None) -> str:
        ...
