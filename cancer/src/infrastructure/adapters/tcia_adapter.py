"""
Adaptador que implementa TciaPort usando utils.tcia_client.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...ports.tcia_port import TciaPort
from ...utils.tcia_client import TCIAClient


class TciaAdapter(TciaPort):
    def __init__(self, config_path: Optional[str] = None) -> None:
        self._impl = TCIAClient(config_path=config_path)

    def get_collections(self) -> List[Dict[str, Any]]:
        return self._impl.get_collections()

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        return self._impl.get_collection_info(collection_name)

    def get_patients(self, collection: str) -> List[Dict[str, Any]]:
        return self._impl.get_patients(collection)

    def get_studies(self, collection: str, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._impl.get_studies(collection, patient_id)

    def get_series(self, collection: str, patient_id: Optional[str] = None, study_uid: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._impl.get_series(collection, patient_id, study_uid)

    def download_series(self, series_uid: str, download_path: Optional[str] = None) -> str:
        return self._impl.download_series(series_uid, download_path)
