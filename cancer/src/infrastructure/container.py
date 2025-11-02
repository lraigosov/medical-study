"""
Contenedor simple de dependencias: construye adaptadores/servicios desde config.json.
"""
from __future__ import annotations

from typing import Optional

from ..utils.config_loader import load_config
from .adapters.genai_adapter import GenAIGeminiAdapter
from .adapters.tcia_adapter import TciaAdapter
from ..application.services.analysis_service import AnalysisService


class Container:
    def __init__(self, config_path: Optional[str] = None) -> None:
        self._config_path = config_path
        self._cfg = load_config()

        # Adaptadores (puertos)
        self.genai = GenAIGeminiAdapter(config_path)
        self.tcia = TciaAdapter(config_path)

        # Servicios de aplicación
        self.analysis_service = AnalysisService(self.genai)


def build_container(config_path: Optional[str] = None) -> Container:
    return Container(config_path)
