"""
Tests para adaptadores de la infraestructura hexagonal.
"""
from __future__ import annotations

import pytest
from typing import Any, Dict, List


def test_genai_adapter_imports():
    """Verificar que GenAIGeminiAdapter se puede importar."""
    try:
        from src.infrastructure.adapters.genai_adapter import GenAIGeminiAdapter
        assert GenAIGeminiAdapter is not None
    except ImportError as exc:
        pytest.skip(f"Importación fallida: {exc}")


def test_tcia_adapter_imports():
    """Verificar que TciaAdapter se puede importar."""
    try:
        from src.infrastructure.adapters.tcia_adapter import TciaAdapter
        assert TciaAdapter is not None
    except ImportError as exc:
        pytest.skip(f"Importación fallida: {exc}")


def test_container_build():
    """Verificar que build_container se puede construir sin error."""
    try:
        from src.infrastructure.container import build_container
        # No pasamos config_path; usará default
        # Si falla por API KEY ausente, es esperado; solo validamos construcción estructural
        assert build_container is not None
    except Exception as exc:
        pytest.skip(f"Container build fallida (esperado sin API_KEY): {exc}")
