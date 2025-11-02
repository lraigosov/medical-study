import os
from pathlib import Path

from src.utils.config import load_config, AppConfig


def test_env_override_gemini_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "TEST_KEY")
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.gemini.api_key == "TEST_KEY"


def test_logging_defaults(tmp_path, monkeypatch):
    # Asegurar que carga sin archivo config.json
    monkeypatch.setenv("GEMINI_API_KEY", "")
    cfg = load_config(path=tmp_path / "missing.json")
    assert cfg.logging.level in ("INFO", "DEBUG", "WARNING", "ERROR")
