import json
from pathlib import Path

from src.utils.config import load_config, AppConfig


def test_config_from_file(tmp_path: Path):
    # Crear config.json temporal con clave de Gemini
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "gemini": {
            "api_key": "FILE_KEY",
            "model": "gemini-1.5-pro",
            "temperature": 0.1,
            "max_tokens": 4096
        }
    }), encoding="utf-8")

    cfg = load_config(path=cfg_path)
    assert isinstance(cfg, AppConfig)
    assert cfg.gemini.api_key == "FILE_KEY"


def test_logging_defaults(tmp_path, monkeypatch):
    # Asegurar que carga sin archivo config.json (usa defaults)
    cfg = load_config(path=tmp_path / "missing.json")
    assert cfg.logging.level in ("INFO", "DEBUG", "WARNING", "ERROR")
