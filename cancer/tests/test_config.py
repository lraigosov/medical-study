import json
from pathlib import Path

from src.utils.config_loader import load_config, AppConfig


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

    cfg_dict = load_config(path=cfg_path)
    cfg = AppConfig(cfg_dict)
    assert isinstance(cfg, AppConfig)
    assert cfg.gemini.api_key == "FILE_KEY"


def test_logging_defaults(tmp_path):
    # Asegurar que carga sin archivo config.json (usa defaults - dict vacío)
    cfg_dict = load_config(path=tmp_path / "missing.json")
    # Si no hay archivo, devuelve dict vacío; logging no estaría presente
    # o podrías tener defaults en tu loader
    assert isinstance(cfg_dict, dict)
