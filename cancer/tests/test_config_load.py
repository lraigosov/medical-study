from pathlib import Path

from src.utils.config import load_config


def test_load_config_default():
    cfg = load_config()
    # Validar llaves principales
    assert cfg is not None
    assert hasattr(cfg, "tcia")
    assert hasattr(cfg, "logging")


def test_load_config_explicit():
    cfg_path = Path("f:/GitHub/medical-study/cancer/config/config.json")
    cfg = load_config(cfg_path)
    assert cfg is not None
    assert cfg.tcia is not None
    assert cfg.logging is not None
