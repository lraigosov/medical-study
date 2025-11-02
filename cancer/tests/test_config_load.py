from pathlib import Path

from src.utils.config_loader import load_config


def test_load_config_default():
    cfg = load_config()
    # Validar llaves principales
    assert cfg is not None
    assert "tcia" in cfg
    assert "logging" in cfg


def test_load_config_explicit():
    cfg_path = Path("f:/GitHub/medical-study/cancer/config/config.json")
    cfg = load_config(cfg_path)
    assert cfg is not None
    assert "tcia" in cfg
    assert "logging" in cfg
