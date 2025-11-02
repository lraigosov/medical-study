"""
Gestor centralizado de configuración para el proyecto cancer/.

Características clave:
- Carga config/config.json si existe.
- Admite variables de entorno (prioridad sobre archivo), p. ej. GEMINI_API_KEY.
- Validación con Pydantic y valores por defecto seguros.
- Helper para obtener rutas absolutas y para configurar logging.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv


# Rutas base
BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "config.json"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"


class GeminiSettings(BaseModel):
    api_key: str = Field(default="", description="API Key para Google Gemini")
    model: str = Field(default="gemini-1.5-pro", description="Nombre del modelo Gemini")
    temperature: float = 0.1
    max_tokens: int = 4096


class TCIASettings(BaseModel):
    base_url: str = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
    collections_url: str = "https://www.cancerimagingarchive.net/wp-json/wp/v2/collections"
    download_path: str = str(DATA_DIR / "raw")
    supported_formats: list[str] = ["DICOM", "NIfTI", "ANALYZE"]


class DataSettings(BaseModel):
    target_collections: list[str] = []
    image_size: list[int] = [224, 224]
    batch_size: int = 32
    train_test_split: float = 0.8
    validation_split: float = 0.2


class EarlyDetectionSettings(BaseModel):
    architecture: str = "ResNet50"
    input_shape: list[int] = [224, 224, 3]
    num_classes: int = 2
    learning_rate: float = 1e-3
    epochs: int = 50
    patience: int = 10


class MultiClassSettings(BaseModel):
    architecture: str = "EfficientNetB0"
    input_shape: list[int] = [224, 224, 3]
    num_classes: int = 5
    learning_rate: float = 1e-4
    epochs: int = 100
    patience: int = 15


class ModelSettings(BaseModel):
    early_detection: EarlyDetectionSettings = EarlyDetectionSettings()
    multiclass_detection: MultiClassSettings = MultiClassSettings()


class RadiomicsSettings(BaseModel):
    feature_classes: list[str] = [
        "firstorder",
        "glcm",
        "glrlm",
        "glszm",
        "gldm",
        "ngtdm",
        "shape",
    ]
    bin_width: int = 25
    normalize: bool = True
    resample_pixel_spacing: list[int] = [1, 1, 1]


class LoggingSettings(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = str(LOGS_DIR / "cancer_analysis.log")


class AppConfig(BaseModel):
    gemini: GeminiSettings = GeminiSettings()
    tcia: TCIASettings = TCIASettings()
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    radiomics: RadiomicsSettings = RadiomicsSettings()
    logging: LoggingSettings = LoggingSettings()


_CONFIG_CACHE: Optional[AppConfig] = None


def _apply_env_overrides(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Aplica overrides desde variables de entorno.

    Prioriza GEMINI_API_KEY sobre gemini.api_key del archivo.
    """
    # Cargar .env si existe
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

    cfg = dict(raw_cfg)
    # Navegar de forma segura
    cfg.setdefault("gemini", {})

    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        cfg["gemini"]["api_key"] = env_key

    return cfg


def load_config(path: Optional[Path] = None, force_reload: bool = False) -> AppConfig:
    """Carga y valida la configuración del proyecto.

    - Si no existe el archivo, devuelve valores por defecto.
    - Variables de entorno tienen prioridad (p. ej. GEMINI_API_KEY).
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    path = Path(path) if path else CONFIG_PATH

    raw: Dict[str, Any]
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Error leyendo configuración: {exc}")
    else:
        raw = {}

    # Overrides desde entorno
    raw = _apply_env_overrides(raw)

    try:
        _CONFIG_CACHE = AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise RuntimeError(f"Configuración inválida: {exc}")

    # Asegurar carpetas base
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)

    return _CONFIG_CACHE


def configure_logging(cfg: Optional[AppConfig] = None) -> logging.Logger:
    """Configura logging del proyecto y retorna un logger base."""
    cfg = cfg or load_config()
    logger = logging.getLogger("cancer")

    # Evitar duplicar handlers
    if not logger.handlers:
        logger.setLevel(getattr(logging, cfg.logging.level.upper(), logging.INFO))

        fmt = logging.Formatter(cfg.logging.format)

        # Consola
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        # Archivo
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(cfg.logging.file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:  # noqa: BLE001
            # Si falla escribir archivo, continuamos con consola
            pass

    return logger


def project_path(*parts: str) -> Path:
    """Devuelve una ruta absoluta dentro del proyecto."""
    return BASE_DIR.joinpath(*parts)

