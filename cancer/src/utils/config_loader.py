"""
Utilidad minimalista para carga de configuración desde config/config.json.
Todos los módulos deben usar este loader para acceder a la configuración centralizada.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config" / "config.json"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Carga configuración desde config/config.json.
    
    Args:
        path: Ruta opcional al archivo de configuración. Si no se especifica, usa CONFIG_PATH.
        
    Returns:
        Diccionario con la configuración completa del proyecto.
    """
    cfg_path = Path(path) if path else CONFIG_PATH
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Error leyendo configuración: {exc}") from exc
    return {}


def configure_logging(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Configura logging del proyecto y retorna un logger base.
    
    Args:
        config: Diccionario de configuración opcional. Si no se proporciona, se carga desde config.json.
        
    Returns:
        Logger configurado para el proyecto.
    """
    if config is None:
        config = load_config()
    
    logging_cfg = config.get("logging", {})
    logger = logging.getLogger("cancer")
    
    if logger.handlers:
        return logger
    
    level_name = str(logging_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    
    fmt = logging.Formatter(logging_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    
    # Handler de consola
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    
    # Handler de archivo
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = logging_cfg.get("file", str(LOGS_DIR / "cancer_analysis.log"))
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:  # noqa: BLE001
        pass
    
    return logger


def project_path(*parts: str) -> Path:
    """Devuelve una ruta absoluta dentro del proyecto.
    
    Args:
        *parts: Componentes de la ruta relativa al directorio base del proyecto.
        
    Returns:
        Path absoluto resultante.
    """
    return BASE_DIR.joinpath(*parts)


# Clase auxiliar para compatibilidad con código que espera acceso por atributos
class AppConfig:
    """Wrapper de configuración con acceso por atributos para compatibilidad."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, AppConfig(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración de vuelta a diccionario."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AppConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
