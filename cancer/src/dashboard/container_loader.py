"""
Wrapper para cargar el contenedor de dependencias desde el dashboard.
Reimplementa todas las clases necesarias para evitar imports relativos.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configurar paths
CURRENT_FILE = Path(__file__).resolve()
DASHBOARD_DIR = CURRENT_FILE.parent  # dashboard/
SRC_DIR = DASHBOARD_DIR.parent  # src/
BASE_DIR = SRC_DIR.parent  # cancer/

# Asegurar que src/ esté en sys.path en la primera posición
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Importar solo los módulos que NO usan imports relativos
try:
    from utils.config_loader import load_config
    from utils.gemini_analyzer import GeminiAnalyzer
    
    # Reimplementar el port (protocolo/interfaz)
    from typing import Protocol
    
    class GenAIAnalyzerPort(Protocol):
        """Puerto/interfaz para análisis con IA generativa"""
        def analyze_medical_image(self, image_path: str, analysis_type: str = "general") -> Dict[str, Any]:
            ...
        
        def batch_analyze_images(self, image_paths: List[str], analysis_type: str = "general") -> List[Dict[str, Any]]:
            ...
        
        def compare_images(self, image_path_1: str, image_path_2: str, comparison_type: str = "temporal") -> Dict[str, Any]:
            ...
        
        def analyze_roi(self, image_path: str, roi_coordinates: Dict[str, int]) -> Dict[str, Any]:
            ...
        
        def generate_report(self, analysis_results: List[Dict[str, Any]], patient_info: Optional[Dict[str, Any]] = None) -> str:
            ...
    
    # Reimplementar el adaptador
    class GenAIGeminiAdapter(GenAIAnalyzerPort):
        """Adaptador para Gemini AI"""
        def __init__(self, config_path: Optional[str] = None) -> None:
            self._impl = GeminiAnalyzer(config_path=config_path)

        def analyze_medical_image(self, image_path: str, analysis_type: str = "general") -> Dict[str, Any]:
            return self._impl.analyze_medical_image(image_path, analysis_type)

        def batch_analyze_images(self, image_paths: List[str], analysis_type: str = "general") -> List[Dict[str, Any]]:
            return self._impl.batch_analyze_images(image_paths, analysis_type)

        def compare_images(self, image_path_1: str, image_path_2: str, comparison_type: str = "temporal") -> Dict[str, Any]:
            return self._impl.compare_images(image_path_1, image_path_2, comparison_type)

        def analyze_roi(self, image_path: str, roi_coordinates: Dict[str, int]) -> Dict[str, Any]:
            return self._impl.analyze_roi(image_path, roi_coordinates)

        def generate_report(self, analysis_results: List[Dict[str, Any]], patient_info: Optional[Dict[str, Any]] = None) -> str:
            return self._impl.generate_report(analysis_results, patient_info)
    
    # Reimplementar el servicio de aplicación
    class AnalysisService:
        """Servicio de análisis de imágenes médicas"""
        def __init__(self, genai: GenAIAnalyzerPort) -> None:
            self._genai = genai

        def analyze_image(self, image_path: str, analysis_type: str = "general") -> Dict[str, Any]:
            return self._genai.analyze_medical_image(image_path, analysis_type)

        def analyze_batch(self, image_paths: List[str], analysis_type: str = "general") -> List[Dict[str, Any]]:
            return self._genai.batch_analyze_images(image_paths, analysis_type)

        def compare(self, image1: str, image2: str, comparison_type: str = "temporal") -> Dict[str, Any]:
            return self._genai.compare_images(image1, image2, comparison_type)
    
    # Crear el container
    class Container:
        """Contenedor de dependencias (sin imports relativos)"""
        def __init__(self, config_path: Optional[str] = None) -> None:
            self._config_path = config_path
            self._cfg = load_config()
            
            # Adaptadores
            self.genai = GenAIGeminiAdapter(config_path)
            
            # Servicios
            self.analysis_service = AnalysisService(self.genai)
    
    def build_container(config_path: Optional[str] = None) -> Container:
        """Construye y retorna un contenedor de dependencias"""
        return Container(config_path)
    
    __all__ = ['build_container', 'Container', 'AnalysisService', 'GenAIGeminiAdapter']

except ImportError as e:
    import traceback
    error_detail = traceback.format_exc()
    raise ImportError(
        f"No se pudo importar los módulos base.\n"
        f"SRC_DIR: {SRC_DIR}\n"
        f"sys.path[0:3]: {sys.path[0:3]}\n"
        f"Error: {e}\n"
        f"Traceback:\n{error_detail}"
    ) from e



