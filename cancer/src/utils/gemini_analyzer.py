"""
Utilidades para integrar Gemini AI en el análisis de imágenes médicas.
Proporciona funcionalidades para análisis de imágenes usando la API de Gemini.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from PIL import Image
import numpy as np

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:  # noqa: BLE001
    GENAI_AVAILABLE = False

from .config import load_config, configure_logging

# Mensajes constantes
GENAI_UNAVAILABLE_MSG = "google-generativeai no disponible"

class GeminiAnalyzer:
    """Cliente para análisis de imágenes médicas usando Gemini AI."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el analizador Gemini.
        
        Args:
            config_path: Ruta al archivo de configuración JSON
        """
        # Cargar configuración
        self.config = load_config(Path(config_path) if config_path else None)

        # Configurar logging y logger de módulo
        configure_logging()
        self.logger = logging.getLogger(__name__)

        # Descargo legal centralizado
        self._disclaimer_text = str(self.config.legal.report_disclaimer)

        # Validar disponibilidad de Gemini
        if not GENAI_AVAILABLE:
            self.logger.warning("google-generativeai no está instalado. Instale con: pip install google-generativeai")
        else:
            api_key = self.config.gemini.api_key
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY no configurada. Defínala en el entorno o en config/config.json")
            genai.configure(api_key=api_key)  # type: ignore[attr-defined]
            self.model: Any = genai.GenerativeModel(self.config.gemini.model)  # type: ignore[attr-defined]
        
        # Parámetros de generación
        self.generation_config = {
            'temperature': self.config.gemini.temperature,
            'max_output_tokens': self.config.gemini.max_tokens,
        }

    def _append_disclaimer(self, text: str) -> str:
        """Adjunta el descargo legal a una salida de texto si no está presente."""
        try:
            disc = self._disclaimer_text.strip()
        except Exception:
            disc = ""
        if not disc:
            return text
        low = text.lower()
        if disc.lower() in low:
            return text
        sep = "\n\n" if not text.endswith("\n") else "\n"
        return f"{text}{sep}{disc}"

    def _call_with_retry(self, parts, *, retries: int = 2, backoff: float = 0.75):
        """Envuelve generate_content con reintentos simples (rate limits/temporales)."""
        if not GENAI_AVAILABLE:
            raise RuntimeError(GENAI_UNAVAILABLE_MSG)
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return self.model.generate_content(parts, generation_config=self.generation_config)  # type: ignore[arg-type]
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if attempt == retries:
                    break
                import time
                sleep_s = backoff * (2 ** attempt)
                self.logger.warning(f"Gemini fallo transitorio ({e}); reintentando en {sleep_s:.2f}s…")
                time.sleep(sleep_s)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Fallo desconocido en llamada a Gemini")
    
    def analyze_medical_image(self, image_path: str, analysis_type: str = "general") -> Dict[str, Any]:
        """
        Analiza una imagen médica usando Gemini.
        
        Args:
            image_path: Ruta a la imagen médica
            analysis_type: Tipo de análisis ('general', 'cancer_detection', 'radiomics')
            
        Returns:
            Diccionario con resultados del análisis
        """
        try:
            # Cargar imagen
            image = Image.open(image_path)
            
            # Generar prompt según el tipo de análisis
            prompt = self._generate_prompt(analysis_type)
            
            # Realizar análisis
            if not GENAI_AVAILABLE:
                raise RuntimeError(GENAI_UNAVAILABLE_MSG)

            self.logger.info(
                "Gemini analyze_medical_image: type=%s, model=%s, image=%s",
                analysis_type, getattr(self.config.gemini, 'model', ''), image_path
            )
            response = self._call_with_retry([prompt, image])
            
            result = {
                'image_path': image_path,
                'analysis_type': analysis_type,
                'gemini_response': self._append_disclaimer(response.text),
                'confidence_indicators': self._extract_confidence_indicators(response.text),
                'findings': self._extract_findings(response.text),
                'recommendations': self._extract_recommendations(response.text),
                'disclaimer_added': True,
            }
            
            self.logger.info(f"Análisis completado para {image_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error en análisis de imagen {image_path}: {e}")
            return {"error": str(e), "image_path": image_path}
    
    def batch_analyze_images(self, image_paths: List[str], 
                           analysis_type: str = "general") -> List[Dict[str, Any]]:
        """
        Analiza múltiples imágenes en lote.
        
        Args:
            image_paths: Lista de rutas de imágenes
            analysis_type: Tipo de análisis
            
        Returns:
            Lista de resultados de análisis
        """
        results = []
        
        for image_path in image_paths:
            self.logger.info(f"Analizando imagen {len(results)+1}/{len(image_paths)}: {image_path}")
            result = self.analyze_medical_image(image_path, analysis_type)
            results.append(result)
            
            # Pequeña pausa para evitar límites de rate
            import time
            time.sleep(1)
        
        return results
    
    def compare_images(self, image_path_1: str, image_path_2: str, 
                      comparison_type: str = "temporal") -> Dict[str, Any]:
        """
        Compara dos imágenes médicas.
        
        Args:
            image_path_1: Ruta a la primera imagen
            image_path_2: Ruta a la segunda imagen  
            comparison_type: Tipo de comparación ('temporal', 'pre_post_treatment')
            
        Returns:
            Resultados de la comparación
        """
        try:
            # Cargar imágenes
            image1 = Image.open(image_path_1)
            image2 = Image.open(image_path_2)
            
            prompt = self._generate_comparison_prompt(comparison_type)
            
            if not GENAI_AVAILABLE:
                raise RuntimeError(GENAI_UNAVAILABLE_MSG)

            self.logger.info(
                "Gemini compare_images: type=%s, model=%s, image1=%s, image2=%s",
                comparison_type, getattr(self.config.gemini, 'model', ''), image_path_1, image_path_2
            )
            response = self._call_with_retry([prompt, "Primera imagen:", image1, "Segunda imagen:", image2])
            
            result = {
                'image_1': image_path_1,
                'image_2': image_path_2,
                'comparison_type': comparison_type,
                'gemini_response': self._append_disclaimer(response.text),
                'changes_detected': self._extract_changes(response.text),
                'progression_assessment': self._extract_progression(response.text),
                'disclaimer_added': True,
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en comparación de imágenes: {e}")
            return {"error": str(e)}
    
    def analyze_roi(self, image_path: str, roi_coordinates: Dict[str, int]) -> Dict[str, Any]:
        """
        Analiza una región de interés específica en una imagen.
        
        Args:
            image_path: Ruta a la imagen
            roi_coordinates: Coordenadas de la ROI {'x': int, 'y': int, 'width': int, 'height': int}
            
        Returns:
            Análisis de la ROI
        """
        try:
            # Cargar y recortar imagen
            image = Image.open(image_path)
            roi = image.crop((
                roi_coordinates['x'],
                roi_coordinates['y'],
                roi_coordinates['x'] + roi_coordinates['width'],
                roi_coordinates['y'] + roi_coordinates['height']
            ))
            
            prompt = """
            Analiza esta región de interés (ROI) extraída de una imagen médica.
            Proporciona un análisis detallado de:
            1. Características morfológicas observadas
            2. Densidad y textura
            3. Posibles hallazgos patológicos
            4. Recomendaciones para investigación adicional
            
            Sé específico y técnico en tu análisis.
            """
            
            if not GENAI_AVAILABLE:
                raise RuntimeError(GENAI_UNAVAILABLE_MSG)

            self.logger.info(
                "Gemini analyze_roi: model=%s, image=%s, roi=%s",
                getattr(self.config.gemini, 'model', ''), image_path, roi_coordinates
            )
            response = self._call_with_retry([prompt, roi])
            
            result = {
                'image_path': image_path,
                'roi_coordinates': roi_coordinates,
                'gemini_response': self._append_disclaimer(response.text),
                'roi_analysis': self._extract_roi_findings(response.text),
                'disclaimer_added': True,
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en análisis de ROI: {e}")
            return {"error": str(e)}
    
    def generate_report(self, analysis_results: List[Dict[str, Any]], 
                       patient_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Genera un reporte médico basado en múltiples análisis.
        
        Args:
            analysis_results: Lista de resultados de análisis
            patient_info: Información del paciente (opcional)
            
        Returns:
            Reporte médico estructurado
        """
        try:
            # Preparar contexto para el reporte
            findings_summary = []
            for result in analysis_results:
                if 'findings' in result:
                    findings_summary.extend(result['findings'])
            
            prompt = f"""
            Basándote en los siguientes hallazgos de análisis de imágenes médicas, 
            genera un reporte médico estructurado y profesional:
            
            HALLAZGOS:
            {json.dumps(findings_summary, indent=2)}
            
            {"INFORMACIÓN DEL PACIENTE:" + json.dumps(patient_info, indent=2) if patient_info else ""}
            
            El reporte debe incluir:
            1. RESUMEN EJECUTIVO
            2. HALLAZGOS PRINCIPALES
            3. INTERPRETACIÓN CLÍNICA
            4. RECOMENDACIONES
            5. SEGUIMIENTO SUGERIDO
            
            Usa terminología médica apropiada y mantén un tono profesional.
            """
            
            if not GENAI_AVAILABLE:
                raise RuntimeError("google-generativeai no disponible")

            self.logger.info(
                "Gemini generate_report: n_results=%d, model=%s, has_patient_info=%s",
                len(analysis_results), getattr(self.config.gemini, 'model', ''), bool(patient_info)
            )

            response = self._call_with_retry([prompt])
            return self._append_disclaimer(response.text)
            
        except Exception as e:
            self.logger.error(f"Error en generación de reporte: {e}")
            return f"Error al generar reporte: {e}"
    
    def _generate_prompt(self, analysis_type: str) -> str:
        """Genera prompts específicos según el tipo de análisis."""
        prompts = {
            "general": """
            Analiza esta imagen médica y proporciona:
            1. Tipo de imagen/modalidad identificada
            2. Estructuras anatómicas visibles
            3. Hallazgos normales y anormales
            4. Posibles diagnósticos diferenciales
            5. Calidad de la imagen y limitaciones
            
            Sé preciso y técnico en tu análisis.
            """,
            
            "cancer_detection": """
            Analiza esta imagen médica específicamente para detección de cáncer:
            1. Identifica cualquier lesión o masa sospechosa
            2. Evalúa características morfológicas (forma, márgenes, densidad)
            3. Assess probability of malignancy based on imaging features
            4. Suggest staging if applicable
            5. Recommend additional imaging or biopsy if needed
            
            Proporciona un nivel de confianza para tus hallazgos.
            """,
            
            "radiomics": """
            Realiza un análisis radiómico de esta imagen:
            1. Describe características de textura
            2. Evalúa heterogeneidad de la lesión
            3. Analiza patrones de enhancemet si es visible
            4. Identifica características cuantificables
            5. Suggest relevant radiomic features for further analysis
            
            Enfócate en características que podrían ser útiles para análisis cuantitativo.
            """
        }
        
        return prompts.get(analysis_type, prompts["general"])
    
    def _generate_comparison_prompt(self, comparison_type: str) -> str:
        """Genera prompts para comparación de imágenes."""
        prompts = {
            "temporal": """
            Compara estas dos imágenes médicas tomadas en diferentes momentos:
            1. Identifica cambios morfológicos
            2. Evalúa progresión o regresión de lesiones
            3. Assess treatment response if applicable
            4. Note any new findings
            5. Provide overall assessment of disease progression
            """,
            
            "pre_post_treatment": """
            Compara estas imágenes pre y post-tratamiento:
            1. Evaluate treatment response
            2. Measure changes in lesion size/characteristics
            3. Identify any treatment-related changes
            4. Assess for progression or new lesions
            5. Provide RECIST or similar response criteria if applicable
            """
        }
        
        return prompts.get(comparison_type, prompts["temporal"])
    
    def _extract_confidence_indicators(self, text: str) -> List[str]:
        """Extrae indicadores de confianza del texto de respuesta."""
        confidence_terms = [
            "altamente probable", "probable", "posible", "unlikely", "definitivo",
            "sugestivo de", "consistente con", "compatible con", "sospechoso de"
        ]
        
        found_indicators = []
        text_lower = text.lower()
        
        for term in confidence_terms:
            if term in text_lower:
                found_indicators.append(term)
        
        return found_indicators
    
    def _extract_findings(self, text: str) -> List[str]:
        """Extrae hallazgos principales del texto de respuesta."""
        # Implementación simplificada - podría mejorarse con NLP más sofisticado
        lines = text.split('\n')
        findings = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['lesión', 'masa', 'nodulo', 'anormalidad']):
                findings.append(line)
        
        return findings
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extrae recomendaciones del texto de respuesta."""
        lines = text.split('\n')
        recommendations = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['recomiendo', 'sugiero', 'seguimiento', 'biopsia']):
                recommendations.append(line)
        
        return recommendations
    
    def _extract_changes(self, text: str) -> List[str]:
        """Extrae cambios detectados en comparación de imágenes."""
        lines = text.split('\n')
        changes = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['cambio', 'diferencia', 'progresion', 'regresion']):
                changes.append(line)
        
        return changes
    
    def _extract_progression(self, text: str) -> str:
        """Extrae evaluación de progresión."""
        text_lower = text.lower()
        
        if 'progresion' in text_lower:
            return 'Progresión detectada'
        elif 'regresion' in text_lower or 'mejoria' in text_lower:
            return 'Regresión/Mejoría detectada'
        elif 'estable' in text_lower:
            return 'Enfermedad estable'
        else:
            return 'Evaluación no concluyente'
    
    def _extract_roi_findings(self, text: str) -> Dict[str, Any]:
        """Extrae hallazgos específicos de análisis de ROI."""
        return {
            'morphology': self._extract_morphology_features(text),
            'texture': self._extract_texture_features(text),
            'pathological_findings': self._extract_findings(text)
        }
    
    def _extract_morphology_features(self, text: str) -> List[str]:
        """Extrae características morfológicas."""
        morphology_terms = ['forma', 'margen', 'contorno', 'tamaño', 'densidad']
        found_features = []
        
        for term in morphology_terms:
            if term in text.lower():
                # Extraer contexto alrededor del término
                sentences = text.split('.')
                for sentence in sentences:
                    if term in sentence.lower():
                        found_features.append(sentence.strip())
        
        return found_features
    
    def _extract_texture_features(self, text: str) -> List[str]:
        """Extrae características de textura."""
        texture_terms = ['textura', 'heterogeneo', 'homogeneo', 'rugoso', 'liso']
        found_features = []
        
        for term in texture_terms:
            if term in text.lower():
                sentences = text.split('.')
                for sentence in sentences:
                    if term in sentence.lower():
                        found_features.append(sentence.strip())
        
        return found_features


def load_gemini_analyzer():
    """Función helper para cargar el analizador Gemini con configuración por defecto."""
    return GeminiAnalyzer()


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar analizador
    analyzer = GeminiAnalyzer()
    
    # Ejemplo de análisis (requiere imagen real)
    # result = analyzer.analyze_medical_image("path/to/medical/image.jpg", "cancer_detection")
    # print(json.dumps(result, indent=2))
    
    print("Analizador Gemini inicializado correctamente")