"""
Utilidades para el proyecto de análisis de cáncer.
Incluye módulos para conexión TCIA, análisis con Gemini AI y procesamiento de imágenes DICOM.
"""

from .tcia_client import TCIAClient, load_tcia_utils

try:
    from .gemini_analyzer import GeminiAnalyzer, load_gemini_analyzer
except ImportError:
    print("Advertencia: google-generativeai no está instalado. Instale con: pip install google-generativeai")
    GeminiAnalyzer = None
    load_gemini_analyzer = None

__all__ = [
    'TCIAClient', 
    'load_tcia_utils',
    'GeminiAnalyzer', 
    'load_gemini_analyzer'
]