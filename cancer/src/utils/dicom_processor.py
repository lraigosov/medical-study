"""
Utilidades para procesamiento de imágenes DICOM en análisis de cáncer.
Incluye funciones para cargar, procesar, normalizar y preprocesar imágenes médicas.
"""

import json
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage

# Importaciones opcionales para DICOM y imágenes
try:
    import pydicom
    from pydicom import dcmread
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    warnings.warn("pydicom no está instalado. Instale con: pip install pydicom")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("Pillow no está instalado. Instale con: pip install Pillow")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    warnings.warn("OpenCV no está instalado. Instale con: pip install opencv-python")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    warnings.warn("nibabel no está instalado. Instale con: pip install nibabel")

from .config_loader import load_config, configure_logging


class DICOMProcessor:
    """Procesador de imágenes DICOM para análisis de cáncer."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el procesador DICOM.
        
        Args:
            config_path: Ruta al archivo de configuración JSON
        """
        cfg = load_config(Path(config_path) if config_path else None)

        # Configurar logging
        self.logger = configure_logging(cfg)
        self.logger = logging.getLogger(__name__)
        
        # Parámetros de configuración
        data_cfg = cfg.get("data", {})
        tcia_cfg = cfg.get("tcia", {})
        self.image_size = tuple(data_cfg.get("image_size", [224, 224]))
        self.supported_formats = tcia_cfg.get("supported_formats", ["DICOM", "NIfTI", "ANALYZE"])
        
        self.logger.info("Procesador DICOM inicializado")
    
    def extract_zip_to_temp(self, zip_path: str) -> str:
        """
        Extrae un archivo ZIP a un directorio temporal.
        
        Args:
            zip_path: Ruta al archivo ZIP
            
        Returns:
            Ruta del directorio temporal
        """
        temp_dir = tempfile.mkdtemp()
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            self.logger.info(f"ZIP extraído a: {temp_dir}")
            return temp_dir
            
        except Exception as e:
            self.logger.error(f"Error al extraer ZIP {zip_path}: {e}")
            return ""
    
    def load_dicom_file(self, dicom_path: str) -> Optional[Any]:
        """
        Carga un archivo DICOM.
        
        Args:
            dicom_path: Ruta al archivo DICOM
            
        Returns:
            Dataset DICOM o None si hay error
        """
        if not DICOM_AVAILABLE:
            self.logger.error("pydicom no está disponible")
            return None
        
        try:
            dicom_data = dcmread(dicom_path)
            self.logger.debug(f"DICOM cargado: {dicom_path}")
            return dicom_data
            
        except Exception as e:
            self.logger.error(f"Error al cargar DICOM {dicom_path}: {e}")
            return None
    
    def load_dicom_series(self, directory: str) -> List[Any]:
        """
        Carga una serie completa de archivos DICOM desde un directorio.
        
        Args:
            directory: Directorio que contiene archivos DICOM
            
        Returns:
            Lista de datasets DICOM
        """
        if not DICOM_AVAILABLE:
            self.logger.error("pydicom no está disponible")
            return []
        
        dicom_files = []
        directory_path = Path(directory)
        
        # Buscar archivos DICOM (pueden no tener extensión .dcm)
        for file_path in directory_path.rglob('*'):
            if file_path.is_file():
                try:
                    dicom_data = dcmread(str(file_path))
                    dicom_files.append(dicom_data)
                except Exception:
                    # No es un archivo DICOM válido
                    continue
        
        # Ordenar por posición de slice si está disponible
        try:
            dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except (AttributeError, KeyError, TypeError):
            # Si no hay información de posición, ordenar por instance number
            try:
                dicom_files.sort(key=lambda x: int(x.InstanceNumber))
            except (AttributeError, KeyError, TypeError):
                self.logger.warning("No se pudo ordenar la serie DICOM")
        
        self.logger.info(f"Cargados {len(dicom_files)} archivos DICOM de {directory}")
        return dicom_files
    
    def dicom_to_array(self, dicom_data: Any) -> Optional[np.ndarray]:
        """
        Convierte datos DICOM a array NumPy.
        
        Args:
            dicom_data: Dataset DICOM
            
        Returns:
            Array NumPy con los datos de la imagen
        """
        try:
            # Obtener array de píxeles
            pixel_array = dicom_data.pixel_array
            
            # Aplicar transformaciones de ventana/nivel si están disponibles
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                def _first_float(v):
                    try:
                        # MultiValue u otras colecciones
                        if hasattr(v, '__iter__') and not isinstance(v, (str, bytes)):
                            return float(v[0])
                        return float(v)
                    except Exception:
                        return float(np.mean(v)) if hasattr(v, '__iter__') else float(v)

                wc = _first_float(dicom_data.WindowCenter)
                ww = _first_float(dicom_data.WindowWidth)
                pixel_array = self.apply_windowing(pixel_array, wc, ww)
            
            # Aplicar escala de rescate si está disponible
            if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                try:
                    slope = float(dicom_data.RescaleSlope)
                    inter = float(dicom_data.RescaleIntercept)
                except Exception:
                    slope = 1.0
                    inter = 0.0
                pixel_array = pixel_array * slope + inter
            
            return pixel_array.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error al convertir DICOM a array: {e}")
            return None
    
    def apply_windowing(self, pixel_array: np.ndarray, window_center: float, 
                       window_width: float) -> np.ndarray:
        """
        Aplica windowing/leveling a una imagen DICOM.
        
        Args:
            pixel_array: Array de píxeles
            window_center: Centro de la ventana
            window_width: Ancho de la ventana
            
        Returns:
            Array con windowing aplicado
        """
        window_min = window_center - window_width // 2
        window_max = window_center + window_width // 2
        
        # Aplicar windowing
        windowed = np.clip(pixel_array, window_min, window_max)
        
        # Normalizar a rango 0-255
        windowed = ((windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)
        
        return windowed
    
    def normalize_image(self, image: np.ndarray, method: str = "minmax") -> np.ndarray:
        """
        Normaliza una imagen usando diferentes métodos.
        
        Args:
            image: Array de imagen
            method: Método de normalización ('minmax', 'zscore', 'robust')
            
        Returns:
            Imagen normalizada
        """
        if method == "minmax":
            # Normalización Min-Max [0, 1]
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                normalized = (image - img_min) / (img_max - img_min)
            else:
                normalized = image
                
        elif method == "zscore":
            # Normalización Z-score
            mean, std = image.mean(), image.std()
            if std > 0:
                normalized = (image - mean) / std
            else:
                normalized = image - mean
                
        elif method == "robust":
            # Normalización robusta usando percentiles
            p2, p98 = np.percentile(image, [2, 98])
            if p98 > p2:
                normalized = np.clip((image - p2) / (p98 - p2), 0, 1)
            else:
                normalized = image
                
        else:
            self.logger.warning(f"Método de normalización desconocido: {method}")
            normalized = image
        
        return normalized.astype(np.float32)
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                    method: str = "bilinear") -> np.ndarray:
        """
        Redimensiona una imagen.
        
        Args:
            image: Array de imagen
            target_size: Tamaño objetivo (height, width)
            method: Método de interpolación
            
        Returns:
            Imagen redimensionada
        """
        if not OPENCV_AVAILABLE:
            # Fallback usando scipy
            zoom_factors = (target_size[0] / image.shape[0], target_size[1] / image.shape[1])
            if len(image.shape) == 3:
                zoom_factors = zoom_factors + (1,)
            
            return ndimage.zoom(image, zoom_factors, order=1)
        
        # Usar OpenCV si está disponible
        if method == "bilinear":
            interpolation = cv2.INTER_LINEAR
        elif method == "cubic":
            interpolation = cv2.INTER_CUBIC
        elif method == "nearest":
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_LINEAR
        
        return cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
    
    def enhance_contrast(self, image: np.ndarray, method: str = "clahe") -> np.ndarray:
        """
        Mejora el contraste de una imagen.
        
        Args:
            image: Array de imagen
            method: Método de mejora ('clahe', 'histogram_eq', 'adaptive')
            
        Returns:
            Imagen con contraste mejorado
        """
        if not OPENCV_AVAILABLE:
            self.logger.warning("OpenCV no disponible para mejora de contraste")
            return image
        
        # Convertir a uint8 si es necesario
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.copy()
        
        if method == "clahe":
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image_uint8)
            
        elif method == "histogram_eq":
            # Ecualización de histograma global
            enhanced = cv2.equalizeHist(image_uint8)
            
        elif method == "adaptive":
            # Ecualización adaptiva
            enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image_uint8)
            
        else:
            self.logger.warning(f"Método de mejora desconocido: {method}")
            enhanced = image_uint8
        
        return enhanced.astype(np.float32) / 255.0
    
    def remove_noise(self, image: np.ndarray, method: str = "gaussian") -> np.ndarray:
        """
        Reduce el ruido en una imagen.
        
        Args:
            image: Array de imagen
            method: Método de reducción de ruido
            
        Returns:
            Imagen con ruido reducido
        """
        if method == "gaussian":
            # Filtro gaussiano
            return ndimage.gaussian_filter(image, sigma=0.5)
            
        elif method == "median" and OPENCV_AVAILABLE:
            # Filtro mediano
            if image.dtype != np.uint8:
                img_uint8 = (image * 255).astype(np.uint8)
                filtered = cv2.medianBlur(img_uint8, 3)
                return filtered.astype(np.float32) / 255.0
            else:
                return cv2.medianBlur(image, 3)
                
        elif method == "bilateral" and OPENCV_AVAILABLE:
            # Filtro bilateral
            if image.dtype != np.uint8:
                img_uint8 = (image * 255).astype(np.uint8)
                filtered = cv2.bilateralFilter(img_uint8, 9, 75, 75)
                return filtered.astype(np.float32) / 255.0
            else:
                return cv2.bilateralFilter(image, 9, 75, 75)
        else:
            # Fallback a filtro gaussiano
            return ndimage.gaussian_filter(image, sigma=0.5)
    
    def extract_roi(self, image: np.ndarray, roi_coords: Dict[str, int]) -> np.ndarray:
        """
        Extrae una región de interés de una imagen.
        
        Args:
            image: Array de imagen
            roi_coords: Coordenadas {'x': int, 'y': int, 'width': int, 'height': int}
            
        Returns:
            ROI extraída
        """
        x, y = roi_coords['x'], roi_coords['y']
        w, h = roi_coords['width'], roi_coords['height']
        
        # Validar coordenadas
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        x2 = min(x + w, image.shape[1])
        y2 = min(y + h, image.shape[0])
        
        return image[y:y2, x:x2]
    
    def preprocess_for_ml(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Preprocesa una imagen para análisis de machine learning.
        
        Args:
            image: Array de imagen
            target_size: Tamaño objetivo (opcional)
            
        Returns:
            Imagen preprocesada
        """
        if target_size is None:
            target_size = self.image_size
        
        # Pipeline de preprocesamiento
        processed = image.copy()
        
        # 1. Reducir ruido
        processed = self.remove_noise(processed, method="gaussian")
        
        # 2. Normalizar
        processed = self.normalize_image(processed, method="robust")
        
        # 3. Mejorar contraste
        processed = self.enhance_contrast(processed, method="clahe")
        
        # 4. Redimensionar
        if processed.shape[:2] != target_size:
            processed = self.resize_image(processed, target_size)
        
        # 5. Asegurar que tenga 3 canales para modelos RGB
        if len(processed.shape) == 2:
            processed = np.stack([processed] * 3, axis=-1)
        elif processed.shape[-1] == 1:
            processed = np.repeat(processed, 3, axis=-1)
        
        return processed
    
    def save_processed_image(self, image: np.ndarray, output_path: str, format: str = "PNG") -> bool:
        """
        Guarda una imagen procesada.
        
        Args:
            image: Array de imagen
            output_path: Ruta de salida
            format: Formato de imagen
            
        Returns:
            True si se guardó exitosamente
        """
        if not PIL_AVAILABLE:
            self.logger.error("PIL no está disponible para guardar imágenes")
            return False
        
        try:
            # Convertir a rango 0-255 si es necesario
            if image.dtype == np.float32 or image.dtype == np.float64:
                if image.max() <= 1.0:
                    image_save = (image * 255).astype(np.uint8)
                else:
                    image_save = image.astype(np.uint8)
            else:
                image_save = image
            
            # Crear imagen PIL
            if len(image_save.shape) == 2:
                pil_image = Image.fromarray(image_save, mode='L')
            elif len(image_save.shape) == 3:
                pil_image = Image.fromarray(image_save, mode='RGB')
            else:
                self.logger.error(f"Forma de imagen no soportada: {image_save.shape}")
                return False
            
            # Guardar imagen
            pil_image.save(output_path, format=format)
            self.logger.info(f"Imagen guardada: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al guardar imagen {output_path}: {e}")
            return False
    
    def get_dicom_metadata(self, dicom_data: Any) -> Dict[str, Any]:
        """
        Extrae metadatos relevantes de un dataset DICOM.
        
        Args:
            dicom_data: Dataset DICOM
            
        Returns:
            Diccionario con metadatos
        """
        metadata = {}
        
        # Metadatos básicos
        metadata_fields = [
            'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID',
            'Modality', 'BodyPartExamined', 'SliceThickness',
            'PixelSpacing', 'ImagePositionPatient', 'ImageOrientationPatient',
            'Rows', 'Columns', 'WindowCenter', 'WindowWidth',
            'RescaleSlope', 'RescaleIntercept', 'PhotometricInterpretation'
        ]
        
        for field in metadata_fields:
            try:
                metadata[field] = getattr(dicom_data, field, None)
            except Exception:
                metadata[field] = None
        
        return metadata
    
    def process_dicom_series(self, zip_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Procesa una serie DICOM completa desde un archivo ZIP.
        
        Args:
            zip_path: Ruta al archivo ZIP con la serie DICOM
            output_dir: Directorio de salida para imágenes procesadas
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        results = {
            'zip_path': zip_path,
            'output_dir': output_dir,
            'processed_images': [],
            'metadata': [],
            'errors': []
        }
        
        # Crear directorio de salida
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extraer ZIP
        temp_dir = self.extract_zip_to_temp(zip_path)
        if not temp_dir:
            results['errors'].append("Error al extraer ZIP")
            return results
        
        try:
            # Cargar serie DICOM
            dicom_series = self.load_dicom_series(temp_dir)
            
            if not dicom_series:
                results['errors'].append("No se encontraron archivos DICOM válidos")
                return results
            
            # Procesar cada imagen de la serie
            for i, dicom_data in enumerate(dicom_series):
                try:
                    # Convertir a array
                    image_array = self.dicom_to_array(dicom_data)
                    if image_array is None:
                        continue
                    
                    # Preprocesar
                    processed_image = self.preprocess_for_ml(image_array)
                    
                    # Guardar imagen procesada
                    output_filename = f"image_{i:04d}.png"
                    output_filepath = output_path / output_filename
                    
                    if self.save_processed_image(processed_image, str(output_filepath)):
                        results['processed_images'].append(str(output_filepath))
                    
                    # Extraer metadatos
                    metadata = self.get_dicom_metadata(dicom_data)
                    metadata['processed_image_path'] = str(output_filepath)
                    metadata['slice_index'] = i
                    results['metadata'].append(metadata)
                    
                except Exception as e:
                    error_msg = f"Error procesando slice {i}: {e}"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            self.logger.info(f"Procesadas {len(results['processed_images'])} imágenes de {zip_path}")
            
        finally:
            # Limpiar directorio temporal
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        
        return results


def load_dicom_processor():
    """Función helper para cargar el procesador DICOM con configuración por defecto."""
    return DICOMProcessor()


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar procesador
    processor = DICOMProcessor()
    
    # Ejemplo de procesamiento (requiere archivo DICOM real)
    # results = processor.process_dicom_series("path/to/series.zip", "output/directory")
    # print(json.dumps(results, indent=2))
    
    print("Procesador DICOM inicializado correctamente")
    print(f"DICOM disponible: {DICOM_AVAILABLE}")
    print(f"PIL disponible: {PIL_AVAILABLE}")
    print(f"OpenCV disponible: {OPENCV_AVAILABLE}")
    print(f"Nibabel disponible: {NIBABEL_AVAILABLE}")