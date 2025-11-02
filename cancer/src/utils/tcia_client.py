"""
Utilidades para conexión y descarga de datos desde The Cancer Imaging Archive (TCIA).
Implementa conexión con la API REST de TCIA para acceder a colecciones de datos de cáncer.
"""

import json
import requests
import time
import logging
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path
import pandas as pd

from .config_loader import load_config, configure_logging, project_path

class TCIAClient:
    """Cliente para interactuar con la API REST de TCIA."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el cliente TCIA.
        
        Args:
            config_path: Ruta al archivo de configuración JSON
        """
        cfg = load_config(Path(config_path) if config_path else None)
        tcia_cfg = cfg.get("tcia", {})
        self.base_url = tcia_cfg.get("base_url", "https://services.cancerimagingarchive.net/nbia-api/services/v1")
        self.collections_url = tcia_cfg.get("collections_url", "https://www.cancerimagingarchive.net/wp-json/wp/v2/collections")
        # Resolver ruta de descarga a absoluta dentro del proyecto
        download_path_str = tcia_cfg.get("download_path", "./data/raw")
        self.download_path = project_path(download_path_str).resolve()
        self.session = requests.Session()

        # Configurar logging
        self.logger = configure_logging(cfg)
        self.logger = logging.getLogger(__name__)

    def _request_with_retry(
        self,
        method: Callable[..., requests.Response],
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        retries: Optional[int] = None,
        backoff: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Optional[requests.Response]:
        """Ejecuta una petición HTTP con reintentos exponenciales y timeout.

        - Reintenta en errores de red y códigos 5xx.
        - Aplica backoff exponencial con jitter pequeño.
        """
        # Leer valores desde config
        cfg = load_config()
        tcia_cfg = cfg.get("tcia", {})
        _retries = retries if retries is not None else tcia_cfg.get("retries", 3)
        _backoff = backoff if backoff is not None else tcia_cfg.get("backoff", 0.75)
        _timeout = timeout if timeout is not None else tcia_cfg.get("timeout", 30.0)
        
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= _retries:
            try:
                resp = method(url, params=params, timeout=_timeout)
                # Reintentar en 5xx
                if 500 <= resp.status_code < 600:
                    raise requests.HTTPError(f"{resp.status_code} server error", response=resp)
                return resp
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
                last_exc = e
                if attempt == _retries:
                    break
                sleep_s = _backoff * (2 ** attempt) + (0.05 * (attempt + 1))
                self.logger.warning(
                    f"HTTP fallo ({e}); reintentando en {sleep_s:.2f}s (intento {attempt+1}/{_retries})"
                )
                time.sleep(sleep_s)
                attempt += 1
            except Exception as e:  # noqa: BLE001
                # Errores no esperados: no reintentar
                self.logger.error(f"Fallo inesperado en petición HTTP: {e}")
                return None
        self.logger.error(f"Agotados reintentos para {url}: {last_exc}")
        return None
        
    def get_collections(self) -> List[Dict[str, Any]]:
        """
        Obtiene lista de todas las colecciones disponibles en TCIA.
        
        Returns:
            Lista de diccionarios con información de las colecciones
        """
        try:
            endpoint = f"{self.base_url}/getCollectionValues"
            response = self._request_with_retry(self.session.get, endpoint, params={"format": "json"})
            if response is None:
                return []
            response.raise_for_status()
            
            collections = response.json()
            self.logger.info(f"Obtenidas {len(collections)} colecciones de TCIA")
            return collections
            
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener colecciones: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Obtiene información detallada de una colección específica.
        
        Args:
            collection_name: Nombre de la colección
            
        Returns:
            Diccionario con información de la colección
        """
        try:
            # Usar la API de WordPress para obtener metadatos detallados
            response = self._request_with_retry(
                self.session.get,
                self.collections_url,
                params={"search": collection_name, "per_page": 1},
            )
            if response is None:
                return {}
            response.raise_for_status()
            
            collections = response.json()
            if collections:
                return collections[0]
            else:
                self.logger.warning(f"No se encontró información para la colección: {collection_name}")
                return {}
                
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener información de la colección {collection_name}: {e}")
            return {}
    
    def get_patients(self, collection: str) -> List[Dict[str, Any]]:
        """
        Obtiene lista de pacientes en una colección.
        
        Args:
            collection: Nombre de la colección
            
        Returns:
            Lista de pacientes
        """
        try:
            endpoint = f"{self.base_url}/getPatient"
            response = self._request_with_retry(
                self.session.get,
                endpoint,
                params={"Collection": collection, "format": "json"},
            )
            if response is None:
                return []
            response.raise_for_status()
            
            patients = response.json()
            self.logger.info(f"Obtenidos {len(patients)} pacientes de la colección {collection}")
            return patients
            
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener pacientes de {collection}: {e}")
            return []
    
    def get_studies(self, collection: str, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtiene estudios de una colección y opcionalmente de un paciente específico.
        
        Args:
            collection: Nombre de la colección
            patient_id: ID del paciente (opcional)
            
        Returns:
            Lista de estudios
        """
        try:
            endpoint = f"{self.base_url}/getPatientStudy"
            params = {"Collection": collection, "format": "json"}
            
            if patient_id:
                params["PatientID"] = patient_id
                
            response = self._request_with_retry(self.session.get, endpoint, params=params)
            if response is None:
                return []
            response.raise_for_status()
            
            studies = response.json()
            self.logger.info(f"Obtenidos {len(studies)} estudios")
            return studies
            
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener estudios: {e}")
            return []
    
    def get_series(self, collection: str, patient_id: Optional[str] = None, study_uid: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtiene series de imágenes.
        
        Args:
            collection: Nombre de la colección
            patient_id: ID del paciente (opcional)
            study_uid: UID del estudio (opcional)
            
        Returns:
            Lista de series
        """
        try:
            endpoint = f"{self.base_url}/getSeries"
            params = {"Collection": collection, "format": "json"}
            
            if patient_id:
                params["PatientID"] = patient_id
            if study_uid:
                params["StudyInstanceUID"] = study_uid
                
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            series = response.json()
            self.logger.info(f"Obtenidas {len(series)} series")
            return series
            
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener series: {e}")
            return []
    
    def get_images(self, series_uid: str) -> List[Dict[str, Any]]:
        """
        Obtiene lista de imágenes en una serie.
        
        Args:
            series_uid: UID de la serie
            
        Returns:
            Lista de imágenes
        """
        try:
            endpoint = f"{self.base_url}/getImage"
            response = self._request_with_retry(
                self.session.get,
                endpoint,
                params={"SeriesInstanceUID": series_uid, "format": "json"},
            )
            if response is None:
                return []
            response.raise_for_status()
            
            images = response.json()
            self.logger.info(f"Obtenidas {len(images)} imágenes de la serie {series_uid}")
            return images
            
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener imágenes de la serie {series_uid}: {e}")
            return []
    
    def download_series(self, series_uid: str, download_path: Optional[str] = None) -> str:
        """
        Descarga una serie completa de imágenes DICOM.
        
        Args:
            series_uid: UID de la serie a descargar
            download_path: Ruta de descarga (opcional)
            
        Returns:
            Ruta del archivo ZIP descargado
        """
        target_dir: Path
        if download_path is None:
            target_dir = Path(self.download_path)
        else:
            target_dir = Path(download_path)

        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Leer timeout desde config
        cfg = load_config()
        tcia_cfg = cfg.get("tcia", {})
        download_timeout = tcia_cfg.get("download_timeout", 120.0)
        
        try:
            endpoint = f"{self.base_url}/getImage"
            response = self._request_with_retry(
                self.session.get,
                endpoint,
                params={"SeriesInstanceUID": series_uid},
                timeout=download_timeout,
            )
            if response is None:
                return ""
            response.raise_for_status()
            
            # Generar nombre de archivo único
            filename = f"series_{series_uid.replace('.', '_')}.zip"
            filepath = target_dir / filename
            
            # Descargar archivo
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
            
            self.logger.info(f"Serie descargada: {filepath}")
            return str(filepath)
            
        except requests.RequestException as e:
            self.logger.error(f"Error al descargar serie {series_uid}: {e}")
            return ""
    
    def download_collection_sample(self, collection: str, max_patients: int = 10, 
                                 max_series_per_patient: int = 5) -> List[str]:
        """
        Descarga una muestra de datos de una colección.
        
        Args:
            collection: Nombre de la colección
            max_patients: Máximo número de pacientes
            max_series_per_patient: Máximo series por paciente
            
        Returns:
            Lista de rutas de archivos descargados
        """
        downloaded_files = []
        
        # Obtener pacientes
        patients = self.get_patients(collection)[:max_patients]
        
        for patient in patients:
            patient_id = patient.get('PatientID', '')
            self.logger.info(f"Procesando paciente: {patient_id}")
            
            # Obtener series del paciente
            series = self.get_series(collection, patient_id)[:max_series_per_patient]
            
            for serie in series:
                series_uid = serie.get('SeriesInstanceUID', '')
                if series_uid:
                    # Introducir delay para evitar sobrecarga del servidor
                    time.sleep(1)
                    
                    filepath = self.download_series(series_uid)
                    if filepath:
                        downloaded_files.append(filepath)
        
        self.logger.info(f"Descarga completada. {len(downloaded_files)} archivos descargados")
        return downloaded_files
    
    def get_collection_statistics(self, collection: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas de una colección.
        
        Args:
            collection: Nombre de la colección
            
        Returns:
            Diccionario con estadísticas
        """
        patients = self.get_patients(collection)
        studies = self.get_studies(collection)
        
        # Contar series por modalidad
        series = self.get_series(collection)
        modalities = {}
        for serie in series:
            modality = serie.get('Modality', 'Unknown')
            modalities[modality] = modalities.get(modality, 0) + 1
        
        stats = {
            'collection': collection,
            'total_patients': len(patients),
            'total_studies': len(studies),
            'total_series': len(series),
            'modalities': modalities
        }
        
        return stats

def load_tcia_utils():
    """Función helper para cargar utilidades TCIA con configuración por defecto."""
    return TCIAClient()

# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar cliente
    client = TCIAClient()
    
    # Obtener colecciones disponibles
    collections = client.get_collections()
    print(f"Colecciones disponibles: {len(collections)}")
    
    # Ejemplo con colección de cáncer de pulmón
    if collections:
        lung_collection = "CMB-LCA"  # Colección de cáncer de pulmón
        
        # Obtener estadísticas
        stats = client.get_collection_statistics(lung_collection)
        print(f"Estadísticas de {lung_collection}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Descargar muestra pequeña (opcional)
        # downloaded = client.download_collection_sample(lung_collection, max_patients=2, max_series_per_patient=1)
        # print(f"Archivos descargados: {downloaded}")