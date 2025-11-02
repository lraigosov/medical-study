"""
Análisis radiómicos avanzado para caracterización cuantitativa de lesiones cancerosas.
Implementa extracción de características de imagen y análisis estadístico.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
import warnings

import numpy as np
import pandas as pd

# Importaciones opcionales para análisis radiómico
try:
    import radiomics
    from radiomics import featureextractor
    from radiomics.imageoperations import resampleImage
    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False
    warnings.warn("PyRadiomics no está instalado. Instale con: pip install pyradiomics")

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    warnings.warn("SimpleITK no está instalado. Instale con: pip install SimpleITK")

try:
    from scipy import stats, ndimage
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy no está instalado. Instale con: pip install scipy")

try:
    from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn no está instalado. Instale con: pip install scikit-learn")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn no están instalados. Instale con: pip install matplotlib seaborn")

class RadiomicsAnalyzer:
    """Analizador de características radiómicas para imágenes médicas."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el analizador radiómico.
        
        Args:
            config_path: Ruta al archivo de configuración JSON
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.json"
        
        with open(str(config_path), 'r') as f:
            self.config = json.load(f)
        
        # Configurar logging
        logging.basicConfig(level=getattr(logging, self.config['logging']['level']))
        self.logger = logging.getLogger(__name__)
        
        # Parámetros radiómicos
        self.radiomics_config = self.config['radiomics']
        self.feature_classes = self.radiomics_config['feature_classes']
        self.bin_width = self.radiomics_config['bin_width']
        self.normalize = self.radiomics_config['normalize']
        self.resampling = self.radiomics_config['resample_pixel_spacing']
        
        # Inicializar extractor de características
        self.feature_extractor = None
        if RADIOMICS_AVAILABLE:
            self._initialize_feature_extractor()
        
        self.logger.info("Analizador radiómico inicializado")
    
    def _initialize_feature_extractor(self):
        """Inicializa el extractor de características PyRadiomics."""
        try:
            # Configurar parámetros
            settings = {
                'binWidth': self.bin_width,
                'resampledPixelSpacing': self.resampling,
                'interpolator': sitk.sitkBSpline,
                'normalize': self.normalize,
                'normalizeScale': 100,
                'removeOutliers': None
            }

            # Inicializar extractor con settings
            self.feature_extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
            self.feature_extractor.addProvenance(False)
            
            # Habilitar clases de características especificadas
            for feature_class in self.feature_classes:
                self.feature_extractor.enableFeatureClassByName(feature_class)
            
            # Configurar filtros adicionales
            self.feature_extractor.enableImageTypeByName('LoG', sigma=[2.0, 3.0, 4.0, 5.0])
            self.feature_extractor.enableImageTypeByName('Wavelet')
            
            self.logger.info("Feature extractor configurado exitosamente")
            
        except Exception as e:
            self.logger.error(f"Error configurando feature extractor: {e}")
            self.feature_extractor = None
    
    def extract_features_from_image(self, image_path: str, mask_path: str = None) -> Dict[str, float]:
        """
        Extrae características radiómicas de una imagen.
        
        Args:
            image_path: Ruta a la imagen médica
            mask_path: Ruta a la máscara de segmentación (opcional)
            
        Returns:
            Diccionario con características extraídas
        """
        if not RADIOMICS_AVAILABLE or self.feature_extractor is None:
            self.logger.error("PyRadiomics no está disponible")
            return {}
        
        try:
            # Cargar imagen
            if not SITK_AVAILABLE:
                self.logger.error("SimpleITK no está disponible")
                return {}
            
            image = sitk.ReadImage(image_path)
            
            # Generar máscara automática si no se proporciona
            if mask_path is None:
                mask = self._generate_automatic_mask(image)
            else:
                mask = sitk.ReadImage(mask_path)
            
            # Extraer características
            features = self.feature_extractor.execute(image, mask)
            
            # Filtrar solo características numéricas
            numeric_features = {
                key: float(value) for key, value in features.items()
                if not key.startswith('diagnostics_') and isinstance(value, (int, float))
            }
            
            self.logger.info(f"Extraídas {len(numeric_features)} características de {image_path}")
            return numeric_features
            
        except Exception as e:
            self.logger.error(f"Error extrayendo características de {image_path}: {e}")
            return {}
    
    def extract_features_batch(self, image_paths: List[str], 
                             mask_paths: List[str] = None) -> pd.DataFrame:
        """
        Extrae características radiómicas de múltiples imágenes.
        
        Args:
            image_paths: Lista de rutas de imágenes
            mask_paths: Lista de rutas de máscaras (opcional)
            
        Returns:
            DataFrame con características de todas las imágenes
        """
        if mask_paths is None:
            mask_paths = [None] * len(image_paths)
        
        features_list = []
        
        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            self.logger.info(f"Procesando imagen {i+1}/{len(image_paths)}: {img_path}")
            
            features = self.extract_features_from_image(img_path, mask_path)
            if features:
                features['image_path'] = img_path
                features['image_index'] = i
                features_list.append(features)
        
        if features_list:
            df = pd.DataFrame(features_list)
            df.set_index('image_path', inplace=True)
            self.logger.info(f"Características extraídas de {len(df)} imágenes")
            return df
        else:
            self.logger.warning("No se pudieron extraer características de ninguna imagen")
            return pd.DataFrame()
    
    def _generate_automatic_mask(self, image: Any) -> Any:
        """
        Genera una máscara automática para la imagen.
        
        Args:
            image: Imagen SimpleITK
            
        Returns:
            Máscara binaria
        """
        try:
            # Convertir a array numpy
            image_array = sitk.GetArrayFromImage(image)
            
            # Aplicar threshold automático (método Otsu)
            if SCIPY_AVAILABLE:
                from skimage.filters import threshold_otsu
                threshold = threshold_otsu(image_array)
            else:
                # Threshold simple basado en percentiles
                threshold = np.percentile(image_array, 75)
            
            # Crear máscara binaria
            mask_array = (image_array > threshold).astype(np.uint8)
            
            # Crear imagen SimpleITK de la máscara
            mask = sitk.GetImageFromArray(mask_array)
            mask.CopyInformation(image)
            
            return mask
            
        except Exception as e:
            self.logger.error(f"Error generando máscara automática: {e}")
            # Máscara de toda la imagen como fallback
            mask_array = np.ones_like(sitk.GetArrayFromImage(image), dtype=np.uint8)
            mask = sitk.GetImageFromArray(mask_array)
            mask.CopyInformation(image)
            return mask
    
    def analyze_feature_correlations(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza correlaciones entre características radiómicas.
        
        Args:
            features_df: DataFrame con características
            
        Returns:
            Análisis de correlaciones
        """
        try:
            # Calcular matriz de correlación
            numeric_features = features_df.select_dtypes(include=[np.number])
            correlation_matrix = numeric_features.corr()
            
            # Encontrar características altamente correlacionadas
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:  # Umbral de alta correlación
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            # Estadísticas de correlación
            corr_stats = {
                'mean_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
                'max_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max(),
                'min_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min(),
                'high_corr_count': len(high_corr_pairs)
            }
            
            results = {
                'correlation_matrix': correlation_matrix,
                'high_correlation_pairs': high_corr_pairs,
                'correlation_statistics': corr_stats
            }
            
            self.logger.info(f"Análisis de correlación completado. {len(high_corr_pairs)} pares altamente correlacionados")
            return results
            
        except Exception as e:
            self.logger.error(f"Error en análisis de correlaciones: {e}")
            return {}
    
    def perform_feature_selection(self, X: pd.DataFrame, y: np.ndarray, 
                                method: str = "univariate", k: int = 50) -> Dict[str, Any]:
        """
        Realiza selección de características.
        
        Args:
            X: DataFrame con características
            y: Array con etiquetas
            method: Método de selección ('univariate', 'rfe', 'lasso')
            k: Número de características a seleccionar
            
        Returns:
            Resultados de la selección de características
        """
        if not SKLEARN_AVAILABLE:
            self.logger.error("scikit-learn no está disponible")
            return {}
        
        try:
            # Preparar datos
            numeric_features = X.select_dtypes(include=[np.number])
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(numeric_features)
            
            results = {}
            
            if method == "univariate":
                # Selección univariada
                selector = SelectKBest(score_func=f_classif, k=k)
                x_selected = selector.fit_transform(x_scaled, y)
                
                selected_features = numeric_features.columns[selector.get_support()].tolist()
                feature_scores = dict(zip(numeric_features.columns, selector.scores_))
                
            elif method == "rfe":
                # Recursive Feature Elimination
                estimator = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=1, max_features='sqrt')
                selector = RFE(estimator, n_features_to_select=k)
                x_selected = selector.fit_transform(x_scaled, y)
                
                selected_features = numeric_features.columns[selector.get_support()].tolist()
                feature_scores = dict(zip(numeric_features.columns, selector.ranking_))
                
            elif method == "lasso":
                # LASSO regularization
                lasso = LassoCV(cv=5, random_state=42)
                lasso.fit(x_scaled, y)
                
                # Seleccionar características con coeficientes no cero
                selected_mask = np.abs(lasso.coef_) > 0
                selected_features = numeric_features.columns[selected_mask].tolist()
                feature_scores = dict(zip(numeric_features.columns, np.abs(lasso.coef_)))
                
                x_selected = x_scaled[:, selected_mask]
                
            else:
                self.logger.error(f"Método de selección no reconocido: {method}")
                return {}
            
            results = {
                'method': method,
                'selected_features': selected_features,
                'feature_scores': feature_scores,
                'n_selected': len(selected_features),
                'X_selected': x_selected
            }
            
            self.logger.info(f"Selección de características ({method}) completada. {len(selected_features)} características seleccionadas")
            return results
            
        except Exception as e:
            self.logger.error(f"Error en selección de características: {e}")
            return {}
    
    def perform_statistical_analysis(self, features_df: pd.DataFrame, 
                                   labels: np.ndarray) -> Dict[str, Any]:
        """
        Realiza análisis estadístico de características.
        
        Args:
            features_df: DataFrame con características
            labels: Etiquetas de clasificación
            
        Returns:
            Resultados del análisis estadístico
        """
        try:
            results = {}
            numeric_features = features_df.select_dtypes(include=[np.number])
            
            # Dividir por grupos según las etiquetas
            unique_labels = np.unique(labels)
            
            if len(unique_labels) == 2:
                # Análisis para clasificación binaria
                group1 = numeric_features[labels == unique_labels[0]]
                group2 = numeric_features[labels == unique_labels[1]]
                
                statistical_tests = []
                
                for feature in numeric_features.columns:
                    try:
                        # Test de normalidad
                        _, p_normal1 = stats.shapiro(group1[feature].dropna())
                        _, p_normal2 = stats.shapiro(group2[feature].dropna())
                        
                        # Seleccionar test apropiado
                        if p_normal1 > 0.05 and p_normal2 > 0.05:
                            # Datos normales: t-test
                            statistic, p_value = stats.ttest_ind(
                                group1[feature].dropna(), 
                                group2[feature].dropna()
                            )
                            test_type = "t-test"
                        else:
                            # Datos no normales: Mann-Whitney U
                            statistic, p_value = stats.mannwhitneyu(
                                group1[feature].dropna(), 
                                group2[feature].dropna(),
                                alternative='two-sided'
                            )
                            test_type = "mann-whitney"
                        
                        # Calcular tamaño del efecto (Cohen's d)
                        mean1, mean2 = group1[feature].mean(), group2[feature].mean()
                        std1, std2 = group1[feature].std(), group2[feature].std()
                        pooled_std = np.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) / 
                                           (len(group1) + len(group2) - 2))
                        cohens_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0
                        
                        statistical_tests.append({
                            'feature': feature,
                            'test_type': test_type,
                            'statistic': statistic,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'cohens_d': cohens_d,
                            'effect_size': self._interpret_effect_size(abs(cohens_d)),
                            'group1_mean': mean1,
                            'group2_mean': mean2,
                            'group1_std': std1,
                            'group2_std': std2
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Error en test estadístico para {feature}: {e}")
                        continue
                
                # Corrección por múltiples comparaciones (Bonferroni)
                p_values = [test['p_value'] for test in statistical_tests]
                from statsmodels.stats.multitest import multipletests
                corrected_p = multipletests(p_values, method='bonferroni')[1]
                
                for i, test in enumerate(statistical_tests):
                    test['p_value_corrected'] = corrected_p[i]
                    test['significant_corrected'] = corrected_p[i] < 0.05
                
                results['statistical_tests'] = statistical_tests
                results['n_significant'] = sum(1 for test in statistical_tests if test['significant_corrected'])
                
            # Análisis de componentes principales
            if SKLEARN_AVAILABLE:
                pca = PCA(n_components=min(10, numeric_features.shape[1]), random_state=42)
                _ = pca.fit_transform(StandardScaler().fit_transform(numeric_features))
                
                results['pca'] = {
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                    'components': pca.components_.tolist(),
                    'feature_names': numeric_features.columns.tolist()
                }
            
            self.logger.info("Análisis estadístico completado")
            return results
            
        except Exception as e:
            self.logger.error(f"Error en análisis estadístico: {e}")
            return {}
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpreta el tamaño del efecto según Cohen's d."""
        if cohens_d < 0.2:
            return "pequeño"
        elif cohens_d < 0.5:
            return "mediano"
        elif cohens_d < 0.8:
            return "grande"
        else:
            return "muy grande"
    
    def cluster_analysis(self, features_df: pd.DataFrame, 
                        n_clusters: int = None) -> Dict[str, Any]:
        """
        Realiza análisis de clustering en características radiómicas.
        
        Args:
            features_df: DataFrame con características
            n_clusters: Número de clusters (automático si None)
            
        Returns:
            Resultados del clustering
        """
        if not SKLEARN_AVAILABLE:
            self.logger.error("scikit-learn no está disponible")
            return {}
        
        try:
            # Preparar datos
            numeric_features = features_df.select_dtypes(include=[np.number])
            x_scaled = StandardScaler().fit_transform(numeric_features)
            
            # Determinar número óptimo de clusters si no se especifica
            if n_clusters is None:
                silhouette_scores = []
                k_range = range(2, min(11, len(features_df)))
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    cluster_labels = kmeans.fit_predict(x_scaled)
                    silhouette_avg = silhouette_score(x_scaled, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                
                n_clusters = list(k_range)[int(np.argmax(silhouette_scores))]
            
            # Realizar clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(x_scaled)
            
            # Análisis de clusters
            cluster_stats = []
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_data = numeric_features[cluster_mask]
                
                cluster_stats.append({
                    'cluster_id': i,
                    'size': np.sum(cluster_mask),
                    'centroid': kmeans.cluster_centers_[i].tolist(),
                    'feature_means': cluster_data.mean().to_dict(),
                    'feature_stds': cluster_data.std().to_dict()
                })
            
            results = {
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'cluster_statistics': cluster_stats,
                'silhouette_score': silhouette_score(x_scaled, cluster_labels),
                'inertia': kmeans.inertia_
            }
            
            self.logger.info(f"Análisis de clustering completado con {n_clusters} clusters")
            return results
            
        except Exception as e:
            self.logger.error(f"Error en análisis de clustering: {e}")
            return {}
    
    def generate_feature_report(self, features_df: pd.DataFrame, 
                              labels: np.ndarray = None) -> Dict[str, Any]:
        """
        Genera un reporte completo de análisis radiómico.
        
        Args:
            features_df: DataFrame con características
            labels: Etiquetas para análisis supervisado (opcional)
            
        Returns:
            Reporte completo
        """
        report = {
            'summary': {
                'n_images': len(features_df),
                'n_features': len(features_df.select_dtypes(include=[np.number]).columns),
                'feature_classes': self._categorize_features(features_df.columns.tolist())
            }
        }
        
        try:
            # Estadísticas descriptivas
            numeric_features = features_df.select_dtypes(include=[np.number])
            report['descriptive_stats'] = {
                'mean': numeric_features.mean().to_dict(),
                'std': numeric_features.std().to_dict(),
                'min': numeric_features.min().to_dict(),
                'max': numeric_features.max().to_dict(),
                'median': numeric_features.median().to_dict()
            }
            
            # Análisis de correlaciones
            corr_analysis = self.analyze_feature_correlations(features_df)
            report['correlation_analysis'] = corr_analysis
            
            # Análisis estadístico si hay etiquetas
            if labels is not None:
                stat_analysis = self.perform_statistical_analysis(features_df, labels)
                report['statistical_analysis'] = stat_analysis
            
            # Análisis de clustering
            cluster_analysis = self.cluster_analysis(features_df)
            report['cluster_analysis'] = cluster_analysis
            
            self.logger.info("Reporte radiómico generado exitosamente")
            
        except Exception as e:
            self.logger.error(f"Error generando reporte: {e}")
            report['error'] = str(e)
        
        return report
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categoriza características por tipo."""
        categories = {
            'firstorder': [],
            'glcm': [],
            'glrlm': [],
            'glszm': [],
            'gldm': [],
            'ngtdm': [],
            'shape': [],
            'wavelet': [],
            'log': [],
            'other': []
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            categorized = False
            
            for category in categories.keys():
                if category in feature_lower:
                    categories[category].append(feature)
                    categorized = True
                    break
            
            if not categorized:
                categories['other'].append(feature)
        
        # Filtrar categorías vacías
        return {k: v for k, v in categories.items() if v}
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> bool:
        """
        Guarda resultados del análisis radiómico.
        
        Args:
            results: Diccionario con resultados
            output_path: Ruta de salida
            
        Returns:
            True si se guardó exitosamente
        """
        try:
            # Convertir numpy arrays a listas para serialización JSON
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                return obj
            
            # Crear copia para no modificar original
            results_copy = json.loads(json.dumps(results, default=convert_numpy))
            
            with open(output_path, 'w') as f:
                json.dump(results_copy, f, indent=2)
            
            self.logger.info(f"Resultados guardados en: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando resultados: {e}")
            return False


def load_radiomics_analyzer():
    """Función helper para cargar el analizador radiómico con configuración por defecto."""
    return RadiomicsAnalyzer()


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar analizador
    analyzer = RadiomicsAnalyzer()
    
    print("Analizador radiómico inicializado correctamente")
    print(f"PyRadiomics disponible: {RADIOMICS_AVAILABLE}")
    print(f"SimpleITK disponible: {SITK_AVAILABLE}")
    print(f"SciPy disponible: {SCIPY_AVAILABLE}")
    print(f"scikit-learn disponible: {SKLEARN_AVAILABLE}")
    print(f"Plotting disponible: {PLOTTING_AVAILABLE}")