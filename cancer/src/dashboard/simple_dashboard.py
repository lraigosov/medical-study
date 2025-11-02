"""
Dashboard simplificado para análisis de cáncer
Interfaz web básica y funcional usando Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib y seaborn para máxima legibilidad
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8fafc'
plt.rcParams['axes.edgecolor'] = '#94a3b8'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.color'] = '#cbd5e1'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['text.color'] = '#1e293b'
plt.rcParams['axes.labelcolor'] = '#1e293b'
plt.rcParams['xtick.color'] = '#475569'
plt.rcParams['ytick.color'] = '#475569'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11

# Configurar paths
BASE_DIR = Path(__file__).parent.parent.parent
SRC_DIR = BASE_DIR / "src"
CONFIG_PATH = BASE_DIR / "config" / "config.json"
RESULTS_DIR = BASE_DIR / "results"

# Asegurar que src esté en el path antes de importar
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Importes internos (DI container) - usando loader local
build_container = None
container_error = None

try:
    # Usar el loader local que maneja los imports correctamente
    from dashboard.container_loader import build_container
except ImportError as e:
    container_error = f"Error importando container_loader: {e}\n\nPath actual: {sys.path[:3]}"
except Exception as e:
    container_error = f"Error inesperado: {type(e).__name__}: {e}"

# Configuración de página
st.set_page_config(
    page_title="Cancer Analytics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS optimizado para máxima legibilidad y contraste
st.markdown("""
<style>
    /* Forzar modo claro y fondo blanco */
    .stApp {
        background-color: #ffffff;
    }
    
    .main .block-container {
        background-color: #ffffff;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header principal con mejor contraste */
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(79, 70, 229, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        margin: 0.8rem 0 0 0;
        font-size: 1.2rem;
        font-weight: 500;
        opacity: 0.95;
    }
    
    /* Metric boxes con colores vibrantes y legibles */
    .metric-box {
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
        border: 2px solid #3b82f6;
        padding: 2rem;
        border-radius: 16px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.25);
        border-color: #2563eb;
    }
    
    .metric-box h4 {
        margin: 0 0 0.8rem 0;
        color: #1e3a8a;
        font-size: 1.4rem;
        font-weight: 700;
    }
    
    .metric-box p {
        margin: 0;
        color: #1e40af;
        font-size: 1rem;
        line-height: 1.6;
        font-weight: 500;
    }
    
    /* Sidebar con fondo claro */
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 2px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #1e293b !important;
    }
    
    /* Títulos en sidebar */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #0f172a !important;
        font-weight: 700;
    }
    
    /* Radio buttons más visibles */
    [data-testid="stSidebar"] .row-widget.stRadio > div {
        background-color: white;
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* Botones con mejor contraste */
    .stButton > button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        font-size: 1.05rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.4);
    }
    
    /* Tablas con mejor legibilidad */
    .dataframe {
        font-size: 1rem;
        color: #1e293b !important;
    }
    
    .dataframe th {
        background-color: #3b82f6 !important;
        color: white !important;
        font-weight: 700;
        padding: 12px !important;
    }
    
    .dataframe td {
        background-color: #ffffff !important;
        color: #1e293b !important;
        padding: 10px !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }
    
    /* Títulos principales con mejor contraste */
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    h2 {
        font-size: 2rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
        margin-top: 1.5rem !important;
    }
    
    /* Métricas de Streamlit más grandes y legibles */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #1e3a8a !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }
    
    /* Texto general más legible */
    p, span, div {
        color: #334155 !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Expanders con mejor contraste */
    [data-testid="stExpander"] {
        background-color: white;
        border: 2px solid #cbd5e1;
        border-radius: 8px;
    }
    
    [data-testid="stExpanderDetails"] {
        background-color: #f8fafc;
        color: #1e293b !important;
    }
    
    /* Success/Error/Warning boxes */
    .stSuccess {
        background-color: #dcfce7 !important;
        border-left: 4px solid #16a34a !important;
        color: #166534 !important;
        padding: 1rem !important;
        font-weight: 600 !important;
    }
    
    .stError {
        background-color: #fee2e2 !important;
        border-left: 4px solid #dc2626 !important;
        color: #991b1b !important;
        padding: 1rem !important;
        font-weight: 600 !important;
    }
    
    .stWarning {
        background-color: #fef3c7 !important;
        border-left: 4px solid #f59e0b !important;
        color: #92400e !important;
        padding: 1rem !important;
        font-weight: 600 !important;
    }
    
    .stInfo {
        background-color: #dbeafe !important;
        border-left: 4px solid #3b82f6 !important;
        color: #1e40af !important;
        padding: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Gráficos con mejor fondo */
    .stPlotlyChart, .stPyplot {
        background-color: white !important;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Optimización de rendimiento */
    img {
        loading: lazy;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🔬 Cancer Analytics Platform</h1>
    <p>Análisis de Datos de Cáncer con Inteligencia Artificial</p>
</div>
""", unsafe_allow_html=True)

# Funciones de utilidad
@st.cache_data
def load_config():
    """Cargar configuración del proyecto"""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            st.warning(f"⚠️ Archivo de configuración no encontrado: {CONFIG_PATH}")
            return {}
    except Exception as e:
        st.error(f"❌ Error cargando configuración: {e}")
        return {}

def check_dependencies():
    """Verificar dependencias instaladas (cacheado para performance)"""
    dependencies = {
        'pandas': False,
        'numpy': False,
        'matplotlib': False,
        'streamlit': True,  # Ya funciona si estamos aquí
        'tensorflow': False,
        'sklearn': False
    }
    
    for lib in dependencies:
        if lib == 'streamlit':
            continue
        try:
            __import__(lib)
            dependencies[lib] = True
        except ImportError:
            dependencies[lib] = False
    
    return dependencies

@st.cache_data(ttl=300)  # Cache por 5 minutos
def get_project_stats():
    """Obtener estadísticas del proyecto (optimizado)"""
    result_files = 0
    if RESULTS_DIR.exists():
        result_files = len(list(RESULTS_DIR.glob("*.json")))
    
    return {
        'result_files': result_files,
        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@st.cache_data
def generate_sample_data():
    """Generar datos de muestra (cacheado para evitar regeneración)"""
    np.random.seed(42)
    
    cancer_types = ['Pulmón', 'Mama', 'Próstata', 'Colon', 'Melanoma']
    sample_data = []
    
    for cancer_type in cancer_types:
        benign = np.random.randint(20, 100)
        malignant = np.random.randint(10, 80)
        sample_data.append({
            'Tipo': cancer_type,
            'Benigno': benign,
            'Maligno': malignant,
            'Total': benign + malignant
        })
    
    return pd.DataFrame(sample_data), cancer_types

@st.cache_data
def generate_model_metrics():
    """Generar métricas de modelos (cacheado)"""
    model_metrics = {
        'Modelo': ['ResNet50', 'EfficientNet', 'Vision Transformer', 'Híbrido'],
        'Accuracy': [0.892, 0.905, 0.878, 0.912],
        'Precision': [0.885, 0.898, 0.871, 0.908],
        'Recall': [0.899, 0.912, 0.885, 0.916],
        'F1-Score': [0.892, 0.905, 0.878, 0.912]
    }
    
    return pd.DataFrame(model_metrics)

# Sidebar con mejor estructura
st.sidebar.title("🎛️ Panel de Control")
st.sidebar.markdown("---")

# Cargar configuración
config = load_config()
dependencies = check_dependencies()
project_stats = get_project_stats()

# Estado del sistema en sidebar
with st.sidebar.expander("📊 Estado del Sistema", expanded=False):
    for dep, status in dependencies.items():
        status_icon = "✅" if status else "❌"
        st.markdown(f"{status_icon} {dep}")
    
    # Gemini status
    gemini_configured = bool(config.get('gemini', {}).get('api_key'))
    gemini_icon = "✅" if gemini_configured else "❌"
    st.markdown(f"{gemini_icon} Gemini AI")

st.sidebar.markdown("---")

# Navegación mejorada
st.sidebar.markdown("### 📍 Navegación")
page = st.sidebar.radio(
    "Ir a:",
    ["🏠 Inicio", "🖼️ Análisis", "📊 Datos", "🤖 Modelos", "⚙️ Config"],
    label_visibility="collapsed"
)

# Página: Inicio
if page == "🏠 Inicio":
    st.header("🏠 Bienvenido")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h4>🔬 Análisis Avanzado</h4>
            <p>Machine Learning y Deep Learning para análisis médico</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h4>🤖 IA Generativa</h4>
            <p>Integración con Gemini AI para análisis cualitativo</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h4>📈 Radiomics</h4>
            <p>Extracción de características cuantitativas</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Estadísticas del proyecto (optimizadas)
    st.subheader("📊 Estadísticas del Proyecto")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dependencias OK", f"{sum(dependencies.values())}/{len(dependencies)}", 
                  delta=None, delta_color="off")
    
    with col2:
        config_items = len(config) if config else 0
        st.metric("Configuraciones", config_items, delta=None, delta_color="off")
    
    with col3:
        st.metric("Archivos Resultado", project_stats['result_files'], 
                  delta=None, delta_color="off")
    
    with col4:
        gemini_configured = bool(config.get('gemini', {}).get('api_key'))
        st.metric("Estado Gemini", "✅ OK" if gemini_configured else "❌ No Config",
                  delta=None, delta_color="off")
    
    # Información del proyecto
    st.subheader("📋 Información del Proyecto")
    
    project_info = {
        "📁 Nombre": "Cancer Analytics Platform",
        "🔢 Versión": "1.0.0",
        "🎯 Objetivo": "Análisis de cáncer con IA para diagnóstico temprano",
        "🗃️ Fuente de Datos": "TCIA (The Cancer Imaging Archive)",
        "🛠️ Tecnologías": "Python, TensorFlow, Streamlit, Gemini AI",
        "📅 Última Actualización": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    
    for key, value in project_info.items():
        st.write(f"**{key}**: {value}")

# Página: Datos
elif page == "📊 Datos":
    st.header("📊 Análisis de Datos")
    
    if config.get('data'):
        st.subheader("🗂️ Colecciones TCIA Configuradas")
        
        collections = config['data'].get('target_collections', [])
        if collections:
            # Crear tabla de colecciones
            collection_data = []
            for col in collections:
                collection_data.append({
                    'Colección': col,
                    'Estado': '✅ Configurado',
                    'Tipo': 'Cáncer' if 'CMB' in col else 'Especializado'
                })
            
            df_collections = pd.DataFrame(collection_data)
            st.dataframe(df_collections, width='stretch')
            
            # Gráfico de distribución
            col_counts = df_collections['Tipo'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 7))
            colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b']
            col_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors,
                           textprops={'fontsize': 14, 'weight': 'bold', 'color': 'white'})
            ax.set_title('Distribución de Tipos de Colecciones', 
                        fontsize=18, fontweight='bold', color='#1e293b', pad=20)
            ax.set_ylabel('')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No hay colecciones configuradas")
    
    # Simulación de datos para demostración (optimizado con caché)
    st.subheader("📈 Datos Simulados (Demo)")
    
    # Generar datos de ejemplo (cacheados)
    df_sample, cancer_types = generate_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(df_sample, width='stretch')
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8fafc')
        
        x = range(len(cancer_types))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], df_sample['Benigno'], width, 
                      label='Benigno', color='#10b981', edgecolor='#065f46', linewidth=2)
        bars2 = ax.bar([i + width/2 for i in x], df_sample['Maligno'], width, 
                      label='Maligno', color='#ef4444', edgecolor='#991b1b', linewidth=2)
        
        ax.set_xlabel('Tipo de Cáncer', fontsize=14, fontweight='bold', color='#1e293b')
        ax.set_ylabel('Número de Casos', fontsize=14, fontweight='bold', color='#1e293b')
        ax.set_title('Distribución Benigno vs Maligno', 
                    fontsize=18, fontweight='bold', color='#1e293b', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(cancer_types, rotation=45, ha='right', 
                          fontsize=12, fontweight='600', color='#334155')
        ax.tick_params(axis='y', labelsize=12, labelcolor='#334155')
        ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
        ax.grid(True, alpha=0.2, linestyle='--', color='#94a3b8')
        
        # Agregar valores sobre las barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10, 
                       fontweight='bold', color='#1e293b')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# Página: Modelos
elif page == "🤖 Modelos":
    st.header("🤖 Modelos de IA")
    
    # Estado de TensorFlow
    tensorflow_available = dependencies.get('tensorflow', False)
    sklearn_available = dependencies.get('sklearn', False)
    
    if tensorflow_available:
        st.success("✅ TensorFlow disponible - Modelos de Deep Learning habilitados")
    else:
        st.error("❌ TensorFlow no disponible")
        st.info("Instalar con: `pip install tensorflow`")
    
    if sklearn_available:
        st.success("✅ Scikit-learn disponible - ML tradicional habilitado")
    else:
        st.error("❌ Scikit-learn no disponible")
        st.info("Instalar con: `pip install scikit-learn`")
    
    # Configuración de modelos desde config
    if config.get('model'):
        st.subheader("⚙️ Configuración de Modelos")
        
        model_config = config['model']
        
        if 'early_detection' in model_config:
            early_config = model_config['early_detection']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Detección Temprana**")
                st.write(f"• Arquitectura: {early_config.get('architecture', 'N/A')}")
                st.write(f"• Input Shape: {early_config.get('input_shape', 'N/A')}")
                st.write(f"• Clases: {early_config.get('num_classes', 'N/A')}")
                st.write(f"• Learning Rate: {early_config.get('learning_rate', 'N/A')}")
                st.write(f"• Épocas: {early_config.get('epochs', 'N/A')}")
            
            with col2:
                if 'multiclass_detection' in model_config:
                    multi_config = model_config['multiclass_detection']
                    st.markdown("**🔬 Detección Multiclase**")
                    st.write(f"• Arquitectura: {multi_config.get('architecture', 'N/A')}")
                    st.write(f"• Input Shape: {multi_config.get('input_shape', 'N/A')}")
                    st.write(f"• Clases: {multi_config.get('num_classes', 'N/A')}")
                    st.write(f"• Learning Rate: {multi_config.get('learning_rate', 'N/A')}")
                    st.write(f"• Épocas: {multi_config.get('epochs', 'N/A')}")
    
    # Simulación de métricas de modelo (optimizado con caché)
    st.subheader("📊 Métricas de Modelos (Simuladas)")
    
    df_metrics = generate_model_metrics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(df_metrics, width='stretch')
    
    with col2:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8fafc')
        
        x = np.arange(len(df_metrics['Modelo']))
        width = 0.2
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
        edge_colors = ['#1e40af', '#065f46', '#92400e', '#5b21b6']
        
        for i, metric in enumerate(metrics):
            bars = ax.bar(x + i*width, df_metrics[metric], width, 
                         label=metric, color=colors[i], 
                         edgecolor=edge_colors[i], linewidth=2)
            
            # Agregar valores sobre las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, 
                       fontweight='bold', color='#1e293b')
        
        ax.set_xlabel('Modelos', fontsize=14, fontweight='bold', color='#1e293b')
        ax.set_ylabel('Valor de Métrica', fontsize=14, fontweight='bold', color='#1e293b')
        ax.set_title('Comparación de Métricas por Modelo', 
                    fontsize=18, fontweight='bold', color='#1e293b', pad=20)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df_metrics['Modelo'], rotation=30, ha='right',
                          fontsize=12, fontweight='600', color='#334155')
        ax.tick_params(axis='y', labelsize=12, labelcolor='#334155')
        ax.legend(fontsize=11, loc='upper left', framealpha=0.95, ncol=2)
        ax.grid(True, alpha=0.2, linestyle='--', axis='y', color='#94a3b8')
        ax.set_ylim(0.85, 0.95)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # Mejor modelo (corregido)
    best_model_idx = int(df_metrics['Accuracy'].idxmax())
    best_model = df_metrics.iloc[best_model_idx]
    
    st.success(f"🏆 **Mejor Modelo**: {best_model['Modelo']} (Accuracy: {best_model['Accuracy']:.3f})")

# Página: Configuración
elif page == "⚙️ Config":
    st.header("⚙️ Configuración del Sistema")
    
    # Mostrar configuración actual
    st.subheader("📋 Configuración Actual")
    
    if config:
        # Ocultar API key por seguridad
        config_display = config.copy()
        if 'gemini' in config_display and 'api_key' in config_display['gemini']:
            api_key = config_display['gemini']['api_key']
            config_display['gemini']['api_key'] = f"{api_key[:8]}***{api_key[-4:]}" if len(api_key) > 12 else "***"
        
        st.json(config_display)
    else:
        st.warning("No se encontró configuración")
    
    # Editor simple
    st.subheader("✏️ Configuración Básica")
    
    with st.form("config_form"):
        st.markdown("**🧠 Gemini AI**")
        gemini_key = st.text_input(
            "API Key de Gemini:",
            value="",
            type="password",
            help="Introduce tu API Key de Google Gemini"
        )
        
        gemini_model = st.selectbox(
            "Modelo Gemini:",
            ['gemini-2.5-flash', 'gemini-pro', 'gemini-pro-vision'],
            index=0
        )
        
        st.markdown("**🤖 Modelos**")
        epochs = st.number_input("Épocas de entrenamiento:", min_value=1, max_value=200, value=100)
        batch_size = st.number_input("Batch size:", min_value=1, max_value=128, value=32)
        learning_rate = st.number_input("Learning rate:", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
        
        if st.form_submit_button("💾 Guardar Configuración"):
            if gemini_key:
                new_config = config.copy() if config else {}
                
                new_config['gemini'] = {
                    'api_key': gemini_key,
                    'model': gemini_model,
                    'temperature': 0.1,
                    'max_tokens': 4096
                }
                
                if 'model' not in new_config:
                    new_config['model'] = {}
                
                new_config['model']['early_detection'] = {
                    'epochs': int(epochs),
                    'batch_size': int(batch_size),
                    'learning_rate': float(learning_rate)
                }
                
                try:
                    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(CONFIG_PATH, 'w') as f:
                        json.dump(new_config, f, indent=2)
                    
                    st.success("✅ Configuración guardada exitosamente")
                    st.info("🔄 Reinicia la aplicación para aplicar cambios")
                except Exception as e:
                    st.error(f"❌ Error guardando configuración: {e}")
            else:
                st.warning("⚠️ Introduce al menos la API Key de Gemini")
    
    # Información del sistema
    st.subheader("💻 Información del Sistema")
    
    system_info = {
        "Python": sys.version.split()[0],
        "Streamlit": st.__version__,
        "Directorio Base": str(BASE_DIR),
        "Archivo Config": str(CONFIG_PATH),
        "Config Existe": "✅" if CONFIG_PATH.exists() else "❌"
    }
    
    for key, value in system_info.items():
        st.write(f"**{key}**: `{value}`")

# Página: Análisis con IA (contenedor hexagonal)
elif page == "🖼️ Análisis":
    st.header("🖼️ Análisis con IA (Gemini)")

    # Verificar contenedor de dependencias
    if build_container is None:
        st.error("❌ No se pudo importar el contenedor de dependencias.")
        if container_error:
            with st.expander("🔍 Ver detalles del error"):
                st.code(container_error)
        st.info("💡 **Posible solución**: Verifica que el archivo `src/infrastructure/container.py` exista y que todas las dependencias estén instaladas.")
        
        # Información de debugging
        with st.expander("🛠️ Información de debugging"):
            st.write(f"**BASE_DIR**: `{BASE_DIR}`")
            st.write(f"**SRC_DIR**: `{SRC_DIR}`")
            st.write(f"**sys.path[0]**: `{sys.path[0]}`")
            st.write(f"**CONFIG_PATH existe**: {CONFIG_PATH.exists()}")
            
            container_path = SRC_DIR / "infrastructure" / "container.py"
            st.write(f"**container.py existe**: {container_path.exists()}")
        st.stop()

    # Verificar configuración de Gemini
    if not gemini_configured:
        st.warning("⚠️ Gemini AI no está configurado.")
        st.info("Ve a la página **⚙️ Config** y configura tu API Key de Gemini para habilitar el análisis.")
        st.stop()

    # Bloque de configuración visible
    with st.expander("⚙️ Configuración actual", expanded=False):
        st.write(f"✅ Gemini configurado: **{config.get('gemini', {}).get('model', 'N/A')}**")
        st.write(f"📁 Archivo de configuración: `{CONFIG_PATH}`")

    # Uploader de archivos
    uploaded_file = st.file_uploader(
        "📤 Sube una imagen médica (PNG/JPG/JPEG)", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=False,
        help="Formatos soportados: PNG, JPG, JPEG"
    )

    analysis_type = st.selectbox(
        "🔬 Tipo de análisis",
        ["general", "cancer_detection", "radiomics"],
        index=0,
        help="Selecciona el tipo de análisis a realizar"
    )

    # Directorio temporal para cargas
    uploads_dir = BASE_DIR / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    run_btn = st.button("🚀 Analizar Imagen", type="primary")

    if run_btn:
        if not uploaded_file:
            st.warning("⚠️ Por favor, sube una imagen primero.")
            st.stop()

        # Guardar archivo temporalmente
        temp_path = uploads_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as e:
            st.error(f"❌ No se pudo guardar el archivo: {e}")
            st.stop()

        # Ejecutar análisis
        with st.spinner("🔄 Analizando imagen con IA..."):
            try:
                container = build_container(str(CONFIG_PATH))
                svc = container.analysis_service
                result = svc.analyze_image(str(temp_path), analysis_type)
            except Exception as e:
                st.error(f"❌ Error durante el análisis: {e}")
                import traceback
                with st.expander("🔍 Ver traceback completo"):
                    st.code(traceback.format_exc())
                st.stop()

        # Mostrar resultados
        st.success("✅ ¡Análisis completado!")
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("📷 Imagen Analizada")
            try:
                st.image(str(temp_path), caption=uploaded_file.name, width='stretch')
            except Exception:
                st.info("ℹ️ Vista previa no disponible para este archivo.")

        with col_right:
            st.subheader("🤖 Resultado del Análisis")
            if isinstance(result, dict) and "error" in result:
                st.error(f"❌ {result.get('error')}")
            else:
                # Texto principal
                response_text = (
                    result.get("gemini_response")
                    or result.get("response_text")
                    or result.get("text")
                    or "⚠️ Sin respuesta de IA"
                )
                st.markdown(response_text)

                # Hallazgos estructurados
                findings = result.get("findings") or []
                recommendations = result.get("recommendations") or []
                confidence = result.get("confidence_indicators") or []

                if findings:
                    st.markdown("**🔍 Hallazgos:**")
                    for item in findings:
                        st.write(f"• {item}")
                
                if recommendations:
                    st.markdown("**💡 Recomendaciones:**")
                    for item in recommendations:
                        st.write(f"• {item}")
                
                if confidence:
                    st.markdown("**📊 Indicadores de confianza:**")
                    st.write(", ".join(confidence))

                # Descargo legal
                disclaimer = (config.get("legal", {}) or {}).get("report_disclaimer")
                if disclaimer:
                    st.info(f"ℹ️ **Nota Legal**: {disclaimer}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🔬 Cancer Analytics Platform v1.0.0</p>
    <p>Última actualización: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</div>
""", unsafe_allow_html=True)
