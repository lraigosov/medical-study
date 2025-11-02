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

# Configurar paths
BASE_DIR = Path(__file__).parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.json"
RESULTS_DIR = BASE_DIR / "results"

# Configuración de página
st.set_page_config(
    page_title="Cancer Analytics",
    page_icon="🔬",
    layout="wide"
)

# CSS básico
st.markdown("""
<style>
    .main-header {
        background-color: #4CAF50;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
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
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error cargando configuración: {e}")
        return {}

def check_dependencies():
    """Verificar dependencias instaladas"""
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

# Sidebar
st.sidebar.title("🎛️ Panel de Control")

# Cargar configuración
config = load_config()
dependencies = check_dependencies()

# Estado del sistema
st.sidebar.markdown("### 📊 Estado del Sistema")
for dep, status in dependencies.items():
    status_icon = "✅" if status else "❌"
    st.sidebar.markdown(f"{status_icon} {dep}")

# Gemini status
gemini_configured = bool(config.get('gemini', {}).get('api_key'))
gemini_icon = "✅" if gemini_configured else "❌"
st.sidebar.markdown(f"{gemini_icon} Gemini AI")

# Navegación
page = st.sidebar.selectbox(
    "Seleccionar Página:",
    ["🏠 Inicio", "📊 Datos", "🤖 Modelos", "⚙️ Config"]
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
    
    # Estadísticas del proyecto
    st.subheader("📊 Estadísticas del Proyecto")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dependencias OK", f"{sum(dependencies.values())}/{len(dependencies)}")
    
    with col2:
        config_items = len(config) if config else 0
        st.metric("Configuraciones", config_items)
    
    with col3:
        result_files = len(list(RESULTS_DIR.glob("*.json"))) if RESULTS_DIR.exists() else 0
        st.metric("Archivos Resultado", result_files)
    
    with col4:
        st.metric("Estado Gemini", "✅ OK" if gemini_configured else "❌ No Config")
    
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
            st.dataframe(df_collections, use_container_width=True)
            
            # Gráfico de distribución
            col_counts = df_collections['Tipo'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            col_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_title('Distribución de Tipos de Colecciones')
            st.pyplot(fig)
        else:
            st.warning("No hay colecciones configuradas")
    
    # Simulación de datos para demostración
    st.subheader("📈 Datos Simulados (Demo)")
    
    # Generar datos de ejemplo
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
    
    df_sample = pd.DataFrame(sample_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(df_sample, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        x = range(len(cancer_types))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], df_sample['Benigno'], width, label='Benigno', color='lightblue')
        ax.bar([i + width/2 for i in x], df_sample['Maligno'], width, label='Maligno', color='salmon')
        
        ax.set_xlabel('Tipo de Cáncer')
        ax.set_ylabel('Número de Casos')
        ax.set_title('Distribución Benigno vs Maligno')
        ax.set_xticks(x)
        ax.set_xticklabels(cancer_types, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)

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
    
    # Simulación de métricas de modelo
    st.subheader("📊 Métricas de Modelos (Simuladas)")
    
    model_metrics = {
        'Modelo': ['ResNet50', 'EfficientNet', 'Vision Transformer', 'Híbrido'],
        'Accuracy': [0.892, 0.905, 0.878, 0.912],
        'Precision': [0.885, 0.898, 0.871, 0.908],
        'Recall': [0.899, 0.912, 0.885, 0.916],
        'F1-Score': [0.892, 0.905, 0.878, 0.912]
    }
    
    df_metrics = pd.DataFrame(model_metrics)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(df_metrics, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(df_metrics['Modelo']))
        width = 0.2
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, df_metrics[metric], width, label=metric, color=colors[i])
        
        ax.set_xlabel('Modelos')
        ax.set_ylabel('Valor de Métrica')
        ax.set_title('Comparación de Métricas por Modelo')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df_metrics['Modelo'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Mejor modelo
    best_model_idx = df_metrics['Accuracy'].idxmax()
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

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🔬 Cancer Analytics Platform v1.0.0</p>
    <p>Última actualización: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</div>
""", unsafe_allow_html=True)