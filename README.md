# Medical Study Repository

## 🏥 Repositorio de Investigación Médica con IA

Este repositorio contiene la plataforma principal `cancer/` orientada a análisis de imágenes médicas y flujos de preparación/entrenamiento.

---

## 📂 Estructura del Repositorio

```
medical-study/
├── cancer/                     # Plataforma de Análisis de Cáncer
│   ├── docs/                   # Documentación técnica
│   ├── src/                    # Código fuente
│   ├── data/                   # Datasets médicos
│   ├── notebooks/              # Análisis exploratorio
│   ├── tests/                  # Tests unitarios
│   └── README.md              # Punto de entrada del proyecto
│
├── [futuros-proyectos]/       # Próximos proyectos médicos
│
└── README.md                  # Este archivo (índice general)
```

---

## 🚀 Proyecto Actual

### 🔬 [Cancer Analytics Platform](./cancer/)

Componentes confirmados (según código existente):
- Arquitectura hexagonal (puertos/adaptadores/servicios) en `cancer/src`
- Cliente TCIA (`src/utils/tcia_client.py`) y procesador DICOM (`src/utils/dicom_processor.py`)
- Integración Gemini (`src/utils/gemini_analyzer.py`) con soporte de dry-run
- Pipelines: `tcia_ingest.py`, `extract_radiomics.py`, `nsclc_prepare.py`, `train_fusion.py`
- Modelos: `src/models/fusion_model.py`, `src/models/cancer_detection.py`
- Dashboard Streamlit: `src/dashboard/simple_dashboard.py`
- Tests unitarios en `cancer/tests/` (actualmente 14 pasando)

Documentación relevante:
- 📖 [README del proyecto](./cancer/README.md)
- 🏗️ [Arquitectura Hexagonal](./cancer/docs/ARCHITECTURE_HEXAGONAL.md)

---

## Alcance

Este índice describe únicamente elementos presentes en el repositorio. No se listan proyectos o funcionalidades no implementadas.

---

## 🛠️ Tecnologías Comunes

### Core Technologies
- **Python 3.8+**: Lenguaje principal
- **TensorFlow/PyTorch**: Deep Learning frameworks
- **Streamlit**: Interfaces web interactivas
- **Docker**: Contenedorización de aplicaciones

### Medical Imaging
- **SimpleITK/PyRadiomics**: Procesamiento de imágenes médicas
- **PyDICOM**: Manejo de archivos DICOM
- **Nibabel**: Procesamiento de neuroimágenes

### AI & Machine Learning
- **Transformers**: Modelos de lenguaje y visión
- **Google Gemini API**: IA generativa
- **Scikit-learn**: ML tradicional
- **XGBoost/LightGBM**: Modelos de gradiente

### Data & Visualization
- **Pandas/NumPy**: Análisis de datos
- **Plotly/Matplotlib**: Visualizaciones
- **Seaborn**: Gráficos estadísticos

---

## 🚀 Inicio Rápido

### Prerrequisitos

- Python 3.8 o superior
- Git
- Entorno virtual (recomendado)
- Claves API según el proyecto (ej: Gemini API)

### Clonar el Repositorio

```bash
# SSH (recomendado)
git clone git@github.com:lraigosov/medical-study.git

# HTTPS
git clone https://github.com/lraigosov/medical-study.git

cd medical-study
```

### Configurar `cancer/`

```powershell
cd cancer
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
# Ejecutar tablero:
streamlit run .\cancer\src\dashboard\simple_dashboard.py
```

---

## Recursos

- **[The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)**

---

## 🔒 Seguridad y Ética

### Principios Fundamentales

1. **Privacidad**: Los datos médicos se manejan con máxima confidencialidad
2. **Anonimización**: Todos los datos personales deben ser anonimizados
3. **Consentimiento**: Solo se usan datos con consentimiento apropiado
4. **Transparencia**: Los modelos y métodos son explicables
5. **Validación**: Los resultados requieren validación clínica

### ⚠️ Advertencia Importante

**Este repositorio es para fines de investigación y educación únicamente.**

- ❌ No usar para diagnóstico médico real sin validación clínica
- ❌ No sustituye la opinión de profesionales médicos
- ❌ Requiere aprobación ética para uso con datos reales de pacientes
- ✅ Diseñado para avanzar la investigación en IA médica
- ✅ Útil para aprendizaje y experimentación académica

---

## 🤝 Contribución

### Cómo Contribuir

1. **Fork** el repositorio
2. **Crea una rama** para tu feature: `git checkout -b feature/nueva-funcionalidad`
3. **Commit** tus cambios: `git commit -m 'Agrega nueva funcionalidad'`
4. **Push** a tu rama: `git push origin feature/nueva-funcionalidad`
5. **Abre un Pull Request**

### Guías de Contribución

- Seguir las convenciones de código Python (PEP 8)
- Incluir docstrings y comentarios apropiados
- Agregar tests para nuevas funcionalidades
- Actualizar documentación según sea necesario
- Respetar las licencias de datasets y librerías

---

## 📄 Licencia

© 2025 **Luis Rai (lraigosov)** - Todos los derechos reservados.

Este repositorio está bajo la **Licencia MIT con Requisito de Atribución**.

**CONDICIONES IMPORTANTES:**
- ✅ Uso libre para investigación científica, académica y comercial
- ✅ Modificaciones y mejoras son bienvenidas y fomentadas
- ⚠️ **OBLIGATORIO**: Mantener créditos al autor original (Luis Rai / lraigosov) en cualquier uso o derivado
- ⚠️ **OBLIGATORIO**: Incluir enlace al repositorio original: https://github.com/lraigosov/medical-study
- ⚠️ **OBLIGATORIO**: Citar como: "Basado en Cancer Analytics Platform por Luis Rai (lraigosov)"

Ver [LICENSE](./LICENSE) para detalles completos.

### 📖 Cómo Citar Este Trabajo

**Para uso académico o investigación:**
```
Cancer Analytics Platform
Autor: Luis Rai (lraigosov)
Año: 2025
Repositorio: https://github.com/lraigosov/medical-study
Licencia: MIT con Atribución Obligatoria
```

**Para uso en aplicaciones o derivados:**
Incluir en la documentación, créditos o "Acerca de":
```
Basado en Cancer Analytics Platform
Desarrollado por: Luis Rai (lraigosov)
https://github.com/lraigosov/medical-study
```

### Licencias de Datasets

Los datasets utilizados pueden tener sus propias licencias. Por favor, revisa y cumple con los términos de uso de cada fuente de datos.

---

## 📞 Contacto y Soporte

### Obtener Ayuda

- **Issues**: Reportar bugs o solicitar features en GitHub Issues
- **Discusiones**: Preguntas y discusiones en GitHub Discussions
- **Wiki**: Documentación extendida en el Wiki del repositorio

### Mantenedor Principal

**Autor y Creador:** Luis Rai (LuisRai)
- **GitHub**: [@lraigosov](https://github.com/lraigosov)
- **Repositorio**: [medical-study](https://github.com/lraigosov/medical-study)
- **Proyecto**: Cancer Analytics Platform

© 2025 Luis Rai - Todos los derechos reservados. El uso de este código requiere atribución al autor original.

---

## 🎓 Referencias y Agradecimientos

### Instituciones y Organizaciones

- The Cancer Imaging Archive (TCIA)
- National Institutes of Health (NIH)
- Google AI for Healthcare
- TensorFlow Medical Imaging Team

### Papers Clave

- [Deep Learning in Medical Imaging](https://www.nature.com/articles/s41746-019-0099-x)
- [Radiomics: Images Are More than Pictures](https://pubmed.ncbi.nlm.nih.gov/26562415/)
- [Artificial Intelligence in Healthcare](https://www.nature.com/articles/s41591-018-0316-z)

### Herramientas y Frameworks

- TensorFlow & Keras
- PyTorch
- SimpleITK
- PyRadiomics
- Streamlit

### Metodología de Documentación

Este proyecto utiliza **IA Generativa como herramienta de apoyo** para optimizar y enriquecer la documentación técnica. Se implementó un proceso de curación riguroso para:

- ✅ **Filtrar alucinaciones**: Validación manual de toda información generada
- ✅ **Verificar coherencia**: Asegurar correspondencia con código y arquitectura reales
- ✅ **Mantener precisión técnica**: Revisión experta de conceptos y terminología
- ✅ **Evitar referencias incorrectas**: Eliminación de conceptos no implementados

**Principio aplicado**: La IA generativa acelera la creación de contenido, pero el criterio humano garantiza la veracidad y relevancia de la documentación final.

---

## Historia y Evolución del Proyecto

### 🕐 Línea de Tiempo

**Agosto 2023** - Inicio del proyecto
- Fase de investigación y conceptualización inicial
- Estudio de arquitecturas y tecnologías disponibles
- Primeros experimentos con modelos de Deep Learning

**2023-2024** - Desarrollo intermitente
- Trabajo en tiempos limitados debido a otros proyectos profesionales
- Múltiples pausas por compromisos laborales y personales
- Evolución orgánica de la arquitectura del sistema
- Implementación de modelos básicos de clasificación

**Finales 2024** - Retoma activa
- Reorganización del código con arquitectura hexagonal profesional
- Integración con The Cancer Imaging Archive (TCIA)
- Consolidación de funcionalidades core
- Mejora de la estructura del proyecto

**Mediados 2025** - 🚀 **Punto de Inflexión**
- **Integración con IA Generativa (Google Gemini AI)**
- Cambio de paradigma: enfoque mucho más realista y práctico
- Salto cualitativo en capacidades de análisis
- Nueva visión del potencial del proyecto

**2025 (Actual)**
- Plataforma `cancer/` operativa con arquitectura hexagonal
- Dashboard Streamlit disponible (`simple_dashboard.py`)
- Integración de pipelines y utilidades confirmadas
- Suite de tests: 14 pasando

### 🔮 Posible Evolución Futura

> **Nota**: Este proyecto se desarrolla en tiempos extracurriculares. Las siguientes ideas representan posibilidades de evolución que requieren planificación adicional:

**Ideas en Consideración**:
- 🤔 Modelos de segmentación avanzada para tumores
- 🤔 API REST para integración externa
- 🤔 Expansión a otros tipos de análisis médico
- 🤔 Mejoras en visualización y reporting
- 🤔 Optimización de rendimiento y escalabilidad

**Proyectos Complementarios Potenciales**:
- 🫀 Análisis cardiovascular
- 🧠 Detección de trastornos neurológicos
- 🦴 Análisis ortopédico
- 🩺 Sistema de apoyo a decisiones clínicas

> La priorización y ejecución de estas ideas dependerá de:
> - Disponibilidad de tiempo del autor
> - Recursos computacionales disponibles
> - Interés y feedback de la comunidad
> - Aparición de nuevas tecnologías relevantes

---

## 📊 Estado Actual del Proyecto

- **Tiempo de Desarrollo**: 2+ años (agosto 2023 - presente)
- **Modalidad**: Desarrollo extracurricular con pausas intermitentes
- **Proyectos Activos**: 1 (Cancer Analytics Platform)
- **Modelos Implementados**: 5+ arquitecturas (CNN, ResNet, EfficientNet, ViT, Swin Transformer)
- **Datasets Integrados**: 6+ colecciones de TCIA
- **Tecnologías**: 15+ frameworks y librerías
- **Tests**: 9 tests unitarios pasando
- **Líneas de Código**: 5,000+ líneas (excl. notebooks)

---

## Uso rápido (Dashboard y CLI)

```powershell
# Dashboard Streamlit
streamlit run .\cancer\src\dashboard\simple_dashboard.py

# CLI de análisis (perfil dev activa dry-run)
python -m src.cli.analyze .\cancer\data\processed\nsclc_small\images\<imagen>.png --type general --profile dev
```

---

**👨‍💻 Creado por:** Luis Rai ([@lraigosov](https://github.com/lraigosov))  
**📅 Última actualización**: Noviembre 2025  
**©️ Copyright**: 2025 Luis Rai - Todos los derechos reservados

---
