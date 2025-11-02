# Arquitectura Hexagonal - Cancer Analytics Platform

## Visión General

Este proyecto implementa una arquitectura hexagonal (también conocida como "puertos y adaptadores") para separar la lógica de negocio de las implementaciones técnicas externas. Esta arquitectura permite:

- **Independencia de frameworks y bibliotecas externas**
- **Testabilidad**: fácil mock de dependencias
- **Flexibilidad**: cambiar implementaciones sin afectar el dominio
- **Claridad**: separación explícita entre capas

---

## Estructura de Capas

```
cancer/src/
├── domain/                # Entidades y value objects del negocio
│   └── entities.py       # ROI, PatientInfo, AnalysisResult
├── ports/                 # Interfaces (contratos) que definen capacidades
│   ├── genai_port.py     # Análisis con IA generativa
│   ├── tcia_port.py      # Acceso a TCIA
│   └── dicom_port.py     # Procesamiento DICOM
├── application/           # Casos de uso / servicios de aplicación
│   └── services/
│       └── analysis_service.py  # Orquestador de análisis
├── infrastructure/        # Implementaciones técnicas concretas
│   ├── adapters/
│   │   ├── genai_adapter.py   # Implementa genai_port con GeminiAnalyzer
│   │   └── tcia_adapter.py    # Implementa tcia_port con TCIAClient
│   └── container.py       # DI: construye adaptadores y servicios
├── utils/                 # Código legacy reutilizado (gemini_analyzer, tcia_client, etc.)
├── cli/                   # Interfaz de línea de comandos
│   └── analyze.py
└── dashboard/             # Interfaz web (Streamlit)
    └── simple_dashboard.py
```

---

## Capas Explicadas

### 1. **Dominio** (`domain/`)

Contiene las entidades fundamentales del negocio, sin dependencias externas:

- `ROI`: Región de interés en una imagen
- `PatientInfo`: Información del paciente
- `AnalysisResult`: Resultado de un análisis

Estas clases son **simples dataclasses** que representan conceptos del dominio.

### 2. **Puertos** (`ports/`)

Define **contratos (interfaces)** como `Protocol` de Python. Los puertos especifican **qué** se necesita, no **cómo** se implementa:

- `GenAIAnalyzerPort`: contrato para análisis con IA generativa
- `TciaPort`: contrato para acceso a fuentes de imágenes (TCIA)
- `DicomPort`: contrato para procesamiento DICOM

Estos puertos son **typing.Protocol** que permiten duck typing y validación estática sin acoplamiento.

### 3. **Aplicación** (`application/`)

Contiene la lógica de casos de uso. Orquesta los puertos para resolver requisitos funcionales:

- `AnalysisService`: servicio que coordina el análisis de imágenes usando `GenAIAnalyzerPort`
  - `analyze_image(image_path, analysis_type)`: análisis individual
  - `analyze_batch(image_paths, analysis_type)`: análisis en lote
  - `compare(image1, image2, comparison_type)`: comparación temporal

**No depende de implementaciones concretas**, solo de puertos.

### 4. **Infraestructura** (`infrastructure/`)

Contiene **adaptadores** que implementan los puertos usando tecnologías específicas:

- `GenAIGeminiAdapter`: implementa `GenAIAnalyzerPort` delegando a `GeminiAnalyzer` (de `utils/`)
- `TciaAdapter`: implementa `TciaPort` delegando a `TCIAClient` (de `utils/`)

También incluye el **contenedor de DI** (`container.py`):

- Lee `config.json`
- Construye adaptadores con la configuración necesaria
- Expone servicios de aplicación listos para usar

### 5. **Utilidades** (`utils/`)

Código "legacy" o reutilizable que implementa funcionalidad concreta:

- `gemini_analyzer.py`: cliente de Gemini AI
- `tcia_client.py`: cliente REST de TCIA
- `dicom_processor.py`: procesamiento de imágenes DICOM
- `config_loader.py`: carga de configuración centralizada

Los **adaptadores** envuelven estos utils para implementar puertos, evitando duplicación de código.

### 6. **Interfaces de Usuario** (`cli/`, `dashboard/`)

Capas externas que consumen servicios de aplicación:

- **CLI** (`cli/analyze.py`): punto de entrada por terminal
  ```bash
  python -m src.cli.analyze imagen.jpg --type cancer_detection
  ```
  
- **Dashboard** (`dashboard/simple_dashboard.py`): interfaz web con Streamlit
  - Página "Análisis": sube imagen → ejecuta `AnalysisService` → muestra resultado con disclaimer

---

## Flujo de Ejecución

### Ejemplo: Análisis de una imagen desde el dashboard

1. **Usuario** sube imagen en `🖼️ Análisis` (Streamlit UI)
2. **Dashboard** invoca:
   ```python
   container = build_container(config_path)
   svc = container.analysis_service
   result = svc.analyze_image(image_path, "cancer_detection")
   ```
3. **Container** (`infrastructure/container.py`):
   - Lee `config.json`
   - Construye `GenAIGeminiAdapter` (adaptador)
   - Construye `AnalysisService` inyectando el adaptador
4. **AnalysisService** (`application/services/analysis_service.py`):
   - Recibe llamada `analyze_image(...)`
   - Invoca `self._genai.analyze_medical_image(...)` (puerto)
5. **GenAIGeminiAdapter** (`infrastructure/adapters/genai_adapter.py`):
   - Implementa el puerto
   - Delega a `GeminiAnalyzer` (de `utils/`)
6. **GeminiAnalyzer** (`utils/gemini_analyzer.py`):
   - Llama a API de Gemini
   - Aplica retry/backoff desde config
   - Añade disclaimer legal
   - Retorna resultado estructurado
7. **Resultado** sube por las capas hasta el dashboard, que lo renderiza

---

## Beneficios

### ✅ Testabilidad

Crear mocks de puertos es trivial. Ejemplo:

```python
class MockGenAIPort:
    def analyze_medical_image(self, image_path, analysis_type):
        return {"response_text": "Mock result"}

service = AnalysisService(genai=MockGenAIPort())
result = service.analyze_image("test.jpg", "general")
assert result["response_text"] == "Mock result"
```

### ✅ Cambio de proveedor sin tocar lógica de negocio

Para cambiar de Gemini a otro LLM:

1. Crear `OpenAIAdapter` que implemente `GenAIAnalyzerPort`
2. Actualizar `container.py` para construir `OpenAIAdapter` en lugar de `GenAIGeminiAdapter`
3. **`AnalysisService` no cambia**: sigue usando el puerto

### ✅ Independencia de configuración

La configuración (`config.json`) se inyecta en el container, no hardcodeada en servicios.

### ✅ Reutilización sin duplicación

Los adaptadores **delegan** a `utils/` existentes, evitando copiar código.

---

## Tests

Los tests unitarios verifican cada capa:

- **`tests/test_domain.py`**: entidades (`ROI`, `PatientInfo`, `AnalysisResult`)
- **`tests/test_analysis_service.py`**: servicio de aplicación con mocks
- **`tests/test_adapters.py`**: importación y construcción del container

Ejecutar todos los tests:

```bash
cd cancer
python -m pytest tests/ -v
```

---

## Cómo Extender

### Añadir un nuevo puerto (ej. para PHI anonymization)

1. Crear `src/ports/anonymizer_port.py`:
   ```python
   from typing import Protocol

   class AnonymizerPort(Protocol):
       def anonymize_dicom(self, dicom_path: str) -> str:
           ...
   ```

2. Crear adaptador `src/infrastructure/adapters/anonymizer_adapter.py`:
   ```python
   from ...ports.anonymizer_port import AnonymizerPort
   from ...utils.phi_remover import PHIRemover  # hipotético util

   class AnonymizerAdapter(AnonymizerPort):
       def __init__(self):
           self._impl = PHIRemover()

       def anonymize_dicom(self, dicom_path: str) -> str:
           return self._impl.remove_phi(dicom_path)
   ```

3. Registrar en `container.py`:
   ```python
   self.anonymizer = AnonymizerAdapter()
   ```

4. Usar en servicio de aplicación:
   ```python
   class AnalysisService:
       def __init__(self, genai: GenAIAnalyzerPort, anonymizer: AnonymizerPort):
           self._genai = genai
           self._anonymizer = anonymizer

       def analyze_image(self, image_path: str, analysis_type: str):
           anon_path = self._anonymizer.anonymize_dicom(image_path)
           return self._genai.analyze_medical_image(anon_path, analysis_type)
   ```

### Añadir un nuevo caso de uso

Crear nuevo servicio en `application/services/`:

```python
from ...ports.tcia_port import TciaPort

class DataIngestionService:
    def __init__(self, tcia: TciaPort):
        self._tcia = tcia

    def ingest_collection(self, collection_name: str):
        # lógica de caso de uso
        series_list = self._tcia.get_series(collection_name)
        # ... procesar
```

Registrar en `container.py`:

```python
self.data_ingestion_service = DataIngestionService(self.tcia)
```

---

## Principios de Diseño

1. **Dependency Inversion**: las capas internas (dominio, aplicación) **no dependen** de capas externas (infraestructura, UI). Las dependencias apuntan hacia adentro.

2. **Separation of Concerns**: cada capa tiene una responsabilidad clara:
   - **Dominio**: conceptos del negocio
   - **Puertos**: contratos
   - **Aplicación**: orquestación de casos de uso
   - **Infraestructura**: implementaciones técnicas
   - **UI**: presentación

3. **Single Source of Truth**: la configuración vive en `config/config.json` y se inyecta vía DI, no hardcodeada.

4. **Open/Closed**: abierto a extensión (nuevos adaptadores), cerrado a modificación (servicios de aplicación estables).

---

## Referencias

- [Hexagonal Architecture (Ports & Adapters)](https://alistair.cockburn.us/hexagonal-architecture/)
- [Clean Architecture (Uncle Bob)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Dependency Inversion Principle](https://en.wikipedia.org/wiki/Dependency_inversion_principle)

---

## Estado Actual

✅ Puertos definidos: `genai_port`, `tcia_port`, `dicom_port`  
✅ Adaptadores implementados: `GenAIGeminiAdapter`, `TciaAdapter`  
✅ Servicio de aplicación: `AnalysisService`  
✅ DI Container: `build_container()`  
✅ UI: Dashboard con página "Análisis", CLI `analyze.py`  
✅ Tests unitarios: 9 tests pasando  

🚧 **Pendiente** (según plan_proyecto.md):  
- Puertos y adaptadores para PHI anonymization  
- Servicio de auditoría/trazabilidad  
- Servicio de ingesta y curación (data-ingestor, data-curator)  
- Servicio de feature engineering multimodal  
- Servicio MLOps (trainer, model registry)  

---

**Fin del documento.**
