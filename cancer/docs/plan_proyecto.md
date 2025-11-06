# Instrucciones maestras para que una IA generativa diseñe y construya una Plataforma de Soporte al Diagnóstico Temprano de Cáncer

> IMPORTANTE: El sistema es **soporte clínico** y análisis adelantado de riesgo. **No reemplaza la evaluación médica ni emite diagnóstico definitivo.**
>
> El sistema debe operar cumpliendo estrictamente regulaciones de privacidad y ética médica. Sólo puede usarse en entornos clínicos o de investigación autorizados.

---

## 0. Objetivo general

Diseñar, implementar y operar una solución integral que:

1. Ingiere y armoniza datos médicos abiertos sobre cáncer (imágenes radiológicas, patología digital, genómica, proteómica, datos clínicos anonimizados y epidemiología poblacional) provenientes de fuentes públicas científicas.
2. Entrena modelos de IA/ML (clásica, deep learning y modelos multimodales) para **detección temprana, estratificación de riesgo y priorización de casos sospechosos** de distintos tipos de cáncer.
3. Expone una **interfaz de usuario muy intuitiva** para perfiles clínicos y de investigación, con explicaciones visuales y reportes estructurados.
4. Ofrece un **backend operable** (pipelines de ingesta, preprocesamiento, entrenamiento, despliegue y auditoría) que pueda ejecutarse automáticamente.
5. Integra una capa de **IA generativa** (ej. Gemini) para: redacción de hallazgos, resúmenes multimodales, generación de reportes preliminares y ayuda interactiva.
6. Implementa una **capa centralizada de configuración y secretos**, incluida la gestión segura de claves como `GEMINI_API_KEY`.

---

## 1. Alcance funcional

### 1.1 Casos de uso principales

* **Soporte de diagnóstico asistido**:

  * El usuario clínico sube imágenes médicas (por ejemplo TAC de pulmón, mamografía, RM cerebral) y opcionalmente datos moleculares/laboratorio asociados.
  * El sistema calcula probabilidad de lesión sospechosa / subtipo tumoral / estadio aproximado y genera una explicación visual (mapas de calor, regiones marcadas).
  * El sistema devuelve un **informe preliminar estructurado** + advertencia legal de que se requiere confirmación de especialista.

* **Priorización temprana / triaje**:

  * Dado un lote de estudios (ej. screening de mama o pulmón), el sistema marca automáticamente los casos potencialmente críticos para revisión prioritaria.

* **Panel de investigación / entrenamiento**:

  * El usuario de investigación puede explorar datasets abiertos armonizados, lanzar nuevos entrenamientos, comparar métricas entre versiones de modelo y auditar sesgos poblacionales.

* **Monitoreo epidemiológico**:

  * Dashboard que muestra incidencias, prevalencias, tasas de supervivencia y distribución geográfica por tipo de cáncer (sin exponer información personal identificable), útil para salud pública y priorización de tamizajes.

### 1.2 Tipos de cáncer iniciales soportados

La primera versión del sistema debe enfocarse en cánceres con fuerte disponibilidad de datos abiertos multimodales:

* Pulmón
* Mama
* Próstata
* Cerebro / gliomas
* Hígado / riñón
* Leucemias y otros cánceres hematológicos (a través de datos de biopsia líquida)
* Cáncer pediátrico (neuroblastoma, leucemias pediátricas) para el caso de fuentes pediátricas armonizadas

Cada tipo debe mapear:

* Modalidades de imagen disponibles (CT, MRI, mamografía, patología digital, etc.)
* Datos ómicos disponibles (genómica somática, proteómica tumoral, biomarcadores en sangre)
* Etiquetas clínicas disponibles (diagnóstico confirmado, estadio, outcome clínico)

---

## 2. Arquitectura lógica de alto nivel

La plataforma debe dividirse en 6 capas principales:

1. **Ingesta y Normalización de Datos**

   * Descarga/ingesta automatizada desde repositorios científicos abiertos.
   * Conversión a formatos internos estandarizados (imágenes médicas en DICOM/NIfTI limpios, genómica en VCF/parquet, clínico-tabular en parquet/Delta, epidemiología en tablas agregadas).
   * Limpieza, control de calidad, eliminación de identificadores personales, enriquecimiento con metadatos.

2. **Data Lake + Catálogo + Metadata Governance**

   * Almacenamiento en bruto (raw zone) sólo para datos públicos ya desidentificados.
   * Zona curada (curated zone) con esquemas uniformes y diccionarios de variables.
   * Catálogo técnico/funcional con linaje de datos, versión de cada dataset y licencia de uso.
   * Registro de procedencia y hash de integridad.

3. **Feature Store Multimodal**

   * Extracción de características (features) para cada modalidad:

     * Radiología / patología digital → embeddings de visión profunda, mapas de calor de regiones tumorales.
     * Genómica / proteómica → mutaciones relevantes, firmas moleculares, rutas alteradas, perfiles proteómicos diferenciales.
     * Clínico-tabular → edad, sexo, estado funcional, estadio clínico, tratamientos previos, outcome.
     * Epidemiología → incidencia/ prevalencia/ sobrevida por cohorte demográfica y región.
   * Normalización de escalas y manejo de datos faltantes.

4. **Motor de Entrenamiento y MLOps**

   * Pipelines reproducibles para: split train/val/test estratificado, entrenamiento supervisado, validación cruzada, cálculo de métricas clínicas (sensibilidad, especificidad, AUC ROC, PPV/NPV por cohorte), interpretabilidad y fairness.
   * Registro en un **Model Registry** con: versión, hiperparámetros, dataset usado, métricas, curva ROC, explicación de interpretabilidad, fecha de aprobación interna.
   * Publicación controlada de modelos listos para inferencia.

5. **Motor de Inferencia Clínica en Producción**

   * Servicio en línea que recibe input del usuario (imagen, panel molecular, variables clínicas) y devuelve predicciones + explicaciones interpretables.
   * Generación de reporte estructurado y resumen clínico en lenguaje natural con ayuda de IA generativa.
   * Registro de auditoría (quién consultó, qué modelo se usó, versión, resultados).
   * Sistema de alertas tempranas para priorización.

6. **Interfaz de Usuario (UI/UX)**

   * Portal clínico (para médicos / radiólogos / oncólogos):

     * Subir estudio.
     * Ver predicción de riesgo / localización de lesión.
     * Ver explicación visual (heatmap, regiones sospechosas).
     * Descargar informe preliminar tipo PDF/HL7 FHIR.
   * Portal de investigación / ciencia de datos:

     * Explorar datasets abiertos armonizados.
     * Lanzar/monitorear nuevos entrenamientos.
     * Comparar versiones de modelos.
     * Analizar sesgos poblacionales y desempeño por subgrupo demográfico.
   * Portal de gobernanza / auditoría:

     * Historial de inferencias.
     * Quién accedió qué dato/modelo.
     * Estado de cumplimiento y licencias de cada dataset.

---

## 3. Flujo de datos extremo a extremo

### 3.1 Pipeline de ingesta y entrenamiento (offline / batch)

1. **Descarga / Acceso a fuentes externas abiertas**

   * Imágenes oncológicas abiertas (ej. TAC pulmón, mamografías, RM tumores cerebrales, patología digital).
   * Datos genómicos/proteómicos tumorales y clínicos asociados (mutaciones somáticas, expresión, outcomes).
   * Datos clínico-epidemiológicos agregados: incidencia, supervivencia, prevalencia.

2. **Estandarización y curación**

   * Conversión a formatos estándar internos.
   * Revisión de integridad (dimensiones de imagen, metadatos obligatorios, rangos válidos).
   * Eliminación/mascaramiento de cualquier rastro de identificación personal en headers de imagen o tablas clínicas (PHI).
   * Normalización de labels (por ejemplo: "cáncer de pulmón de células no pequeñas", "glioblastoma multiforme", "lesión benigna").

3. **Feature Engineering multimodal**

   * Visión: extracción de embeddings y máscaras de lesión.
   * Ómica: selección de mutaciones y firmas moleculares asociadas a progresión / agresividad / respuesta a tratamiento.
   * Clínico-tabular: consolidación de variables demográficas y de estado del paciente.
   * Epidemiológico: agregación de tasas de incidencia y supervivencia por cohorte comparable.

4. **Entrenamiento / Validación**

   * Entrenamiento de modelos unimodales especializados (visión médica, ómica, clínico-tabular).
   * Entrenamiento de modelos multimodales de fusión tardía o atención cruzada (por ejemplo, fusionar embeddings de imagen con marcadores genómicos críticos).
   * Cálculo de métricas clínicas relevantes por tipo de cáncer, estadio, sexo, rango de edad y cohorte poblacional.

5. **Registro del modelo**

   * Guardar pesos, arquitectura, hiperparámetros, dataset exacto utilizado, licencia de los datos, métricas y explicaciones de interpretabilidad.
   * Marcar versión como "Aprobada para Inferencia en Entorno Clínico Controlado" sólo si supera umbrales internos definidos (ej. sensibilidad mínima para screening).

### 3.2 Pipeline de inferencia (tiempo real / near real-time)

1. Usuario clínico carga estudio + metadatos básicos.
2. API de inferencia valida permisos y anonimiza cualquier identificador personal antes de procesar.
3. Motor de inferencia ejecuta el modelo vigente (según tipo de cáncer).
4. Se genera reporte preliminar y se invoca la capa de IA generativa para producir:

   * Resumen clínico legible.
   * Justificación en lenguaje natural.
   * Recomendaciones de siguiente paso (por ejemplo, "sugiere evaluación presencial urgente con oncología"), nunca sustituyendo la decisión médica.
5. Se guarda en auditoría: entrada (hash), salida, versión de modelo, usuario responsable.

---

## 4. Requisitos de IA Generativa (ej. Gemini)

### 4.1 Rol de la IA generativa

* Redactar hallazgos preliminares para los médicos basados en:

  * Resultados numéricos de los modelos de visión/ómica/tabular.
  * Señales epidemiológicas (probabilidad base en la cohorte).
* Generar resúmenes multimodales tipo "caso clínico" para comités oncológicos.
* Explicar el razonamiento del modelo en lenguaje natural para auditoría interna.
* Generar alertas priorizadas en lenguaje claro ("riesgo alto de lesión maligna en lóbulo superior derecho").

### 4.2 Restricciones a la IA generativa

* Debe incluir siempre un descargo de responsabilidad explícito:
  "Esto es un análisis automatizado de apoyo clínico. No es un diagnóstico definitivo ni una indicación terapéutica. Requiere revisión humana especializada".
* Debe evitar afirmaciones absolutas del tipo "tienes cáncer".
* Debe evitar recomendaciones terapéuticas directas (fármacos, dosis, tratamientos) sin validación humana.
* Debe registrar cada generación junto con la versión del modelo base y la versión del prompt/instrucciones usadas.

---

## 5. Seguridad, privacidad y cumplimiento

### 5.1 Privacidad / PHI / PII

* Toda la data de entrenamiento debe provenir de fuentes públicas ya desidentificadas o pasar por un proceso estricto de desidentificación que elimine identificadores personales (nombres, fechas exactas de nacimiento, ID hospitalarios, etc.).
* El pipeline de ingesta debe inspeccionar y limpiar automáticamente metadatos DICOM (imágenes médicas) y columnas clínicas sensibles:

  * Remover nombres, direcciones, teléfonos, correos, números de historia clínica u otros campos que permitan identificar a una persona.
  * Reemplazar fechas exactas de eventos clínicos por rangos relativos (por ejemplo, "día 0", "día +7").
* Todos los datasets internos deben etiquetarse con su nivel de sensibilidad ("público desidentificado", "uso restringido", etc.), licencia y restricciones de redistribución.

### 5.2 Cumplimiento y auditoría

* Cada acceso a datos sensibles y cada inferencia clínica debe quedar trazado (quién, cuándo, para qué paciente/caso).
* La plataforma debe generar logs de auditoría inmutables.
* Se debe habilitar un panel de trazabilidad para revisiones regulatorias internas.

### 5.3 Control de acceso / IAM

* Autenticación fuerte (MFA) para perfiles clínicos y de investigación.
* Autorización basada en rol (ej. "Clínico Diagnóstico", "Investigador IA", "Administrador de Cumplimiento").
* Segmentación de datos: los usuarios de investigación NO pueden ver datos clínicos propietarios cargados por usuarios clínicos.

---

## 6. Configuración centralizada y manejo seguro de secretos

### 6.1 Estructura propuesta de configuración

Crear un módulo interno `config/` con al menos:

* `config/settings.yaml`
  Parámetros de despliegue, nombres de buckets, rutas de almacenamiento, flags de activación de módulos, versiones mínimas de modelo aprobadas.

* `config/secrets/`  (NUNCA versionado en Git)
  Variables críticas gestionadas por variables de entorno:

  * `GEMINI_API_KEY` → clave privada para llamadas a la IA generativa.
  * `MODEL_REGISTRY_URI` → conexión segura al registro de modelos.
  * `DB_CONN_STRING` → cadena de conexión cifrada para metadatos clínicos/auditoría.
  * `EXTERNAL_DATA_TOKENS` → tokens para acceder a fuentes de datos abiertas.

* `config/policies/`
  Reglas sobre retención, auditoría, niveles de acceso, licencias de datasets, disclaimers regulatorios para los reportes generados.

### 6.2 Reglas obligatorias de secretos

* Ninguna clave o credencial puede quedar hardcodeada en el código fuente, notebooks, prompts o logs.
* Los contenedores/servicios deben leer secretos **sólo en runtime** vía variables de entorno.
* Debe existir rotación periódica de llaves y registro de uso.

---

## 7. Componentes técnicos mínimos a implementar

### 7.1 Servicios backend (microservicios / módulos)

1. **data-ingestor**

   * Descarga datasets abiertos.
   * Valida licencias.
   * Aplica desidentificación / limpieza de metadatos sensibles.
   * Publica data cruda y curada en el data lake.

2. **data-curator**

   * Estandariza esquemas.
   * Genera diccionarios de datos y catálogos.
   * Actualiza el lineage.

3. **feature-engineering**

   * Extrae embeddings de imagen, firmas genómicas/proteómicas y variables clínicas tabulares.
   * Publica features consolidadas en la Feature Store.

4. **trainer-mlops**

   * Ejecuta entrenamiento (batch).
   * Calcula métricas y fairness.
   * Registra modelo en el Model Registry.

5. **inference-service**

   * Expone API segura para inferencia en tiempo casi real.
   * Genera mapas de calor / explicaciones visuales.
   * Devuelve probabilidades y hallazgos estructurados.

6. **audit-compliance**

   * Centraliza los logs de acceso, inferencia, versiones de modelo usadas y evidencia de cumplimiento.
   * Expone dashboard para auditores internos.

### 7.2 Frontend / UI

* **Portal Clínico (Diagnóstico):**

  * Página de carga de estudio (arrastrar y soltar imagen / ZIP DICOM / panel genómico).
  * Vista de resultados con:

    * Probabilidad estimada por tipo de lesión / tipo de cáncer.
    * Heatmap/segmentación sobre la imagen.
    * Texto explicativo generado por IA generativa + descargo legal.
  * Botón "Descargar Informe Preliminar" (PDF).

* **Portal Investigación / Ciencia de Datos:**

  * Catálogo de datasets armonizados.
  * Lanzar nuevos entrenamientos y ver métricas comparativas.
  * Curvas ROC, matrices de confusión, análisis de sesgo.

* **Portal Cumplimiento / Auditoría:**

  * Historial de inferencias.
  * Versiones de modelo por caso.
  * Alertas de acceso indebido.
  * Evidencia de que cada reporte incluyó descargo legal obligatorio.

---

## 8. Reglas de explicabilidad y responsabilidad clínica

1. Cada predicción debe venir acompañada de una explicación visual o estadística comprensible (por ejemplo: "la región sospechosa está resaltada en rojo" o "se detectó una firma genómica asociada a alto riesgo").
2. El sistema debe registrar y mostrar siempre la versión exacta del modelo que generó la predicción.
3. El sistema **no puede** sugerir tratamientos específicos ni pronósticos individuales definitivos sin revisión humana.
4. Cada salida visible al personal médico debe contener un recordatorio legal:
   "Herramienta de apoyo clínico. La decisión final corresponde al equipo médico tratante."

---

## 9. Roadmap de implementación recomendado

### Fase 0. Fundaciones técnicas

* Montar configuración centralizada.
* Definir buckets/zones de datos (raw / curated / features).
* Definir esquema base del registro de modelos.
* Implementar autenticación y control de acceso básico.

### Fase 1. Ingesta + Curación inicial

* Conectar `data-ingestor` a las primeras fuentes abiertas.
* Normalizar formatos de imágenes (DICOM → NIfTI estandarizado cuando aplique).
* Estandarizar tablas clínicas y genómicas.
* Etiquetar datasets con tipo de cáncer, licencia, nivel de sensibilidad.

### Fase 2. Entrenamiento unimodal inicial

* Entrenar modelos específicos por modalidad (visión médica para pulmón y mama, tabular clínico-epidemiológico, genómica/proteómica).
* Medir sensibilidad, especificidad, AUC ROC.
* Registrar resultados en el Model Registry.

### Fase 3. Fusión multimodal

* Implementar feature-level fusion o late fusion para combinar imagen + variables clínicas + marcadores moleculares.
* Validar mejoras en sensibilidad para casos tempranos.

### Fase 4. Interfaz Clínica

* Construir el Portal Clínico con carga de estudios y visualización de resultados.
* Incluir descargos legales automáticos.

### Fase 5. Cumplimiento y Auditoría End-to-End

* Activar `audit-compliance`.
* Exponer dashboard de auditoría y trazabilidad.
* Establecer procedimientos de revisión periódica de desempeño del modelo, sesgos y drift de datos.

---

## 10. Criterios de Aceptación (Definition of Done por macro-módulo)

### Ingesta / Curación

* Soporta descarga automatizada de al menos 3 fuentes abiertas.
* Limpia y desidentifica metadatos sensibles antes de almacenarlos.
* Versiona datasets con huella criptográfica.

### Entrenamiento / MLOps

* Puede lanzar entrenamiento reproducible desde línea de comando o UI de investigación.
* Publica métricas clínicamente relevantes y fairness por subgrupo.
* Registra pesos, hiperparámetros y dataset exacto en el Model Registry.

### Inferencia Clínica

* Acepta nuevos estudios subidos por un usuario clínico autenticado.
* Devuelve probabilidad de lesión sospechosa / tipo de cáncer y una explicación visual.
* Genera reporte preliminar en lenguaje natural con el descargo legal.
* Loggea versión del modelo y guarda auditoría.

### UI / Seguridad / Cumplimiento

* Autenticación MFA.
* Control de acceso basado en rol.
* Panel de auditoría disponible.
* Secretos gestionados vía variables de entorno, nunca hardcodeados.
* Todos los reportes incluyen descargo legal obligatorio.

---

**Fin del documento.**
