# **Pipeline de Generación y Procesamiento de Dataset Conversacional para Tutoría Matemática**

## **1. Descripción General**

Este paquete (`src/`) implementa un **pipeline modular** para la construcción y procesamiento de datasets conversacionales enfocados en tutoría matemática para estudiantes de **3° a 5° de primaria**.

Su propósito es:

- **Filtrar** problemas matemáticos apropiados al nivel escolar.
- **Simular conversaciones** entre un estudiante y tutores virtuales.
- **Generar datasets de preferencias (Dp)** para técnicas como **DPO (Direct Preference Optimization)**.
- **Traducir y clasificar pedagógicamente** el contenido al español.
- **Mantener trazabilidad y control de calidad** mediante rúbricas predefinidas.

---

## **2. Estructura de la Carpeta `src/`**

| Archivo | Función |
|---------|---------|
| `build_dp.py` | Convierte un dataset conversacional (Dc) en un dataset de preferencias (Dp) generando respuestas *chosen* y *rejected*. |
| `generate_dc.py` | Simula interacciones alumno–tutor usando modelos locales y genera el dataset conversacional (Dc). |
| `data_download_chunking.py` | Descarga el dataset base (`OpenMathInstruct-2`) y lo divide en fragmentos manejables en formato `.jsonl`. |
| `data_filtering.py` | Filtra problemas usando LLMs y reglas explícitas para asegurar que sean adecuados para 3°–5° grado. |
| `data_filtering_2.py` | Variante simplificada del filtrado, con menos ejemplos pero misma lógica de clasificación. |
| `data_translate.py` | Traduce todos los textos a español y agrega clasificación pedagógica (`area_conocimiento` y `objetivo_pedagogico`). |
| `translate_lib.py` | Función de alto nivel para obtener *translators* (traductor, clasificador, resumidor) desde un modelo Ollama. |
| `util_translate_summarize.py` | Implementa la lógica principal de traducción, clasificación y resumen. |
| `llms.py` | Fábrica de instancias LLM locales usando `langchain_ollama`. |
| `context.py` | Construye el contexto conversacional a partir de turnos previos de alumno y tutor. |
| `validate.py` | Verifica que las salidas de los modelos cumplan con las rúbricas y formato JSON esperado. |
| `rubrics.py` | Tablas maestras para codificación de evaluación pedagógica. |


---

## **3. Flujo de Trabajo del Pipeline**

### **Paso 1 – Descargar y Dividir el Dataset Base**
Se usa `data_download_chunking.py` para descargar `nvidia/OpenMathInstruct-2` desde Hugging Face y dividirlo en partes de 250,000 líneas.

```bash
python src/data_download_chunking.py
```

### Paso 2 – Filtrar Problemas Adecuados

El filtrado elimina problemas que excedan el nivel de primaria baja y clasifica por tema y objetivo pedagógico.

Ejemplo con data_filtering.py:

```bash
python src/data_filtering.py split_openmathinstruct2/openmathinstruct2_part_1.jsonl \
       filtered_part_1.jsonl --model gemma3:1b
```

o:

```bash
python src/data_filtering_2.py input.jsonl output.jsonl --model gemma3:1b
```

### Paso 3 – Generar Dataset Conversacional (Dc)

Simula un diálogo guiado por rúbricas entre un alumno virtual y dos tutores virtuales (roles AT y AS). El resultado es un .jsonl con múltiples turnos por pregunta.

```bash
python src/generate_dc.py --questions filtered_part_1.jsonl --output dc_part_1.jsonl
```

### Paso 4 – Construir Dataset de Preferencias (Dp)

Genera pares chosen/rejected para entrenamiento con DPO.

```bash
python src/build_dp.py --dc dc_part_1.jsonl --dp dp_part_1.jsonl
```

## 4. Componentes de Utilidad

context.py: Genera cadenas de contexto de conversación (Alumno: / Tutor:) para prompts.

validate.py: Garantiza que las salidas de los LLM sigan el formato y códigos válidos.

rubrics.py: Define:

```
Eval of Student Response (a–g)

Action Based on Eval (1–12)

Subproblem State (w–z)
```

llms.py: Inicializa modelos Ollama con parámetros de temperatura y timeout.

## 5. Notas Finales

El pipeline está diseñado para ser modular: cada etapa se puede ejecutar de forma independiente.

Se recomienda mantener los archivos .jsonl intermedios para depuración y trazabilidad.

Los modelos Ollama deben estar previamente descargados y disponibles localmente para evitar llamadas externas.