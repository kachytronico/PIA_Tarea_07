# UD7_P2 – Guía del agente (uso general)

Este documento explica **cómo debe trabajar el agente** contigo para resolver el Problema 2 de la UD7 (`¿Esto es real?`).[file:44]

Está pensado para que lo leas tú y también para que el agente lo tenga como referencia adicional.

---

## 1. Objetivo de la guía

- Que el agente tenga claro **cómo organizar el trabajo en el cuaderno de Colab** para P2.
- Que sepa **qué estilo de ayuda quieres** (progresivo, por hitos, con código mínimo y explicaciones cortas).
- Que maximice la reutilización de tus chuletas RA2/RA4 y del código de UD5.

---

## 2. Cómo debe trabajar el agente dentro del Colab de P2

### 2.1 Estructura del cuaderno recomendada

El cuaderno ideal de P2 debería tener secciones tipo:

1. **Introducción y carga de librerías**
   - Texto breve explicando el objetivo del problema.
   - Celda con imports (fastai.vision.all, pathlib, matplotlib, etc.).
2. **Exploración y preparación del dataset (Hitos 1–3, RA2)**
   - Carga y visualización rápida de la estructura de P2.zip.
   - Fusión/reorganización de carpetas (Hito 1).
   - División train/test (Hito 2).
   - Creación de DataBlock/DataLoaders con split 90/10 sobre train (Hito 3).
3. **Entrenamiento de modelos (Hitos 4–6, RA4)**
   - Modelo base con ResNet18.
   - Búsqueda de LR y progressive resizing.
   - Segundo modelo con otra arquitectura.
4. **Evaluación y test (Hito 7, RA4)**
   - Test en el conjunto de test.
   - Matriz de confusión.
5. **Interpretabilidad (Hito 8, RA4)**
   - Ejemplos de interpretabilidad.
   - Comentarios sobre en qué se fija el modelo.

### 2.2 Estilo de ayuda

El agente debe:

- Trabajar **hito a hito**.
- Para cada hito:
  - Explicar en 1–2 frases qué se pide.
  - Proponer celdas de código mínimas.
  - Indicar qué debes adaptar (ruta base, nombre de carpetas, tamaños de imagen).
- Mantener código reproducible, con `random_seed` fijo cuando sea relevante.
- Usar buenas prácticas aprendidas en UD5: `DataBlock`, `vision_learner`, `fine_tune`, etc.

---

## 3. Correspondencia Hito ↔ RA ↔ Unidad

- Hitos 1–3 → **RA2** (preprocesamiento y gestión de datos en visión) → relacionados con UD2/UD3 pero aplicados a imágenes (UD5).[file:44]
- Hitos 4–8 → **RA4** (deep learning, evaluación, interpretabilidad) → UD5 (visión) + proyecto UD7.[file:44]

El agente debe mencionar estos RA cuando explique el propósito de cada bloque de código, para reforzar la conexión con las chuletas.

---

## 4. Pautas específicas por hito

### Hito 1 – Fusionar carpetas (RA2)

- Pedir al usuario que muestre la estructura inicial (por ejemplo con `!tree` o `os.listdir`).
- Proponer código que:
  - Localice las carpetas `real` y `ai`.
  - Dentro de cada una, fusione las subcarpetas de "categorías" en una estructura uniforme, por ejemplo:
    - `dataset/all/real/...`
    - `dataset/all/ai/...`
- Insistir en no borrar datos sin copia de seguridad (usar `shutil.copy` o documentar bien si se usa `move`).

### Hito 2 – Train/test 75/25 (RA2)

- Sugerir una función que:
  - Lea todas las rutas de `dataset/all`.
  - Haga un split estratificado 75/25 por clase.
  - Cree dos carpetas: `dataset/train` y `dataset/test`, copiando/moviendo las imágenes.
- Recordar que la parte de test NO debe usarse en entrenamiento ni validación.

### Hito 3 – DataBlock y DataLoaders (RA2)

- Proponer un `DataBlock` como:
  - `blocks=(ImageBlock, CategoryBlock)`.
  - `get_items=get_image_files` desde `dataset/train`.
  - `get_y=parent_label`.
  - `splitter=RandomSplitter(valid_pct=0.1, seed=33)`.
  - `item_tfms=Resize(224)` inicialmente.
- Crear `dls = data_block.dataloaders(path, bs=32)` y mostrar un batch.

### Hitos 4–6 – Modelos deep (RA4)

- Hito 4: modelo base con ResNet18 (por ejemplo `vision_learner(dls, resnet18, metrics=accuracy)` + `fine_tune`).
- Hito 5: `lr_find()` + progressive resizing:
  - Entrenar primero con un tamaño menor (ej. 128), luego 224, luego 384.
  - Explicar cómo cambiar `item_tfms` en el DataBlock o usar `Resize()` en `dls`.
- Hito 6: repetir proceso con otra arquitectura (ej. `resnet34`, `resnet50`).

### Hito 7 – Test y matriz de confusión (RA4)

- Crear un `DataLoader` para test (por ejemplo, otro `DataBlock` que apunte a `dataset/test`).
- Obtener predicciones con `learn.get_preds(dl=test_dl)`.
- Construir la matriz de confusión con `sklearn.metrics.confusion_matrix` o con `ClassificationInterpretation`.

### Hito 8 – Interpretabilidad (RA4)

- Si la versión de FastAI lo permite, usar `ClassificationInterpretation.from_learner(learn)` y `plot_top_losses`/métodos similares.
- Proponer al menos una visualización que ayude a ver en qué se fija el modelo.
- Guiar al usuario a escribir 4–5 frases que conecten la interpretación con el objetivo del problema (“¿cómo distingue la IA fotos reales de imágenes generadas?”).

---

## 5. Cosas que el agente debe evitar

- Cambiar el significado de los hitos o ignorar alguno sin avisar.[file:44]
- Reescribir toda la tarea sin permitir que el usuario entienda los pasos.
- Usar un stack completamente distinto (por ejemplo, solo Keras sin FastAI) a menos que el usuario lo pida expresamente.
- Hacer suposiciones fuertes sobre la estructura de carpetas sin mirar primero.

---

## 6. Cómo debe interactuar con el usuario

- Preguntar siempre:
  - “¿En qué hito estás?” o “¿Seguimos por el siguiente hito pendiente?”
- Cuando el usuario copie un error de Colab, ayudar a depurarlo paso a paso (sin rehacer todo desde cero si no hace falta).
- Referenciar las chuletas de RA2 y RA4 cuando un bloque sea equivalente (por ejemplo, pipeline de datos, DataBlock de imágenes, evaluación con matriz de confusión).

Fin del archivo `UD7_P2_GuiaAgente`.
