# UD7_P2 – Tarea y solución estructurada

Este documento resume el **Problema 2: “¿Esto es real?”** de la UD7 y propone una **solución estructurada por hitos**, sin entrar en todos los detalles de código, para que puedas usarlo como guía y contexto del agente.[file:44]

---

## 1. Resumen del enunciado (parafraseado)

El Problema 2 plantea entrenar un modelo de visión por computador capaz de distinguir entre:

- Imágenes que son **fotografías reales**.
- Imágenes que han sido **generadas por IA**.

El dataset se proporciona en un ZIP (`P2.zip`) con carpetas separadas para imágenes reales y generadas. Tu objetivo es construir un sistema de clasificación binaria que identifique correctamente el origen de cada imagen.

La estructura del trabajo se organiza en 8 hitos, que combinan RA2 (gestión y preparación de datos) y RA4 (modelado deep, evaluación e interpretabilidad).[file:44]

---

## 2. Hitos del Problema 2 y RA asociados

### Hito 1 (RA2) – Fusionar carpetas de categorías

**Enunciado (parafraseado):**
Reorganiza las carpetas de “categorías” que están dentro de `real` y `ai` en una sola estructura coherente, apta para ser usada por un DataBlock de FastAI.

**Objetivo técnico:**
- Pasar de una organización posiblemente fragmentada a una estructura limpia del tipo:
  - `dataset/all/real/...`
  - `dataset/all/ai/...`

**Pistas de solución:**
- Inspeccionar con `os.listdir` y/o `!tree` para entender la estructura real.
- Usar `pathlib.Path` y `shutil.move` o `shutil.copy` para fusionar las subcarpetas.

---

### Hito 2 (RA2) – División train/test 75%/25%

**Enunciado (parafraseado):**
Divide el dataset en entrenamiento (75%) y test (25%) de forma estratificada por clase.

**Objetivo técnico:**
- Crear dos conjuntos:
  - `dataset/train` (75% de las imágenes de cada clase).
  - `dataset/test` (25% restante).

**Pistas de solución:**
- Listar todas las imágenes por clase.
- Usar una función de split (por ejemplo, seleccionando índices aleatorios) manteniendo la proporción de cada clase.
- Copiar o mover los archivos a carpetas `train` y `test`.

---

### Hito 3 (RA2) – DataBlock y DataLoader con validación 90/10

**Enunciado (parafraseado):**
A partir del conjunto de entrenamiento, crea un `DataBlock` y un `DataLoader` que dividan los datos en un 90% para entrenamiento y un 10% para validación.

**Objetivo técnico:**
- Definir un `DataBlock` similar a:
  - `blocks=(ImageBlock, CategoryBlock)`
  - `get_items=get_image_files` (sobre `dataset/train`)
  - `get_y=parent_label`
  - `splitter=RandomSplitter(valid_pct=0.1, seed=33)`
- Crear `dls = data_block.dataloaders(path_train, bs=32)`.

**Pistas de solución:**
- Ver ejemplos UD5 de DataBlock para clasificación de imágenes.
- Verificar con `dls.show_batch()` que las clases están bien etiquetadas.

---

### Hito 4 (RA4) – Entrenamiento con ResNet-18

**Enunciado (parafraseado):**
Entrena un modelo de deep learning usando una arquitectura ResNet-18 para este problema de clasificación binaria.

**Objetivo técnico:**
- Crear un learner tipo:
  - `learn = vision_learner(dls, resnet18, metrics=accuracy)`
- Usar `fine_tune(n_epochs)` para entrenar.

**Pistas de solución:**
- Usar transfer learning (no entrenar desde cero).
- Elegir un número razonable de épocas (por ejemplo 3–5) para ajustar el modelo sin sobreentrenar.

---

### Hito 5 (RA4) – Búsqueda de LR y progressive resizing

**Enunciado (parafraseado):**
Aplica la técnica de búsqueda del mejor *learning rate* (`lr_find`) y la técnica de **progressive resizing** con al menos 3 tamaños de imagen.

**Objetivo técnico:**
- Llamar a `learn.lr_find()` para estimar un LR adecuado.
- Entrenar en fases con tamaños crecientes, por ejemplo:
  - Fase 1: `Resize(128)`
  - Fase 2: `Resize(224)`
  - Fase 3: `Resize(384)` (o similar)

**Pistas de solución:**
- Ajustar el `DataBlock` o los `item_tfms`/`batch_tfms` entre fases.
- Tomar nota de cómo mejora (o no) la métrica al aumentar el tamaño.

---

### Hito 6 (RA4) – Segundo modelo con otra arquitectura

**Enunciado (parafraseado):**
Entrena un segundo modelo cambiando la familia de la arquitectura (por ejemplo, ResNet34, ResNet50 u otra familia disponible).

**Objetivo técnico:**
- Crear un nuevo learner con otra arquitectura:
  - `learn2 = vision_learner(dls, resnet34, metrics=accuracy)`
- Repetir el proceso de entrenamiento (con o sin progressive resizing, según tiempo disponible).

**Pistas de solución:**
- Comparar resultados entre ResNet18 y la nueva arquitectura (accuracy, tiempo de entrenamiento, estabilidad).

---

### Hito 7 (RA4) – Test y matriz de confusión

**Enunciado (parafraseado):**
Evalúa tu modelo usando el conjunto de test (el 25% reservado) y muestra la matriz de confusión.

**Objetivo técnico:**
- Definir un DataLoader para test (otra instancia de `DataBlock` o `ImageDataLoaders.from_folder` apuntando a `dataset/test`).
- Obtener predicciones con `learn.get_preds(dl=test_dl)`.
- Construir la matriz de confusión mediante `sklearn.metrics.confusion_matrix` o equivalentes.

**Pistas de solución:**
- Asegurarse de que las clases (`vocab`) son las mismas que en entrenamiento.
- Comentar qué tipo de errores comete el modelo (falsos positivos vs falsos negativos).

---

### Hito 8 (RA4) – Interpretabilidad

**Enunciado (parafraseado):**
Aplica técnicas de interpretabilidad para ver en qué se fija el modelo y justifica su respuesta.

**Objetivo técnico:**
- Usar herramientas de interpretación de FastAI si están disponibles:
  - `ClassificationInterpretation.from_learner(learn)`.
  - `interp.plot_top_losses()` u otros métodos similares.
- Alternativamente, integrar alguna variante de Grad-CAM si encaja con la versión de la librería.

**Pistas de solución:**
- Elegir algunos ejemplos de imágenes reales vs IA donde el modelo acierta y falla.
- Escribir un pequeño comentario sobre qué patrones parece usar el modelo (texturas, ruido, artefactos, etc.).

---

## 3. Estructura recomendada del cuaderno (para ti y para el agente)

Una posible estructura de secciones en el notebook de Colab:

1. **Configuración inicial**
   - Imports.
   - Montaje de Drive (si se usa) y descompresión de `P2.zip`.
2. **Hito 1 – Exploración y fusión de carpetas (RA2)**
   - Exploración de estructura.
   - Código de fusión.
3. **Hito 2 – División train/test (RA2)**
   - Split estratificado 75/25.
4. **Hito 3 – DataBlock/DataLoaders (RA2)**
   - Definición de DataBlock.
   - `dls.show_batch()`.
5. **Hito 4–6 – Modelos deep (RA4)**
   - Modelo base ResNet18.
   - LR find + progressive resizing.
   - Segundo modelo.
6. **Hito 7 – Evaluación en test (RA4)**
   - Predicciones en test.
   - Matriz de confusión.
7. **Hito 8 – Interpretabilidad (RA4)**
   - Visualizaciones.
   - Comentarios.

---

## 4. Cómo usar este documento con el agente

- Cárgalo como contexto adicional junto con `PIA_07_Tarea.pdf`.
- Indícale que use este archivo como “mapa” para no saltarse ningún hito.
- Cuando le pidas ayuda, puedes referirte al hito por número y al bloque correspondiente de este documento.

Ejemplo:
> “Estamos en el Hito 5 de P2. Según UD7_P2_TareaSol, toca LR find y progressive resizing. Dame el código mínimo para hacerlo con ResNet18 en FastAI, indicando qué tamaños usar y qué partes debo adaptar.”

Fin del archivo `UD7_P2_TareaSol`.
