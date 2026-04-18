# DOC3 — REGLAS CRÍTICAS Y SEÑALES DE ALARMA (v3)

**Módulo:** Programación de Inteligencia Artificial (PIA)
**Propósito:** 13 reglas inquebrantables + checklists por tipo de problema + señales de alarma a detectar en los outputs antes de avanzar.
**Cómo usar:** El agente aplica este documento como **checklist activo**, no como referencia pasiva. Antes de cualquier entrenamiento, recorre el checklist correspondiente al tipo de problema. Después de cada hito ejecutado, revisa las señales de alarma.

**Cambios respecto a v2:** Nueva Regla 11 (lectura literal del enunciado), Nueva Regla 12 (no silenciar errores con try/except), Nueva Regla 13 (auditoría final). Nuevas señales de alarma basadas en los fallos reales de ORD1 y ORD2.

---

## LAS 13 REGLAS INQUEBRANTABLES

Provienen del Decálogo oficial del curso PIA ampliado con 3 reglas añadidas tras análisis de fallos reales en simulacros de examen. Violarlas penaliza directamente la nota.

---

### REGLA 1 — ERRADICACIÓN DEL DATA LEAKAGE PREDICTIVO
**Aplica a: RA2, RA3 | Momento: durante el AED, antes de modelar**

Incluir una variable que revela directamente la respuesta hace que el modelo aprenda el atajo en lugar de un patrón real.

**Señal de alarma:** F1-score o accuracy = 1.0 exacto (o > 0.99) desde las primeras iteraciones.

```python
# Detectar leakage antes del split — cruzar cada categórica con el target
TARGET = "anomaly_label"  # CAMBIAR
for col in df.select_dtypes(include="object").columns:
    if col == TARGET: continue
    tabla = pd.crosstab(df[col], df[TARGET], normalize="index")
    if tabla.max().max() > 0.95:
        print(f"⚠️  LEAKAGE: '{col}' predice '{TARGET}' con {tabla.max().max():.1%} → ELIMINAR")
```

**Fix:** Eliminar la columna sospechosa. Si F1 baja drásticamente → era leakage. Si apenas cambia → era redundante.

---

### REGLA 2 — SECUENCIA: SPLIT → PREPROCESAMIENTO QUE APRENDE DATOS
**Aplica a: RA2, RA3 | Momento: al inicio de cualquier pipeline tabular**

El split (train/val/test) debe ocurrir **antes** de cualquier transformación que aprenda la distribución de los datos: imputación, escalado, codificación ordinal, PCA, **outlier_eliminator**.

**Excepción válida:** eliminar columnas completamente irrelevantes (IDs, nombres propios, columnas de texto libre) SÍ puede hacerse antes del split.

**Ejemplo del error:**
```python
# ❌ MAL: escalar ANTES de dividir → contamina test con estadísticos de todo el dataset
scaler.fit_transform(df)
X_train, X_test = train_test_split(...)

# ✅ BIEN: dividir PRIMERO, escalar solo sobre train
X_train, X_test = train_test_split(...)
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
```

**Anti-ejemplo real de ORD2-P1 (fallo):**
```python
# ❌ MAL: outlier_eliminator aplicado ANTES del split → contamina test
df = outlier_eliminator(df, continuous_cols)
X = df.drop('Heart Attack Risk', axis=1)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, ...)
```

---

### REGLA 3 — PROTOCOLO fit vs transform
**Aplica a: RA2, RA3 | Momento: en cada uso de scaler, imputer, PCA**

`.fit()` o `.fit_transform()` → **exclusivamente sobre X_train o X_fold_train**.
`.transform()` → sobre X_val, X_test, X_fold_val. Nunca `.fit_transform()` sobre conjuntos de evaluación.

```
Correcto:
  scaler.fit_transform(X_train)   ✓
  scaler.transform(X_val)         ✓
  scaler.transform(X_test)        ✓

Error:
  scaler.fit_transform(X_test)    ✗  ← leakage
  scaler.fit_transform(X_val)     ✗  ← leakage
```

---

### REGLA 4 — EL CONJUNTO DE TEST ES SAGRADO
**Aplica a: RA3, RA4 | Momento: durante toda la fase de modelado**

El test solo se usa para la evaluación FINAL, con el modelo ya elegido. No para comparar modelos, no para ajustar hiperparámetros.

**Cómo optimizar sin tocar el test:**
- Validación cruzada (StratifiedKFold) sobre el conjunto de entrenamiento.
- GridSearchCV que hace kfold internamente.
- Evaluar en test una sola vez al final.

**Señal de alarma:** Si aparece `predict(X_test)` más de una vez en el cuaderno para comparar opciones → test leakage.

Prohibido: eliminar filas del test (outliers, nulos, rangos). Prohibido: `fit_transform` sobre test.

---

### REGLA 5 — ESCALADO OBLIGATORIO ANTES DE DISTANCIAS
**Aplica a: RA2, RA3 | Momento: antes de KNN, SVM, KMeans, DBSCAN**

| Algoritmo | ¿Requiere escalado? |
|-----------|---------------------|
| KNN | ✓ Sí (sensible a magnitudes) |
| SVM | ✓ Sí (el kernel trabaja con distancias) |
| KMeans | ✓ Sí (distancia euclidiana) |
| DBSCAN | ✓ Sí (radio eps en unidades del espacio) |
| PCA | ✓ Sí (varianza dominada por magnitud) |
| MLP / Redes neuronales | ✓ Sí |
| Árbol de decisión | ✗ No (umbral en el feature original) |
| Random Forest | ✗ No (ensemble de árboles) |

---

### REGLA 6 — ESTRATIFICACIÓN OBLIGATORIA EN DESBALANCEO
**Aplica a: RA2, RA3 | Momento: en train_test_split y StratifiedKFold**

Si la clase minoritaria representa < 30% del total → siempre `stratify=y` y `StratifiedKFold`.

```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y,
    test_size=0.2, stratify=y, random_state=random_seed)

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
```

---

### REGLA 7 — THRESHOLD EN SELF-TRAINING
**Aplica a: RA3 | Momento: al usar SelfTrainingClassifier**

Solo añadir pseudoetiquetas cuando el modelo tiene alta confianza. Threshold mínimo recomendado: 0.9 (90%).

```python
st_model = SelfTrainingClassifier(base_model, threshold=0.9)
```

---

### REGLA 8 — JUSTIFICACIÓN DEL K EN CLUSTERING
**Aplica a: RA2 (UD3) | Momento: antes de entrenar KMeans**

Nunca elegir K arbitrariamente. Mostrar siempre la gráfica del método del codo antes de instanciar `KMeans(n_clusters=K)`.

---

### REGLA 9 — PREVENCIÓN DE OVERFITTING EN DEEP LEARNING
**Aplica a: RA4 | Momento: antes y durante el entrenamiento FastAI**

```python
callbacks = [EarlyStoppingCallback(patience=3)]   # 1. EarlyStopping
learn = vision_learner(dls, resnet18, cbs=callbacks, metrics=accuracy)
learn.fine_tune(n)   # 2. Transfer learning (fine_tune, no fit desde cero)
# 3. Progressive resizing (ver DOC2)
```

**Señales de overfitting:** `train_loss` baja continuamente mientras `valid_loss` sube → EarlyStopping debe activarse.

---

### REGLA 10 — DOCUMENTACIÓN Y JUSTIFICACIÓN (CAJA BLANCA)
**Aplica a: todos los RAs**

El código sin justificación técnica se evalúa a la baja. Incluir comentarios:

```python
# Uso F1-score porque el dataset está desbalanceado (clase positiva = 15%)
# Escalo porque KNN es sensible a la magnitud de las variables
# Uso stratify=y para mantener la proporción de clases en train y test
# Uso fine_tune para aprovechar pesos preentrenados en ImageNet
```

---

### REGLA 11 — LECTURA LITERAL DEL ENUNCIADO (NUEVA v3)
**Aplica a: todos los RAs | Momento: antes de escribir código de cualquier hito**

**Problema detectado en ORD1-P2, ORD2-P1, ORD2-P2:** el agente reinterpretó o ignoró palabras literales del enunciado que cambiaban el pipeline entero.

**Protocolo obligatorio:**

Antes de proponer código, extrae las palabras clave del enunciado y comprueba si están en tu plan. Si una palabra aparece en el enunciado y no en tu plan → PARAR y corregir.

**Tabla de palabras clave críticas:**

| Palabra literal en el enunciado | Implicación técnica obligatoria |
|---------------------------------|-------------------------------|
| "Unet" | `unet_learner`, segmentación. NO `vision_learner`. |
| "kfold", "validación cruzada" | Patrón B. Preprocesamiento DENTRO del fold. |
| "reserva el X% para testeo" | `test_size=X/100`, un solo split. No añadir validación extra. |
| "entrena un embedding", "embedding propio" | gensim Word2Vec. NO Keras Embedding. |
| "aumento de datos" | TransformPipeline con albumentations. |
| "reducción de la precisión" | `.to_fp16()` en el learner. |
| "este entrenamiento podría durar días" | Dejar fine_tune COMENTADO. No ejecutar. |
| "cualquier X, incluso los no contemplados" | OneHotEncoder con `handle_unknown='ignore'`. |
| "matriz de confusión" | `ConfusionMatrixDisplay` + comentario interpretativo. |
| "interpretabilidad" | `plot_top_losses` + `most_confused`. |
| "ensemble" | Promedio manual (probabilidades o OR). NO VotingClassifier. |
| "AWD_LSTM" | fastai.text.all + `text_classifier_learner`. |

**Anti-ejemplo real de ORD2-P2 (fallo):**
```python
# Enunciado: "Entrena una arquitectura Unet con un esqueleto ResNet-18"
# ❌ Esto NO es Unet, es clasificación normal:
learn = vision_learner(dls, resnet18, metrics=accuracy)

# ✓ Correcto (palabra "Unet" respetada):
learn = unet_learner(dls, resnet18, metrics=accuracy)
```

**Anti-ejemplo real de ORD1-P2 (fallo):**
```python
# Enunciado: "Entrena un embedding a partir de tus datos de entrenamiento"
# ❌ Esto NO es un embedding propio del curso, es Keras:
from tensorflow.keras.layers import Embedding

# ✓ Correcto (palabra "embedding" respetada):
from gensim.models import Word2Vec
```

---

### REGLA 12 — NO SILENCIAR ERRORES CON try/except (NUEVA v3)
**Aplica a: todos los RAs | Momento: al encontrar un error o output inesperado**

**Problema detectado en ORD2-P2:** el agente rodeó `lr_find` con un `try/except` genérico que capturaba todo, fallback hardcoded a `1e-3`, y presentó la decisión como deliberada en la conclusión.

**Código problemático real:**
```python
# ❌ MAL: try/except captura todo y oculta el error real
try:
    suggestion = learn.lr_find(show_plot=True)
    lr_to_use = suggestion.valley
except:
    lr_to_use = 1e-3
    print(f"El buscador automático falló. Usando LR manual conservador: {lr_to_use}")
```

**Por qué está prohibido:**
1. El `except:` desnudo captura todo (incluidos errores de sintaxis, de datos, de librería).
2. No se muestra al usuario qué falló realmente.
3. El fallback hardcoded se presenta como decisión técnica, no como parche.

**Protocolo correcto:**

1. **Lee el traceback completo.** El error relevante está en la última línea.
2. **Identifica la causa concreta** (ruta, columna, tipo, versión, tamaño de dataset).
3. **Propón solo el fix mínimo.**
4. **Documenta el error real** antes de aplicar el fix.

**Ejemplo correcto para lr_find:**
```python
# Ejecutar directamente — si falla, leemos el traceback
learn.lr_find(show_plot=True)

# Si el output no es utilizable (dataset muy pequeño, pocos batches):
# Diagnóstico: comprobar len(dls.train)
# Si < 10 batches → bs demasiado grande o dataset muy pequeño
# Solución documentada: usar LR estándar del curso con justificación explícita
# Ejemplo: "Dataset pequeño (435 imágenes, 14 batches con bs=32).
#          lr_find no produce gráfico utilizable en esta escala.
#          Usamos LR=1e-3 por ser el estándar del curso para transfer learning."
```

**Excepción válida (raras):** try/except específico para un error concreto conocido, con log claro.

```python
# ✓ Aceptable: try/except específico y documentado
try:
    dataset = load_dataset_remoto()
except ConnectionError as e:
    print(f"Sin conexión. Usando dataset local como fallback. Error: {e}")
    dataset = load_dataset_local()
```

---

### REGLA 13 — AUDITORÍA FINAL DEL CUADERNO (NUEVA v3)
**Aplica a: todos los RAs | Momento: antes de declarar un problema completo**

**Problema detectado en ORD2-P1 y ORD2-P2:** el agente declaró problemas completos con hitos saltados (ORD2-P2 saltó H5), modelos que violaban reglas (VotingClassifier en ORD2-P1), y output alarmante sin atender (accuracy=1.000 no detectado como leakage).

**Checklist obligatorio antes de cerrar un problema:**

```
AUDITORÍA FINAL — Problema [P1|P2]

□ Mapeo completo con el enunciado:
  □ Los N hitos del enunciado están presentes en el cuaderno
  □ Cada hito cumple literalmente lo que pide (sin reinterpretar)
  □ Ningún hito ha sido saltado, fusionado ni renombrado sin avisar

□ Palabras clave literales (Regla 11):
  □ Recorrer el enunciado y subrayar palabras clave
  □ Verificar que cada palabra tiene su implicación técnica aplicada

□ Stack correcto:
  □ Tabular: solo pandas + sklearn + (FastAI/gensim si el enunciado lo pide)
  □ Visión: solo FastAI
  □ Texto: gensim o FastAI (NUNCA TensorFlow/Keras)
  □ Ningún import de tensorflow ni keras en todo el cuaderno

□ Reglas críticas:
  □ Orden del pipeline: split → outliers → nulos → codif → escalado → PCA
  □ outlier_eliminator solo en train, MEDIA como centro, <5%
  □ fit_transform solo en train, transform en val/test
  □ random_seed = 33 en todos los modelos y splits
  □ Ensemble tabular por promedio manual (NO VotingClassifier)
  □ Métrica correcta según tarea (RMSE regresión, F1 desbalanceo)

□ Señales de alarma atendidas:
  □ Accuracy/F1 = 1.0 → revisado y justificado (si no era leakage)
  □ Accuracy ≈ % clase mayoritaria → modelo no es degenerado
  □ NaN en métricas → ausente
  □ F1 = 0.0 con accuracy alto → ausente
  □ std > 0.1 entre folds → justificado

□ Test tratado como sagrado:
  □ X_test usado SOLO para la evaluación final
  □ Ninguna transformación `fit` sobre X_test
  □ Ninguna eliminación de filas de X_test

□ Código documentado (Regla 10):
  □ Cada decisión técnica clave tiene comentario justificativo
  □ Conclusiones de hitos escritas DESPUÉS del output real
```

**Informe al usuario tras la auditoría:**

```
AUDITORÍA FINAL — Problema [P1|P2]

✅ Hitos completados: [N/M]
✅ Palabras clave respetadas: [lista]
✅ Reglas críticas: [todas OK | violaciones listadas]
⚠️  Observaciones: [warnings no críticos]
❌ Violaciones graves: [listar] → propongo correcciones antes de cerrar

[Si hay violaciones → el problema NO se cierra hasta corregirlas]
[Si todo OK → "Problema [X] completado. Pasamos al Problema [Y]."]
```

---

## CHECKLISTS PRE-ENTRENAMIENTO POR TIPO DE PROBLEMA

### ☑ CHECKLIST — Problema tabular con kfold (Patrón P1)

```
ANTES DEL SPLIT
☐ Columnas completamente irrelevantes (IDs) eliminadas
☐ Leakage revisado con crosstab o correlación (Regla 1)
☐ Variables fuera de rango lógico filtradas

SPLIT
☐ train_test_split ejecutado ANTES de imputar, escalar o codificar (Regla 2)
☐ stratify=y usado si hay desbalanceo (Regla 6)
☐ X e y separados DESPUÉS del split

DENTRO DEL KFOLD (verificar en el código de kfold_pipeline)
☐ SimpleImputer instanciado y ajustado sobre X_fold_train en cada fold (Regla 3)
☐ StandardScaler instanciado y ajustado sobre X_fold_train en cada fold (Regla 3)
☐ PCA instanciado y ajustado sobre X_fold_train en cada fold (Regla 3)
☐ Solo .transform() sobre X_fold_val (Regla 3)
☐ Escalado aplicado para KNN y SVM (Regla 5)
☐ Árbol/RF sin escalado (Regla 5)

TEST
☐ X_test no ha sido usado para ninguna decisión previa (Regla 4)
☐ La evaluación final en test se hace UNA SOLA VEZ (Regla 4)
☐ random_seed = 33 en todos los modelos y splits

CÓDIGO
☐ Comentarios justificando cada decisión técnica clave (Regla 10)
☐ Variables llamadas X_train_val, X_test (no X_train, X_test si hay kfold)
```

---

### ☑ CHECKLIST — Problema tabular hold-out (Patrón A)

```
ENUNCIADO
☐ Porcentajes del split exactos según enunciado (Regla 11)
☐ No se añaden splits de validación que el enunciado no pide (Regla 11)

SPLIT
☐ train_test_split ejecutado ANTES del preprocesamiento (Regla 2)
☐ stratify=y si hay desbalanceo (Regla 6)

PREPROCESAMIENTO
☐ outlier_eliminator solo sobre train, centro=MEDIA, <5% (Regla 2)
☐ fit_transform() solo sobre X_train (Regla 3)
☐ transform() sobre X_test (Regla 3)
☐ Escalado antes de KNN/SVM (Regla 5)
☐ Árbol/RF sin escalado (Regla 5)
☐ Codificación: binary_categorizer o lambda (Regla 11)
☐ OneHotEncoder SOLO si enunciado dice "valores nuevos en producción"

MODELOS
☐ GridSearchCV usa solo X_train (no X_test) (Regla 4)
☐ X_test usado solo en la evaluación final (Regla 4)
☐ Red neuronal tabular: MLPRegressor/MLPClassifier de sklearn (nunca TF)
☐ Ensemble: promedio manual, NUNCA VotingClassifier (Regla 11)
☐ random_seed = 33

CÓDIGO
☐ Comentarios justificando decisiones (Regla 10)
☐ Ningún try/except genérico (Regla 12)
```

---

### ☑ CHECKLIST — Problema de visión (RA4 FastAI)

```
ENUNCIADO
☐ ¿Dice "Unet"? → unet_learner, no vision_learner (Regla 11)
☐ ¿Dice "aumento de datos"? → TransformPipeline con albumentations (Regla 11)
☐ ¿Dice "reducción de precisión"? → .to_fp16() en el learner (Regla 11)

DATOS
☐ Estructura de carpetas inspeccionada con os.listdir() antes de fusionar
☐ Train/test split de imágenes completado (carpetas físicas separadas)
☐ dls.show_batch() ejecutado y etiquetas verificadas
☐ dls.vocab confirmado antes de entrenar

DATABLOCK
☐ TransformPipeline(train=True) para entrenamiento
☐ TransformPipeline(train=False) para test (sin aumentos)
☐ RandomSplitter(valid_pct=0.X, seed=33) según enunciado
☐ GrandparentSplitter(valid_name="test") para el DataBlock de test

ENTRENAMIENTO
☐ EarlyStoppingCallback configurado (Regla 9)
☐ vision_learner o unet_learner con fine_tune, no fit desde cero (Regla 9)
☐ random_seed = 33 con set_seed(33)
☐ fastprogress==1.0.3 instalado

EVALUACIÓN EN TEST
☐ Vocab de train == Vocab de test (ver señales de alarma abajo)
☐ TODOS los callbacks eliminados antes de learn.validate()
☐ while len(learn.cbs) > 0: learn.cbs.pop()
☐ Matriz de confusión mostrada e interpretada

ALARMAS TÍPICAS EN VISIÓN
☐ accuracy = 1.000 en validación → ¿dataset demasiado pequeño? ¿leakage?
☐ lr_find falla → NO silenciar con try/except (Regla 12). Diagnosticar.

CÓDIGO
☐ Comentarios sobre decisiones de arquitectura y transfer learning (Regla 10)
```

---

### ☑ CHECKLIST — Problema de texto/PLN (RA4 FastAI o gensim)

```
ENUNCIADO
☐ ¿Dice "entrena un embedding"? → gensim Word2Vec (Regla 11)
☐ ¿Dice "AWD_LSTM"? → fastai.text.all + text_classifier_learner
☐ ¿Dice "fine-tune del modelo de lenguaje"? → DEJAR COMENTADO
☐ Ningún import de tensorflow ni keras en todo el cuaderno (Regla 11)

DATOS
☐ Desbalanceo verificado con value_counts()
☐ Downsampling aplicado si clase mayoritaria > 1.2 × clase minoritaria
☐ Columna "set" creada (False=train, True=val) antes del DataBlock
☐ dls_texto.show_batch() ejecutado (tokens xxbos, xxunk son normales)

ENTRENAMIENTO (FastAI)
☐ .to_fp16() añadido al learner (reduce memoria GPU)
☐ bs=64 (estándar del curso para texto)
☐ random_seed = 33 con set_seed(33)

EMBEDDING PROPIO (gensim)
☐ Word2Vec entrenado SOLO con tokens de train
☐ Vectorización por media de vectores de palabras
☐ Vectores como features para sklearn (SVM, RF)

EVALUACIÓN
☐ get_preds(reorder=False) para obtener predicciones
☐ classification_report con target_names=dls_texto.vocab
☐ Métrica acorde al enunciado (ROC-AUC si dice "curva ROC")

CÓDIGO
☐ Justificación del downsampling documentada (Regla 10)
☐ NUNCA import tensorflow ni import keras (Regla 11)
```

---

## SEÑALES DE ALARMA EN LOS OUTPUTS

### En datos tabulares (RA2/RA3)

| Lo que ves | Alarma | Acción inmediata |
|-----------|--------|-----------------|
| F1 o accuracy = 1.0 exacto | Data leakage | PARAR. Regla 1. Revisar columnas con crosstab. |
| `nan` en métricas | Nulos en X_train | `X_train.isnull().sum()` + revisar imputación |
| Accuracy test > accuracy train | Split mal hecho | Verificar X_train/X_test |
| **F1 = 0.0 con accuracy alto** | **VotingClassifier hard colapsado a clase mayoritaria (como ORD2-P1)** | **PARAR. Reemplazar por promedio manual de probabilidades (Regla 11)** |
| Accuracy ≈ % clase mayoritaria | Modelo degenerado | Revisar preprocesamiento y métricas |
| `UserWarning: The least populated class` | Clase demasiado pequeña para kfold | Usar `StratifiedKFold` con `shuffle=True` |
| PCA devuelve mismo nº columnas que entrada | `n_components` inadecuado | Verificar `pca.explained_variance_ratio_.sum()` |
| F1 varía mucho entre folds (std > 0.1) | Dataset inestable o mal estratificado | Verificar distribución y añadir `stratify` |
| `KeyError: 'columna'` | Nombre distinto al esperado | `df.columns.tolist()` |

### En visión / FastAI (RA4)

| Lo que ves | Alarma | Acción inmediata |
|-----------|--------|-----------------|
| **accuracy = 1.000 en validación** | **Dataset demasiado pequeño (como ORD2-P2 con 65 imgs) o leakage** | **PARAR. Documentar el problema y justificar. Regla 1.** |
| `dls.show_batch()` con etiquetas incorrectas | `get_y` o carpetas mal nombradas | Verificar estructura y `parent_label` |
| `valid_loss` = NaN desde la primera época | Error DataBlock o imágenes corruptas | Verificar carga con `dls.show_batch()` |
| Accuracy test = 0.5 (binario) | El modelo no aprende | Vocab mismatch — ver fix abajo |
| `AttributeError` en `learn.validate()` | Callbacks no eliminados | `while len(learn.cbs)>0: learn.cbs.pop()` |
| `CUDA out of memory` | Batch o modelo demasiado grande | `bs=16`, usar resnet18 |
| train_loss baja, valid_loss sube | Overfitting en progreso | EarlyStopping debe activarse (Regla 9) |
| **lr_find falla o no da gráfico** | **Dataset pequeño o bs inadecuado** | **NO silenciar con try/except (Regla 12). Diagnosticar.** |
| `dls.vocab` ≠ `dl_test.vocab` | Clases del DataBlock de test distintas | Forzar vocab del train en db_test |

```python
# FIX — Vocab mismatch entre train y test
vocab_train = learn.dls.vocab
db_test_fixed = DataBlock(
    blocks=(ImageBlock, CategoryBlock(vocab=vocab_train)),  # vocab forzado
    get_items=get_image_files,
    splitter=GrandparentSplitter(valid_name="test"),
    get_y=parent_label,
    item_tfms=[TransformPipeline(train=False)]
)
dl_test_fixed = db_test_fixed.dataloaders(dataset_limpio)
learn.dls = dl_test_fixed
while len(learn.cbs) > 0: learn.cbs.pop()
learn.validate()
```

### En PLN / texto (RA4)

| Lo que ves | Alarma | Acción inmediata |
|-----------|--------|-----------------|
| `import tensorflow` o `import keras` | Stack incorrecto (como ORD1-P2) | PARAR. Regla 11. Reescribir con gensim o FastAI. |
| Accuracy > 90% desde la primera época | Espejismo por desbalanceo | Verificar distribución. Aplicar downsampling. |
| `show_batch()` muestra casi todo `xxunk` | Texto vacío o vocabulario muy pequeño | Verificar columna de texto |
| `OOM` durante entrenamiento | Secuencias largas o bs grande | `bs=32` y/o truncar textos |
| F1 muy bajo solo en clase minoritaria | Desbalanceo no tratado | Downsampling (clase mayor = 120% de la menor) |
| language_model_learner empieza a entrenar | Violación Regla 11 | PARAR inmediatamente. Kernel → Interrupt. Comentar la celda. |

---

## SEÑALES DE ALARMA EN EL CÓDIGO (revisar antes de ejecutar)

Estas son violaciones detectables con lectura simple del código, sin ejecutarlo.

| Patrón en el código | Violación | Corrección |
|---------------------|-----------|------------|
| `import tensorflow` en cualquier parte | Stack incorrecto (Regla 11) | Reemplazar por gensim/FastAI/sklearn |
| `from tensorflow.keras` | Stack incorrecto (Regla 11) | Reemplazar por FastAI para visión/texto |
| `VotingClassifier(...)` | Ensemble prohibido | Promedio manual de probabilidades (DOC2 §3.7) |
| `VotingRegressor(...)` | Ensemble prohibido | Promedio manual `(p1+p2+p3)/3` |
| `except:` desnudo | Regla 12 | Leer traceback, diagnosticar |
| `except Exception:` sin log | Regla 12 | Mostrar el error al usuario |
| `vision_learner(...)` cuando enunciado dice "Unet" | Regla 11 | `unet_learner(...)` |
| `aug_transforms()` | RA4 | `TransformPipeline` con albumentations |
| `learn.fit(...)` en visión/texto | Regla 9 | `learn.fine_tune(...)` |
| `TextDataLoaders.from_df(...)` | RA4 | `DataBlock` con `ColSplitter("set")` |
| `Tokenizer(num_words=...)` (Keras) | Regla 11 | `gensim.utils.simple_preprocess` |
| `pad_sequences(...)` | Regla 11 | No existe en el stack del curso |
| `scaler.fit_transform(X_test)` | Regla 3 | `scaler.transform(X_test)` |
| `scaler.fit_transform(df)` antes del split | Regla 2 | Split primero, fit solo en train |
| `outlier_eliminator(df, ...)` sin split previo | Regla 2 | Split primero, luego outlier_eliminator(X_train) |
| `outlier_eliminator` usando Q1/Q3 como centro | DOC2 §3.1 | Usar MEDIA como centro |
| `shap.Explainer(...)` | Regla 11 | `feature_importances_` o RF auxiliar |
| `permutation_importance(...)` | Regla 11 | `feature_importances_` o RF auxiliar |
| `learn.validate()` sin eliminar callbacks | DOC1 §6 | `while len(learn.cbs)>0: learn.cbs.pop()` antes |
| `language_model_learner(...).fine_tune(...)` ejecutado | Regla 11 | DEJAR COMENTADO |
| `random_state=42` | Constante del curso | `random_state=33` |
| GridSearchCV con `X_test` como input | Regla 4 | Usar `X_train` en GridSearch |

---

## REFERENCIA RÁPIDA: CUÁNDO USAR CADA MÉTRICA

| Situación | Métrica principal | Justificación para el cuaderno |
|-----------|-------------------|-------------------------------|
| Dataset balanceado | Accuracy | "Las clases están equilibradas, accuracy es fiable." |
| Dataset desbalanceado, clasificación | F1-score weighted | "Hay desbalanceo, F1 penaliza errores en la clase minoritaria." |
| Diagnóstico médico, evitar falsos negativos | Recall | "Coste de falso negativo muy alto." |
| Sistema de alarma, evitar falsas alarmas | Precision | "Coste de falsa alarma alto." |
| Comparación global de modelos | ROC-AUC | "Independiente del umbral de clasificación." |
| Regresión | MAE o RMSE | "Interpretable en unidades de la variable objetivo." |
| Clustering | Inercia + método del codo | "Sin ground truth; validamos compacidad." |
| Visión, clasificación | Accuracy + confusion matrix | "Estándar del curso UD5." |
| Visión, segmentación | DiceMulti o accuracy (según enunciado) | "Dice: estándar en segmentación semántica." |
| PLN con desbalanceo | F1-score o ROC-AUC | "Robustos con texto desbalanceado." |

---

*Fin del DOC3 v3 — Reglas Críticas y Señales de Alarma*
