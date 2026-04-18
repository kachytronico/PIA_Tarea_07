# DOC3 — REGLAS CRÍTICAS Y SEÑALES DE ALARMA (v2)
**Módulo:** Programación de Inteligencia Artificial (PIA)
**Propósito:** Las 10 reglas que no se pueden violar + checklists por tipo de problema + señales de alarma a detectar en los outputs antes de avanzar.
**Cómo usar:** El agente aplica este documento como **checklist activo**, no como referencia pasiva. Antes de cualquier entrenamiento, recorre el checklist correspondiente al tipo de problema.

---

## LAS 10 REGLAS INQUEBRANTABLES

Provienen del Decálogo oficial del curso PIA. Violarlas penaliza directamente la nota.

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

El split (train/val/test) debe ocurrir **antes** de cualquier transformación que aprenda la distribución de los datos: imputación, escalado, codificación ordinal, PCA.

**Excepción válida:** eliminar columnas completamente irrelevantes (IDs, nombres propios, columnas de texto libre) SÍ puede hacerse antes del split, porque no contienen distribución estadística que pueda contaminar. Lo que no puede hacerse antes del split: `SimpleImputer.fit()`, `StandardScaler.fit()`, `PCA.fit()`.

**Ejemplo del error:**
```python
# ❌ MAL: escalar ANTES de dividir → contamina test con estadísticos de todo el dataset
scaler.fit_transform(df)
X_train, X_test = train_test_split(...)

# ✅ BIEN: dividir PRIMERO, escalar solo sobre train
X_train, X_test = train_test_split(...)
X_train_s = scaler.fit_transform(X_train)  # aprende solo de train
X_test_s  = scaler.transform(X_test)       # aplica la misma escala
```

**En kfold:** el fit del scaler va dentro del bucle, sobre `X_fold_train`. Ver DOC2 → PATRÓN P1 COMPLETO.

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
# ✅ Split estratificado
X_tr, X_te, y_tr, y_te = train_test_split(X, y,
    test_size=0.2, stratify=y, random_state=random_seed)

# ✅ KFold estratificado
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
```

---

### REGLA 7 — THRESHOLD EN SELF-TRAINING
**Aplica a: RA3 | Momento: al usar SelfTrainingClassifier**

Solo añadir pseudoetiquetas cuando el modelo tiene alta confianza. Threshold mínimo recomendado: 0.9 (90%).

```python
st_model = SelfTrainingClassifier(base_model, threshold=0.9)  # no bajar de 0.8
```

---

### REGLA 8 — JUSTIFICACIÓN DEL K EN CLUSTERING
**Aplica a: RA2 (UD3) | Momento: antes de entrenar KMeans**

Nunca elegir K arbitrariamente. Mostrar siempre la gráfica del método del codo antes de instanciar `KMeans(n_clusters=K)`.

**Señal de alarma:** código que instancia `KMeans(n_clusters=3)` sin haber mostrado el codo previamente.

---

### REGLA 9 — PREVENCIÓN DE OVERFITTING EN DEEP LEARNING
**Aplica a: RA4 | Momento: antes y durante el entrenamiento FastAI**

```python
# Las tres medidas obligatorias del curso:
callbacks = [EarlyStoppingCallback(patience=3)]  # 1. EarlyStopping
learn = vision_learner(dls, resnet18, cbs=callbacks, metrics=accuracy)
learn.fine_tune(n)   # 2. Transfer learning (fine_tune, no fit desde cero)
# 3. Progressive resizing (ver Hito 5 en DOC2 → RA4)
```

**Señales de overfitting:** `train_loss` baja continuamente mientras `valid_loss` sube → EarlyStopping debe activarse.

---

### REGLA 10 — DOCUMENTACIÓN Y JUSTIFICACIÓN (CAJA BLANCA)
**Aplica a: todos los RAs | Momento: en cada celda relevante**

El código sin justificación técnica se evalúa a la baja. Incluir en comentarios o celdas Markdown:

```python
# Uso F1-score porque el dataset está desbalanceado (clase positiva = 15%)
# Escalo porque KNN es sensible a la magnitud de las variables
# Uso stratify=y para mantener la proporción de clases en train y test
# El threshold 0.9 garantiza solo pseudoetiquetas de alta confianza
# Uso fine_tune (transfer learning) para aprovechar pesos preentrenados en ImageNet
# Aplico downsampling (clase mayoritaria = 120% de la minoritaria) para evitar sesgo
```

---

## CHECKLISTS PRE-ENTRENAMIENTO POR TIPO DE PROBLEMA

Recorre el checklist del tipo de problema antes de ejecutar cualquier celda de entrenamiento.

---

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
SPLIT
☐ train_test_split ejecutado ANTES del preprocesamiento (Regla 2)
☐ stratify=y si hay desbalanceo (Regla 6)
☐ X e y separados DESPUÉS del split

PREPROCESAMIENTO
☐ fit_transform() solo sobre X_train_val (Regla 3)
☐ transform() sobre X_test (Regla 3)
☐ Escalado antes de KNN/SVM (Regla 5)
☐ Árbol/RF sin escalado (Regla 5)

MODELOS
☐ GridSearchCV usa solo X_train_val (no X_test) (Regla 4)
☐ X_test usado solo en la evaluación final (Regla 4)
☐ random_seed = 33

CÓDIGO
☐ Comentarios justificando las decisiones (Regla 10)
```

---

### ☑ CHECKLIST — Problema de visión (RA4 FastAI)

```
DATOS
☐ Estructura de carpetas inspeccionada con os.listdir() antes de fusionar
☐ Train/test split de imágenes completado (carpetas físicas separadas)
☐ dls.show_batch() ejecutado y etiquetas verificadas
☐ dls.vocab confirmado antes de entrenar

DATABLOCK
☐ TransformPipeline(train=True) para entrenamiento
☐ TransformPipeline(train=False) para test (sin aumentos)
☐ RandomSplitter(valid_pct=0.1, seed=33) para train/valid
☐ GrandparentSplitter(valid_name="test") para el DataBlock de test

ENTRENAMIENTO
☐ EarlyStoppingCallback configurado (Regla 9)
☐ vision_learner con fine_tune, no fit desde cero (Regla 9)
☐ random_seed = 33 con set_seed(33)
☐ fastprogress==1.0.3 instalado

EVALUACIÓN EN TEST
☐ Vocab de train == Vocab de test (ver señales de alarma abajo)
☐ TODOS los callbacks eliminados antes de learn.validate()
☐ Código: while len(learn.cbs) > 0: learn.cbs.pop()
☐ Matriz de confusión mostrada

CÓDIGO
☐ Comentarios sobre decisiones de arquitectura y transfer learning (Regla 10)
```

---

### ☑ CHECKLIST — Problema de texto/PLN (RA4 FastAI)

```
DATOS
☐ Desbalanceo verificado con value_counts()
☐ Downsampling aplicado si clase mayoritaria > 1.2 × clase minoritaria
☐ Columna "set" creada (False=train, True=val) antes del DataBlock
☐ dls_texto.show_batch() ejecutado (tokens xxbos, xxunk son normales)

ENTRENAMIENTO
☐ .to_fp16() añadido al learner (reduce memoria GPU)
☐ bs=64 (estándar del curso para texto)
☐ random_seed = 33 con set_seed(33)

EVALUACIÓN
☐ get_preds(reorder=False) para obtener predicciones
☐ classification_report con target_names=dls_texto.vocab
☐ F1-score como métrica principal si hay desbalanceo

CÓDIGO
☐ Justificación del downsampling documentada (Regla 10)
```

---

## SEÑALES DE ALARMA EN LOS OUTPUTS

### En datos tabulares (RA2/RA3)

| Lo que ves | Alarma | Acción inmediata |
|-----------|--------|-----------------|
| F1 o accuracy = 1.0 exacto | Data leakage | PARAR. Regla 1. Revisar columnas con crosstab. |
| `nan` en métricas | Nulos en X_train | `X_train.isnull().sum()` + revisar imputación |
| Accuracy test > accuracy train | Split mal hecho | Verificar que X_train y X_test son correctos |
| F1 = 0.0 o accuracy < 0.5 (binario) | Clases codificadas al revés | `y_train.value_counts()` + revisar codificación |
| `UserWarning: The least populated class` | Clase demasiado pequeña para kfold | Usar `StratifiedKFold` con `shuffle=True` |
| PCA devuelve mismo nº columnas que entrada | `n_components` inadecuado | Verificar `pca.explained_variance_ratio_.sum()` |
| F1 varía mucho entre folds (std > 0.1) | Dataset inestable o mal estratificado | Verificar distribución y añadir `stratify` |
| GridSearchCV tarda eternidades | `param_grid` demasiado grande | Reducir rangos de hiperparámetros |
| `NameError` en variable | Celdas ejecutadas fuera de orden | Reiniciar kernel y ejecutar en secuencia |
| `KeyError: 'columna'` | Nombre distinto al esperado | `df.columns.tolist()` |

### En visión / FastAI (RA4)

| Lo que ves | Alarma | Acción inmediata |
|-----------|--------|-----------------|
| `dls.show_batch()` con etiquetas incorrectas | `get_y` o carpetas mal nombradas | Verificar estructura y `parent_label` |
| `valid_loss` = NaN desde la primera época | Error DataBlock o imágenes corruptas | Verificar carga con `dls.show_batch()` |
| Accuracy test = 0.5 (binario) | El modelo no aprende nada | Vocab mismatch — ver fix abajo |
| `AttributeError` en `learn.validate()` | Callbacks no eliminados | `while len(learn.cbs)>0: learn.cbs.pop()` |
| `CUDA out of memory` | Batch o modelo demasiado grande | `bs=16`, usar resnet18 |
| train_loss baja, valid_loss sube | Overfitting en progreso | Regla 9. EarlyStopping debe activarse. |
| Progressive resizing no mejora el accuracy | DataBlock de fase 2/3 reutiliza Resize incorrecto | Crear DataBlock NUEVO por fase (ver DOC2 Hito 5) |
| `dls.vocab` ≠ `dl_test.vocab` | Clases del DataBlock de test distintas | Fix: ver abajo |

```python
# FIX — Vocab mismatch entre train y test en FastAI
# Síntoma: accuracy = 0.5 en binario o matriz de confusión con clases desordenadas
print("Vocab train:", learn.dls.vocab)
print("Vocab test: ", dl_test.vocab)

# Fix: forzar el mismo vocab en el DataBlock de test
vocab_train = learn.dls.vocab  # guardar el vocab del entrenamiento

db_test_fixed = DataBlock(
    blocks=(ImageBlock, CategoryBlock(vocab=vocab_train)),  # ← vocab forzado
    get_items=get_image_files,
    splitter=GrandparentSplitter(valid_name="test"),
    get_y=parent_label,
    item_tfms=[TransformPipeline(train=False)]
)
dl_test_fixed = db_test_fixed.dataloaders(dataset_limpio)
print("Vocab test corregido:", dl_test_fixed.vocab)
learn.dls = dl_test_fixed
while len(learn.cbs) > 0: learn.cbs.pop()
learn.validate()
```

### En PLN / texto (RA4)

| Lo que ves | Alarma | Acción inmediata |
|-----------|--------|-----------------|
| Accuracy > 90% desde la primera época | Espejismo del accuracy por desbalanceo | Verificar distribución. Aplicar downsampling. |
| `show_batch()` muestra casi todo `xxunk` | Texto vacío o vocabulario muy pequeño | Verificar columna de texto no está vacía |
| `OOM` durante entrenamiento | Secuencias largas o bs grande | `bs=32` y/o truncar textos |
| F1 muy bajo solo en clase minoritaria | Desbalanceo no tratado | Downsampling (clase mayor = 120% de la menor) |

---

## REFERENCIA RÁPIDA: CUÁNDO USAR CADA MÉTRICA

| Situación | Métrica principal | Justificación para el cuaderno |
|-----------|-------------------|-------------------------------|
| Dataset balanceado | Accuracy | "Las clases están equilibradas, accuracy es fiable." |
| Dataset desbalanceado, clasificación | F1-score weighted | "Hay desbalanceo, F1 penaliza errores en la clase minoritaria." |
| Diagnóstico médico, evitar falsos negativos | Recall | "El coste de un falso negativo (no detectar la enfermedad) es muy alto." |
| Sistema de alarma, evitar falsas alarmas | Precision | "El coste de una falsa alarma (dispatch innecesario) es alto." |
| Comparación global de modelos | ROC-AUC | "Es independiente del umbral de clasificación." |
| Regresión | MAE o RMSE | "Interpretable en las mismas unidades que la variable objetivo." |
| Clustering | Inercia + método del codo | "No hay ground truth; validamos la compacidad de los clusters." |
| Visión, clasificación de imágenes | Accuracy + confusion matrix | "Estándar del curso UD5; la matriz muestra qué clases confunde el modelo." |
| PLN con desbalanceo | F1-score + ROC-AUC | "El dataset de texto está desbalanceado; F1 y AUC son más robustos." |

---

*Fin del DOC3 v2 — Reglas Críticas y Señales de Alarma*
