# PIA UD7 — PROBLEMA 1: Células malignas

**Módulo:** Programación de Inteligencia Artificial  
**Unidad:** 7 — Proyecto transversal  
**Problema:** 1 — Células malignas  
**Alumno:** [TU NOMBRE]  
**Entorno:** Google Colab (CPU suficiente)

---

## 1. Enunciado del problema

Un laboratorio biomédico quiere automatizar el diagnóstico preliminar de anomalías en células a partir de características numéricas obtenidas de imágenes y datos clínicos.

- **Objetivo:** predecir si una célula es anómala (`anomaly_label`) y analizar qué factores influyen en esa predicción.  
- **Tipo de problema:** clasificación supervisada binaria.  
- **Métrica principal:** F1-score (importa mucho equilibrar precisión y recall porque hay posibles desbalances y falsos negativos costosos).  
- **Dataset:** fichero tabular incluido en `P1.zip` (ruta: PIA > 2026 > PROY > P1.zip).

El conjunto de datos puede contener valores atípicos (outliers) en las mediciones, y esas mismas anomalías se verán en producción.

---

## 2. Hitos oficiales del PROBLEMA 1

Según el enunciado, el problema se divide en 8 hitos, cada uno asociado a resultados de aprendizaje RA1–RA3.

1. **Hito 1 (RA1) — AED y análisis inicial**  
   - Distribución de `anomaly_label`.  
   - Tipos de datos.  
   - Detección de datos atípicos (outliers).  
   - Análisis de correlación.

2. **Hito 2 (RA2) — División train/test + K-Fold**  
   - Reservar el 20 % de los datos para test.  
   - Aplicar K-Fold con k = 5 sobre el train en el resto de hitos.

3. **Hito 3 (RA2) — Limpieza, nulos y codificación**  
   - Eliminar columnas innecesarias o perjudiciales.  
   - Tratar valores nulos.  
   - Codificar columnas categóricas.

4. **Hito 4 (RA2) — Estandarización y reducción de dimensionalidad**  
   - Estandarizar los datos.  
   - Aplicar reducción de dimensionalidad que explique al menos el 95 % de la varianza (por ejemplo, PCA).

5. **Hito 5 (RA3) — Modelo KNN**  
   - Crear un modelo KNN que resuelva el problema.  
   - Optimizarlo usando F1-score como métrica.

6. **Hito 6 (RA3) — Dos modelos adicionales**  
   - Entrenar dos modelos adicionales a elección.  
   - Optimizar cada uno usando técnicas diferentes.

7. **Hito 7 (RA3) — Ensemble de modelos**  
   - Crear un ensemble usando los modelos anteriores.  
   - Regla: “Si un modelo determina que la célula es maligna, entonces el resultado será ‘maligna’; será ‘benigna’ en caso contrario”.

8. **Hito 8 (RA3) — Patrones de cribado (> 80 % de fiabilidad)**  
   - Determinar si existe algún patrón sencillo, con fiabilidad > 80 %, que permita cribar las células malignas.

---

## 3. Estructura recomendada del notebook de PROBLEMA 1

El cuaderno `PIA_07_P1_Celulas_Malignas.ipynb` seguirá esta estructura:

### 3.1. Portada y contexto

- Celda Markdown: título, datos del alumno, descripción breve del problema, métrica principal (F1-score), mención a RA1–RA3.  
- Celda de código: imports básicos (pandas, numpy, matplotlib, seaborn, sklearn, etc.).

### 3.2. Carga de datos y visión general

- Cargar el dataset desde la ruta correspondiente en Drive.  
- Mostrar `df.head()`, `df.info()`, `df.describe()`.  
- Celda Markdown de conclusión con tamaño del dataset, variables clave y confirmación de que `anomaly_label` es la etiqueta.

---

## 4. Detalle por hito (P1)

### Hito 1 — AED y análisis de `anomaly_label`

Objetivo: entender el dataset antes de modelar.

- Distribución de `anomaly_label`.  
- Tipos de datos.  
- Outliers en variables numéricas.  
- Matriz de correlación y heatmap.

### Hito 2 — Split 80/20 + K-Fold (k=5)

- `train_test_split` con `test_size=0.2` y `stratify=y`.  
- Definir `KFold` o `StratifiedKFold` con `n_splits=5`.

### Hito 3 — Limpieza, nulos y categóricas

- Eliminar columnas irrelevantes.  
- Tratar nulos (imputación o eliminación).  
- Codificar categóricas (One-Hot, LabelEncoder, etc.).

### Hito 4 — Escalado y reducción de dimensionalidad

- `StandardScaler` para variables numéricas.  
- `PCA(n_components=0.95)` u otra técnica similar.

### Hito 5 — Modelo KNN

- Pipeline con escalado + PCA + KNN.  
- GridSearchCV/RandomizedSearchCV con F1 como métrica.

### Hito 6 — Dos modelos adicionales

- Entrenar dos modelos adicionales (por ejemplo, LogisticRegression, RandomForest).  
- Usar técnicas de optimización distintas.

### Hito 7 — Ensemble con regla OR

- Combinar predicciones de los modelos usando la regla OR sobre la clase maligna.  
- Comparar F1 del ensemble con los modelos individuales.

### Hito 8 — Patrón de cribado (> 80 %)

- Buscar reglas sencillas (árboles poco profundos, umbrales) con fiabilidad > 80 %.

---

## 5. Variables clave sugeridas

- `df`, `X`, `y`  
- `X_train`, `X_test`, `y_train`, `y_test`  
- `cv` (K-Fold/StratifiedKFold)  
- `modelo_knn`, `modelo_1`, `modelo_2`  
- `f1_knn`, `f1_modelo_1`, `f1_modelo_2`, `f1_ensemble`
