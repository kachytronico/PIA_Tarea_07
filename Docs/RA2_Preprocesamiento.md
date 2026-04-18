# DOCUMENTO RA2 — PREPROCESAMIENTO Y CLUSTERING
**Cuándo cargarlo:** hitos de split, limpieza, codificación, escalado, PCA, clustering.
**Instrucción para el agente:** Lee este documento completo antes de proponer cualquier celda de este hito. Las funciones de este documento son las únicas válidas para este curso. No las sustituyas por alternativas de sklearn aunque conozcas otras.

---

## REGLA FUNDAMENTAL — ORDEN DEL PIPELINE

```
1. Eliminar columnas irrelevantes / leakage  (puede ser antes del split)
2. SPLIT train/test                          (SIEMPRE antes de transformar)
3. outlier_eliminator                        (SOLO sobre train)
4. Tratamiento de nulos                      (fit en train, transform en test)
5. Codificación de categóricas               (fit en train, transform en test)
6. Escalado StandardScaler                   (fit en train, transform en test)
7. PCA (si el enunciado lo pide)             (fit en train, transform en test)
```

Nunca inviertas este orden. Si haces fit sobre el test → data leakage → penalización.

**REGLA SOBRE EL SPLIT — LEE ANTES DE DIVIDIR DATOS:**
Si el enunciado especifica exactamente los porcentajes del split (ej: "reserva el 10% para testeo", "divide en 80/20"), úsalos literalmente con un solo `train_test_split`. No añadas un conjunto de validación adicional salvo que el enunciado lo pida explícitamente. Con un único split train/test puedes usar GridSearchCV (que hace kfold interno) para optimizar modelos sin tocar el test. Añadir un val extra no solicitado cambia las proporciones del enunciado y es un error evaluable.

**REGLA SOBRE EL TEST — NUNCA SE LIMPIA:**
El test representa datos reales de producción que llegan tal cual. Solo se le aplican transformaciones aprendidas del train (transform, nunca fit). Está absolutamente prohibido:
- Eliminar filas del test (outliers, nulos, valores sin sentido)
- Filtrar rangos en el test
- Cualquier operación que cambie el número de filas del test
- fit_transform sobre el test de ningún tipo

Si el modelo recibe un dato atípico en producción, debe saberlo manejar. Limpiar el test falsea la evaluación real del modelo.

---

## DECISIÓN DE PATRÓN: HOLD-OUT vs KFOLD

```
¿El enunciado menciona literalmente "kfold", "k-fold",
"validación cruzada" o "k=N"?
├─ SÍ → Patrón B: kfold. El preprocesamiento (steps 3-7)
│        va DENTRO del bucle del fold sobre X_fold_train.
│        Ver sección "Patrón B" al final de este documento.
└─ NO → Patrón A: hold-out. SIEMPRE por defecto.
         No uses kfold por iniciativa propia aunque creas
         que es técnicamente mejor.
```

**REGLA DEL SPLIT LITERAL:**
Si el enunciado especifica exactamente los porcentajes del split,
úsalos sin modificar. Ejemplos:
- "reserva el 10% para testeo" → test_size=0.10, un solo split
- "divide en 80/20" → test_size=0.20, un solo split

No añadas un conjunto de validación separado salvo que el
enunciado lo pida explícitamente. Con un único split train/test
puedes usar GridSearchCV (que hace kfold interno) para optimizar
sin tocar el test. Añadir un val extra cambia las proporciones
y contradice el enunciado.

---

## PATRÓN A — HOLD-OUT (por defecto)

### Split

```python
from sklearn.model_selection import train_test_split

TARGET = "charges"  # CAMBIAR: nombre de la columna objetivo
random_seed = 33

# CAMBIAR test_size según lo que diga el enunciado (típico: 0.2)
train_df, test_df = train_test_split(df,
    test_size=0.2,
    stratify=df[TARGET] if es_clasificacion else None,
    random_state=random_seed)

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
```

### Outliers — función del curso (usa MEDIA, no mediana)

```python
# ESTA ES LA ÚNICA FUNCIÓN VÁLIDA PARA OUTLIERS EN ESTE CURSO
# Elimina FILAS, no recorta valores. NO uses clip() ni cap().
# Aplica SOLO sobre el conjunto de entrenamiento.
def outlier_eliminator(df, threshold=0.05):
    s_df = df.describe()
    for col in s_df.columns:
        n    = s_df.loc["count", col]
        mean = s_df.loc["mean", col]   # ← MEDIA, no mediana ni Q2
        iqr  = (s_df.loc["75%", col] - s_df.loc["25%", col]) * 1.5
        rng  = [mean - iqr, mean + iqr]
        out  = df[(df[col] < rng[0]) | (df[col] > rng[1])]
        print(f"Outliers en {col}: {len(out)}/{int(n)}.", end=" ")
        if len(out) != 0 and len(out) / n < threshold:
            print("Eliminados.")
            df = df[(df[col] > rng[0]) & (df[col] < rng[1])]
        else:
            print("Conservados (supera umbral o no hay).")
    return df

# USO: solo sobre X_train, nunca sobre X_test
X_train_clean = outlier_eliminator(X_train.select_dtypes(include="number"))
# Después de aplicar outlier_eliminator, sincronizar y_train con el índice resultante:
y_train = y_train.loc[X_train_clean.index]
X_train = X_train.loc[X_train_clean.index]
```

### Nulos

```python
# Detectar
print(X_train.isnull().sum())

# Estrategia 1: eliminación con máscara (patrón preferido del curso)
# X_train = X_train[X_train["col"].notnull()]  # CAMBIAR col

# Estrategia 2: imputación — FIT en train, TRANSFORM en test
from sklearn.impute import SimpleImputer
num_cols = X_train.select_dtypes(include="number").columns
imp = SimpleImputer(strategy="mean")           # "median" o "most_frequent" según el caso
X_train[num_cols] = imp.fit_transform(X_train[num_cols])  # fit en train
X_test[num_cols]  = imp.transform(X_test[num_cols])        # solo transform en test
```

### Codificación de categóricas

```python
# TIPO 1: Binaria (2 valores) → lambda
# X_train["sex"] = X_train["sex"].apply(lambda e: 0 if e=="male" else 1)
# X_test["sex"]  = X_test["sex"].apply(lambda e: 0 if e=="male" else 1)

# TIPO 2: Ordinal (tiene orden) → diccionario de mapeo
# mapeo = {"low": 1, "mid": 2, "high": 3}  # CAMBIAR
# X_train["col"] = X_train["col"].map(mapeo)
# X_test["col"]  = X_test["col"].map(mapeo)

# TIPO 3: Nominal sin valores nuevos en producción → binary_categorizer
import math
def binary_categorizer(dataframe, column, code_map=None, cols=None):
    result = []
    if not cols:
        cols = math.ceil(math.log2(len(dataframe[column].unique())))
    if not code_map:
        code_map = {v: k for k, v in enumerate(dataframe[column].unique())}
    for value in dataframe[column]:
        b = format(code_map[value], "b").rjust(cols, "0")
        result.append([int(x) for x in b])
    new_cols  = [f"{column}_{i}" for i in range(len(result[0]))]
    result_df = pd.DataFrame(result, index=dataframe.index, columns=new_cols)
    return dataframe.drop(columns=[column]).join(result_df), code_map

# Uso (guardar code_map para aplicar el mismo mapeo en test):
# X_train, code_map = binary_categorizer(X_train, "region")
# X_test,  _        = binary_categorizer(X_test,  "region", code_map=code_map)

# EXCEPCIÓN: si el enunciado dice que en producción pueden llegar valores
# nuevos no vistos en entrenamiento (ej: "cualquier región, incluso las no
# contempladas") → usar OneHotEncoder con handle_unknown="ignore":
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# enc.fit(X_train[["region"]])  # fit en train
# col_enc = enc.transform(X_train[["region"]])  # transform train
# col_enc_test = enc.transform(X_test[["region"]])  # transform test
```

### Escalado

```python
from sklearn.preprocessing import StandardScaler
# Obligatorio antes de: KNN, SVM, KMeans, DBSCAN, PCA, MLP
# NO necesario para: árboles de decisión, Random Forest

sc = StandardScaler()
num_cols = X_train.select_dtypes(include="number").columns
X_train_scaled = sc.fit_transform(X_train[num_cols])   # fit en train
X_test_scaled  = sc.transform(X_test[num_cols])         # solo transform
```

### PCA (solo si el enunciado lo pide)

```python
from sklearn.decomposition import PCA
# n_components como float → % de varianza a retener (ej: 0.95 = al menos 95%)
pca = PCA(n_components=0.95)  # CAMBIAR según enunciado
X_train_pca = pca.fit_transform(X_train_scaled)  # fit en train
X_test_pca  = pca.transform(X_test_scaled)

print(f"Componentes: {pca.n_components_} | Varianza: {pca.explained_variance_ratio_.sum()*100:.1f}%")
# Si n_components_ == n_features → threshold demasiado alto, ajustar
```

---

## PATRÓN B — KFOLD CON PREPROCESAMIENTO DENTRO DEL FOLD

Solo usar si el enunciado menciona kfold explícitamente.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score  # CAMBIAR por la métrica del enunciado

def kfold_pipeline(X_tv, y_tv, model, n_splits=5, use_pca=True, pca_var=0.95):
    """Imputer, scaler y PCA se ajustan DENTRO de cada fold — sin leakage."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    resultados = []
    for i, (tr, vl) in enumerate(skf.split(X_tv, y_tv)):
        Xtr, Xvl = X_tv.iloc[tr].copy(), X_tv.iloc[vl].copy()
        ytr, yvl = y_tv.iloc[tr],         y_tv.iloc[vl]
        num = Xtr.select_dtypes(include="number").columns
        imp = SimpleImputer(strategy="mean")
        Xtr[num] = imp.fit_transform(Xtr[num])  # fit SOLO en fold_train
        Xvl[num] = imp.transform(Xvl[num])
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr[num])       # fit SOLO en fold_train
        Xvl_s = sc.transform(Xvl[num])
        if use_pca:
            p = PCA(n_components=pca_var)
            Xtr_f, Xvl_f = p.fit_transform(Xtr_s), p.transform(Xvl_s)
        else:
            Xtr_f, Xvl_f = Xtr_s, Xvl_s
        model.fit(Xtr_f, ytr)
        s_tr = f1_score(ytr, model.predict(Xtr_f), average="weighted", zero_division=0)
        s_vl = f1_score(yvl, model.predict(Xvl_f), average="weighted", zero_division=0)
        resultados.append({"fold": i+1, "train": round(s_tr,4), "val": round(s_vl,4)})
        print(f"Fold {i+1}: train={s_tr:.3f} | val={s_vl:.3f}")
    df_r = pd.DataFrame(resultados)
    print(f"\nMedia val: {df_r['val'].mean():.3f} ± {df_r['val'].std():.3f}")
    if df_r["val"].std() > 0.10:
        print("⚠️ Alta varianza entre folds — revisar estratificación")
    return df_r
```

---

## CLUSTERING (UD3) — solo si no hay variable objetivo

### KMeans — método del codo obligatorio

```python
from sklearn.cluster import KMeans
# SIEMPRE escalar antes de KMeans

# Paso 1: método del codo (nunca elegir K arbitrariamente)
inercias = []
for k in range(1, 12):
    km = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    km.fit(X_scaled)
    inercias.append(km.inertia_)
plt.plot(range(1,12), inercias, "bo-")
plt.xlabel("K"); plt.ylabel("Inercia")
plt.title("Método del codo"); plt.show()
# Elegir K en el 'codo' de la curva

K = 4  # CAMBIAR según el gráfico
km = KMeans(n_clusters=K, random_state=random_seed, n_init=10)
df["cluster"] = km.fit_predict(X_scaled)
```

### DBSCAN

```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)  # CAMBIAR eps y min_samples
labels = dbscan.fit_predict(X_scaled)
print(f"Clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Outliers (etiqueta -1): {list(labels).count(-1)}")
```

---

## PROHIBICIONES EN ESTE RA

- **Nunca** eliminar filas del test (outliers, nulos, rangos). El test es sagrado e intocable.
- **Nunca** usar `.clip()` ni `.cap()` para outliers. Siempre `outlier_eliminator`.
- **Nunca** hacer `fit_transform` sobre X_test ni X_val. Solo `transform`.
- **Nunca** elegir K en KMeans sin el método del codo.
- **Nunca** usar kfold por iniciativa propia si el enunciado no lo menciona.

---

*Fin del documento RA2*
