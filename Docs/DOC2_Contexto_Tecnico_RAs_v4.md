# DOC2 — CONTEXTO TÉCNICO DEL CURSO PIA (v4)
**Propósito:** Este documento NO es un libro de recetas. Solo contiene lo que este curso hace diferente al ML estándar. Para todo lo demás usa tu conocimiento general de Python y sklearn. Las negaciones explícitas de este documento tienen prioridad sobre tu conocimiento general previo.

---

## 1. ÁRBOL DE DECISIÓN — IDENTIFICA EL PROBLEMA Y EL PATRÓN

### 1.1 Tipo de dato y tarea

```
¿Qué tipo de dato?
│
├─ TABULAR (CSV/DataFrame)
│   ├─ ¿Variable objetivo continua (precio, salario, coste, temperatura)?
│   │   └─ REGRESIÓN → métrica RMSE o MAE → usar ...Regressor, no ...Classifier
│   ├─ ¿Variable objetivo categórica (clase, etiqueta, sí/no)?
│   │   └─ CLASIFICACIÓN → F1 si hay desbalanceo, accuracy si está balanceado
│   └─ ¿Sin variable objetivo? → CLUSTERING → KMeans con método del codo obligatorio
│
├─ IMÁGENES (ZIP con carpetas de clases)
│   └─ FastAI visión → ver Sección 4
│
└─ TEXTO (CSV con columna de texto)
    ├─ ¿Pide "entrena un embedding" o "Word2Vec" desde cero?
    │   └─ gensim Word2Vec → vectores como features para sklearn
    └─ ¿Pide modelo deep o clasificador de texto con FastAI?
        └─ fastai.text.all → AWD_LSTM → ver Sección 5
```

### 1.2 Patrón de validación — REGLA ESTRICTA

```
¿El enunciado menciona literalmente "kfold", "k-fold",
"validación cruzada" o "k=N"?
│
├─ SÍ → Patrón B (kfold)
│   └─ Split 80/20 test primero. Kfold sobre el 80%.
│      Imputer + scaler + PCA se ajustan DENTRO de cada fold.
│      Ver Sección 3.3.
│
└─ NO → Patrón A (hold-out) — SIEMPRE por defecto
    └─ No uses kfold por iniciativa propia aunque creas que
       es mejor opción técnica. El curso evalúa que sigas
       el enunciado, no que optimices la validación.
       train_test_split estándar. Fit en train, transform en test.
```

### 1.3 RAs del curso — EXACTAMENTE ESTOS CUATRO, NO MÁS

| RA | Unidades | Contenido |
|----|----------|-----------|
| RA1 | UD1 | Python, NumPy, Pandas, AED |
| RA2 | UD2 + UD3 | Preprocesamiento tabular + clustering |
| RA3 | UD4 | Modelos supervisados + ensemble + semisupervisado |
| RA4 | UD5 + UD6 | Deep learning visión + PLN |

No existen RA5, RA6 ni ningún otro. No inventes RAs.

---

## 2. CONSTANTES UNIVERSALES DEL CURSO

```python
random_seed = 33   # SIEMPRE 33. Nunca 42 ni ningún otro valor.

# Imports base tabular
import math, numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
```

---

## 3. DEVIACIONES DEL CURSO RESPECTO AL ML ESTÁNDAR (tabular)

### 3.1 Outlier eliminator — función propia del curso

```python
# El curso usa la MEDIA como centro del IQR, no la mediana.
# Solo elimina si los outliers son < 5% del total.
# Aplicar SOLO sobre el conjunto de entrenamiento.
def outlier_eliminator(df, threshold=0.05):
    s_df = df.describe()
    for col in s_df.columns:
        n    = s_df.loc["count", col]
        mean = s_df.loc["mean", col]        # ← MEDIA, no mediana
        iqr  = (s_df.loc["75%", col] - s_df.loc["25%", col]) * 1.5
        rng  = [mean - iqr, mean + iqr]
        out  = df[(df[col] < rng[0]) | (df[col] > rng[1])]
        if len(out) != 0 and len(out) / n < threshold:
            df = df[(df[col] > rng[0]) & (df[col] < rng[1])]
    return df
```

### 3.2 Codificación de categóricas — el curso NO usa OneHotEncoder por defecto

```python
# TIPO 1: Binaria (2 valores) → lambda
df["sex"] = df["sex"].apply(lambda e: 0 if e == "male" else 1)  # CAMBIAR

# TIPO 2: Ordinal (orden importa) → diccionario de mapeo
mapeo = {"low": 1, "mid": 2, "high": 3}   # CAMBIAR
df["col"] = df["col"].apply(lambda e: mapeo.get(e, e))

# TIPO 3: Nominal sin valores nuevos en producción → binary_categorizer del curso
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
# Uso: df_train, code_map = binary_categorizer(df_train, "col")
#       df_test,  _        = binary_categorizer(df_test,  "col", code_map=code_map)

# EXCEPCIÓN — si el enunciado dice que en producción pueden llegar valores
# de una categoría nunca vista en entrenamiento → usar OneHotEncoder:
# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# enc.fit(X_train[["col"]])
# Señal en el enunciado: "cualquier X, incluso los no contemplados en entrenamiento"
```

### 3.3 Kfold con preprocesamiento dentro del fold (Patrón B)

Solo usar si el enunciado menciona kfold explícitamente. Ver regla 1.2.

```python
from sklearn.metrics import f1_score   # CAMBIAR métrica según enunciado

def kfold_pipeline(X_tv, y_tv, model, n_splits=5, use_pca=True, pca_var=0.95):
    """Preprocesamiento DENTRO del fold — evita data leakage entre folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    resultados = []
    for i, (tr, vl) in enumerate(skf.split(X_tv, y_tv)):
        Xtr, Xvl = X_tv.iloc[tr].copy(), X_tv.iloc[vl].copy()
        ytr, yvl = y_tv.iloc[tr],         y_tv.iloc[vl]

        num = Xtr.select_dtypes(include="number").columns
        imp = SimpleImputer(strategy="mean")
        Xtr[num] = imp.fit_transform(Xtr[num])   # fit SOLO en Xtr
        Xvl[num] = imp.transform(Xvl[num])

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr[num])        # fit SOLO en Xtr
        Xvl_s = sc.transform(Xvl[num])

        if use_pca:
            p = PCA(n_components=pca_var)
            Xtr_f, Xvl_f = p.fit_transform(Xtr_s), p.transform(Xvl_s)
        else:
            Xtr_f, Xvl_f = Xtr_s, Xvl_s

        model.fit(Xtr_f, ytr)
        score_tr = f1_score(ytr, model.predict(Xtr_f), average="weighted", zero_division=0)
        score_vl = f1_score(yvl, model.predict(Xvl_f), average="weighted", zero_division=0)
        resultados.append({"fold": i+1, "train": round(score_tr,4), "val": round(score_vl,4)})
        print(f"Fold {i+1}: train={score_tr:.3f} | val={score_vl:.3f}")

    df_r = pd.DataFrame(resultados)
    print(f"\nMedia val: {df_r['val'].mean():.3f} ± {df_r['val'].std():.3f}")
    if df_r["val"].std() > 0.10:
        print("⚠️  Alta varianza entre folds — revisar estratificación o dataset")
    return df_r
```

### 3.4 Red neuronal tabular — el curso usa sklearn, NO TensorFlow ni Keras

```python
# Para problemas tabulares (clasificación o regresión), el curso usa MLPClassifier
# o MLPRegressor de sklearn, no TensorFlow, no Keras, no PyTorch.
# Solo usar FastAI/TensorFlow si el enunciado lo pide explícitamente.
from sklearn.neural_network import MLPClassifier   # clasificación
from sklearn.neural_network import MLPRegressor    # regresión
```

### 3.5 Importancia de variables — el curso usa feature_importances_, NO shap

```python
# El curso usa el atributo nativo del modelo. No uses shap, LIME,
# permutation_importance ni ninguna librería externa de XAI.

# Para modelos con feature_importances_ (RandomForest, árboles, GradientBoosting):
importances = pd.Series(model.feature_importances_,
    index=X_train.columns).sort_values(ascending=False)
print(importances)

# Si el modelo no tiene feature_importances_ (KNN, SVM, red neuronal),
# entrena un RandomForest auxiliar SOLO para obtener la importancia:
from sklearn.ensemble import RandomForestRegressor   # o Classifier según la tarea
rf_aux = RandomForestRegressor(n_estimators=100, random_state=random_seed)
rf_aux.fit(X_train, y_train)
importances = pd.Series(rf_aux.feature_importances_, index=X_train.columns)
importances.sort_values(ascending=False).plot(kind="bar")
plt.title("Importancia de variables"); plt.tight_layout(); plt.show()
```

### 3.6 Ensemble tabular — el curso promedia manualmente, NO usa VotingRegressor

```python
# Para regresión: promedio simple de predicciones de todos los modelos.
# No uses VotingRegressor ni StackingRegressor salvo que el enunciado lo pida.
pred_knn   = model_knn.predict(X_test)
pred_arbol = model_arbol.predict(X_test)
pred_mlp   = model_mlp.predict(X_test)

pred_ensemble = (pred_knn + pred_arbol + pred_mlp) / 3   # promedio simple

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, pred_ensemble, squared=False)
print(f"RMSE ensemble: {rmse:.4f}")

# Para clasificación: ensemble OR (si cualquier modelo predice clase positiva → positivo)
# Ver lógica OR solo si el enunciado pide este criterio explícitamente.
```

---

## 4. FASTAI VISIÓN — deviaciones respecto al estándar

```python
# Instalación con versión fijada — OBLIGATORIO
!pip install fastai -Uqqq
!pip install fastprogress==1.0.3   # sin esto hay bugs de progreso

from albumentations import *        # el curso usa albumentations, NO aug_transforms()
from fastai.vision.all import *
set_seed(random_seed)

# TransformPipeline: el curso NO usa aug_transforms() de FastAI
class TransformPipeline(ItemTransform):
    def __init__(self, train=True):
        self.aug = Compose([VerticalFlip(p=0.2), GaussNoise(p=0.2),
                            HorizontalFlip(p=0.2), GridDistortion(p=0.2)]) if train \
                   else Compose([Resize(256, 256, p=1)])
    def encodes(self, x):
        if len(x) == 1: return x
        img, lbl = x
        return PILImage.create(self.aug(image=np.array(img))["image"]), lbl

# DataBlock mínimo
db = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.1, seed=random_seed),
    get_y=parent_label,
    item_tfms=[TransformPipeline(train=True)]
)
dls = db.dataloaders(train_path, bs=32)
dls.show_batch(); print("Vocab:", dls.vocab)

# fine_tune (transfer learning) — NUNCA fit desde cero
learn = vision_learner(dls, resnet18,
    cbs=[EarlyStoppingCallback(patience=3)], metrics=accuracy)
learn.fine_tune(3)

# Antes de evaluate en test: eliminar TODOS los callbacks
learn.dls = dl_test
while len(learn.cbs) > 0: learn.cbs.pop()
learn.validate()
```

---

## 5. FASTAI TEXTO — deviaciones respecto al estándar

```python
!pip install fastai -Uqqq
from fastai.text.all import *
set_seed(random_seed)

# El curso usa columna "set" (False=train, True=val) como splitter
# El DataBlock usa TextBlock.from_df y ColSplitter — no ImageDataLoaders
db_txt = DataBlock(
    blocks=(TextBlock.from_df("text"), CategoryBlock),  # CAMBIAR "text"
    get_x=ColReader("text"),
    get_y=ColReader("label"),
    splitter=ColSplitter("set")   # columna "set": False=train, True=val
)
dls_txt = db_txt.dataloaders(df_tv, bs=64)   # bs=64 estándar del curso

# .to_fp16() es OBLIGATORIO — reduce memoria GPU a la mitad
learner = text_classifier_learner(dls_txt, AWD_LSTM, metrics=accuracy).to_fp16()
learner.fine_tune(4, 1e-2)

# Downsampling si hay desbalanceo: limitar clase mayor al 120% de la menor
```

---

*Fin del DOC2 v4*
