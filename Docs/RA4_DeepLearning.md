# DOCUMENTO RA4 — DEEP LEARNING: VISIÓN Y PLN
**Cuándo cargarlo:** hitos de DataBlock, CNN, clasificador de texto, embeddings, evaluación deep, interpretabilidad.
**Instrucción para el agente:** Lee este documento completo antes de proponer cualquier celda de deep learning. El stack de este curso es FastAI, no TensorFlow ni Keras. Usa exactamente los patrones aquí descritos.

---

## DECISIÓN: ¿VISIÓN O PLN?

```
¿El problema trabaja con imágenes (ZIP con carpetas)?
└─ VISIÓN → fastai.vision.all + albumentations → ver Sección A

¿El problema trabaja con texto (CSV con columna de texto)?
├─ ¿Pide "entrena un embedding" o "Word2Vec" desde cero?
│   └─ gensim Word2Vec → vectores como features para sklearn SVM/RF
└─ ¿Pide clasificador deep o AWD_LSTM?
    └─ fastai.text.all → AWD_LSTM → ver Sección B
```

---

## SECCIÓN A — VISIÓN POR COMPUTADOR (FastAI)

### Setup obligatorio

```python
# Instalación — versión fijada OBLIGATORIA (sin fastprogress==1.0.3 hay bugs)
!pip install fastai -Uqqq
!pip install fastprogress==1.0.3

import gc, os, shutil, random
from pathlib import Path
from albumentations import *           # el curso usa albumentations, NO aug_transforms()
from fastai.vision.all import *
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

random_seed = 33
set_seed(random_seed)
random.seed(random_seed)
```

### Hito de fusión de carpetas (RA2 aplicado a visión)

```python
# SIEMPRE inspeccionar antes de escribir código de fusión
base_dir = Path("/content")   # CAMBIAR: donde se descomprimió el ZIP
print("Estructura:")
for item in sorted(base_dir.iterdir()):
    if item.is_dir():
        for sub in sorted(item.iterdir()):
            n = len(list(sub.glob("*.*"))) if sub.is_dir() else 0
            print(f"  {item.name}/{sub.name}/  ({n} archivos)")

# Fusionar subcarpetas en estructura limpia por clase
labels = ["clase_a", "clase_b"]   # CAMBIAR: clases del problema
dataset_limpio = Path("/content/dataset_limpio")

for label in labels:
    (dataset_limpio / label).mkdir(parents=True, exist_ok=True)
    src = base_dir / label
    if not src.exists(): continue
    subcarpetas = [d for d in src.iterdir() if d.is_dir()]
    for sub in (subcarpetas if subcarpetas else [src]):
        for img in sub.glob("*.*"):
            nuevo = f"{sub.name}_{img.name}" if subcarpetas else img.name
            shutil.copy(img, dataset_limpio / label / nuevo)
print("Fusión completada:", {l: len(list((dataset_limpio/l).glob("*.*"))) for l in labels})
```

### Hito de split train/test para imágenes

```python
test_pct = 0.25   # CAMBIAR según enunciado
for split in ["train", "test"]:
    for label in labels:
        (dataset_limpio / split / label).mkdir(parents=True, exist_ok=True)

for label in labels:
    src = dataset_limpio / label
    imgs = os.listdir(src); random.shuffle(imgs)
    idx  = int(len(imgs) * (1 - test_pct))
    for img in imgs[:idx]: shutil.move(str(src/img), str(dataset_limpio/"train"/label/img))
    for img in imgs[idx:]: shutil.move(str(src/img), str(dataset_limpio/"test"/label/img))
    src.rmdir()
    print(f"'{label}': train={len(os.listdir(dataset_limpio/'train'/label))}, "
          f"test={len(os.listdir(dataset_limpio/'test'/label))}")
```

### TransformPipeline — el curso NO usa aug_transforms()

```python
# Esta clase es OBLIGATORIA. No la sustituyas por aug_transforms() de FastAI.
class TransformPipeline(ItemTransform):
    def __init__(self, train=True):
        self.aug = Compose([VerticalFlip(p=0.2), GaussNoise(p=0.2),
                            HorizontalFlip(p=0.2), GridDistortion(p=0.2)]) \
                   if train else Compose([Resize(256, 256, p=1)])
    def encodes(self, x):
        if len(x) == 1: return x
        img, lbl = x
        return PILImage.create(self.aug(image=np.array(img))["image"]), lbl
```

### DataBlock + DataLoaders

```python
train_path = dataset_limpio / "train"   # CAMBIAR si la ruta es distinta

db = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.1, seed=random_seed),
    get_y=parent_label,
    item_tfms=[TransformPipeline(train=True)]
)
dls = db.dataloaders(train_path, bs=32)
dls.show_batch()
print("Vocab (clases detectadas):", dls.vocab)
# VERIFICAR: el vocab debe mostrar exactamente las clases del problema
```

### Entrenamiento — fine_tune, nunca fit desde cero

```python
# EarlyStoppingCallback SIEMPRE
callbacks = [EarlyStoppingCallback(patience=3)]

# vision_learner aplica transfer learning automáticamente
learn = vision_learner(dls, resnet18, cbs=callbacks, metrics=accuracy)
learn.fine_tune(3)   # CAMBIAR épocas (3-5 razonable en Colab)
learn.save("checkpoint_base")
```

### lr_find + Progressive Resizing

```python
# lr_find: elegir LR justo ANTES del mínimo (zona de máxima pendiente descendente)
learn_lr = vision_learner(dls, resnet18, metrics=accuracy)
learn_lr.lr_find()
lr_optimo = 1e-3   # CAMBIAR según el gráfico

# Progressive resizing: un DataBlock NUEVO por fase (no reutilizar el anterior)
for size, bs, lr_factor in [(128, 64, 1), (224, 32, 5), (384, 16, 20)]:
    db_fase = DataBlock(
        blocks=(ImageBlock, CategoryBlock), get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.1, seed=random_seed),
        get_y=parent_label, item_tfms=Resize(size))
    dls_fase = db_fase.dataloaders(train_path, bs=bs)
    if size == 128:
        learn_pr = vision_learner(dls_fase, resnet18,
            cbs=[EarlyStoppingCallback(patience=3)], metrics=accuracy)
    else:
        learn_pr.dls = dls_fase
        learn_pr.cbs = L([EarlyStoppingCallback(patience=3)])
    learn_pr.fine_tune(2, base_lr=lr_optimo / lr_factor)
    print(f"--- Fase {size}px completada ---")
learn_pr.save("checkpoint_progressive")
```

### Evaluación en test

```python
# DataBlock de test: sin aumentos, GrandparentSplitter apuntando a carpeta "test"
db_test = DataBlock(
    blocks=(ImageBlock, CategoryBlock), get_items=get_image_files,
    splitter=GrandparentSplitter(valid_name="test"),
    get_y=parent_label, item_tfms=[TransformPipeline(train=False)])
dl_test = db_test.dataloaders(dataset_limpio)

# VERIFICAR vocab antes de evaluar
print("Vocab train:", learn_pr.dls.vocab)
print("Vocab test: ", dl_test.vocab)
# Si son distintos → usar CategoryBlock(vocab=learn_pr.dls.vocab) en db_test

learn_pr.load("checkpoint_progressive")
learn_pr.dls = dl_test
while len(learn_pr.cbs) > 0: learn_pr.cbs.pop()   # OBLIGATORIO antes de validate()
resultados_test = learn_pr.validate()
print(f"Loss: {resultados_test[0]:.4f} | Accuracy: {resultados_test[1]:.4f}")

interp = ClassificationInterpretation.from_learner(learn_pr)
interp.plot_confusion_matrix(figsize=(6,6), dpi=60); plt.show()
```

### Interpretabilidad

```python
interp.plot_top_losses(6, figsize=(12, 8)); plt.show()
print(interp.most_confused(min_val=1))
# En la celda de conclusión: describir en qué imágenes falla el modelo y por qué
```

---

## SECCIÓN B — PLN Y TEXTO (FastAI)

### DECISIÓN DE STACK PARA PLN — LEER ANTES DE ESCRIBIR CUALQUIER CELDA

```
¿El enunciado pide "entrena un embedding", "crea un embedding propio"
o "Word2Vec"?
└─ USA: gensim Word2Vec → ver Opción B1
   PROHIBIDO: Keras Embedding, TensorFlow, torch.nn.Embedding
   El curso usa gensim para embeddings propios. Son hitos distintos.

¿El enunciado pide "modelo deep", "FastAI", "AWD_LSTM" o
"clasificador de textos con FastAI"?
└─ USA: fastai.text.all → DataBlock + text_classifier_learner → ver Opción B2
   PROHIBIDO: Keras Sequential, TensorFlow

¿El enunciado pide "fine-tune sobre modelo de lenguaje" u "obtener
el codificador"?
└─ USA: language_model_learner con AWD_LSTM → ver sección Language Model
   REGLA CRÍTICA: el entrenamiento (fine_tune) debe quedar COMENTADO
   y NUNCA ejecutado. Solo instanciar el learner y dejar el código
   comentado explicando que podría durar días.
   Si la celda empieza a ejecutar → CANCELA INMEDIATAMENTE (Kernel → Interrupt).
   Ejecutar aunque sea el 1% del entrenamiento es un error.
```

### Opción B1 — Word2Vec desde cero (si el enunciado pide "entrena un embedding")

```python
# Usar gensim para entrenar Word2Vec sobre los datos de entrenamiento
# NO uses modelos preentrenados (SentenceTransformers) si pide "entrenar desde cero"
!pip install gensim -Uqqq
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np

# Tokenizar (solo sobre train)
tokens_train = [simple_preprocess(texto) for texto in X_train["text"]]  # CAMBIAR col

# Entrenar embedding
w2v = Word2Vec(sentences=tokens_train, vector_size=100, window=5,
               min_count=1, workers=4, seed=random_seed)

# Vectorizar: media de los embeddings de cada palabra del texto
def vectorizar(tokens, model):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

X_train_emb = np.array([vectorizar(t, w2v) for t in tokens_train])
tokens_test  = [simple_preprocess(t) for t in X_test["text"]]  # CAMBIAR col
X_test_emb   = np.array([vectorizar(t, w2v) for t in tokens_test])

# Los vectores ya son features numéricas → usar con cualquier modelo sklearn (SVM, RF, etc.)
from sklearn.svm import SVC
svm_nlp = SVC(kernel="rbf", probability=True, random_state=random_seed)
svm_nlp.fit(X_train_emb, y_train)
```

### Opción B2 — Clasificador deep con AWD_LSTM (FastAI)

```python
!pip install fastai -Uqqq
from fastai.text.all import *
set_seed(random_seed)

# Downsampling si hay desbalanceo (clase mayor > 1.2x la menor)
counts = df["label"].value_counts()   # CAMBIAR col
if counts.iloc[0] > counts.iloc[1] * 1.2:
    min_n = counts.iloc[-1]
    df = df.groupby("label").apply(
        lambda x: x.sample(min(len(x), int(min_n*1.2)), random_state=random_seed)
    ).reset_index(drop=True)

# Columna "set" para el splitter de FastAI (False=train, True=val)
from sklearn.model_selection import train_test_split
df_tr, df_vl = train_test_split(df, test_size=0.2,
    stratify=df["label"], random_state=random_seed)   # CAMBIAR col
df_tr["set"] = False; df_vl["set"] = True
df_tv = pd.concat([df_tr, df_vl]).reset_index(drop=True)

# DataBlock de texto
db_txt = DataBlock(
    blocks=(TextBlock.from_df("text"), CategoryBlock),   # CAMBIAR "text"
    get_x=ColReader("text"), get_y=ColReader("label"),   # CAMBIAR cols
    splitter=ColSplitter("set"))
dls_txt = db_txt.dataloaders(df_tv, bs=64)   # bs=64 estándar del curso
dls_txt.show_batch()

# .to_fp16() OBLIGATORIO — reduce memoria GPU a la mitad
learner = text_classifier_learner(dls_txt, AWD_LSTM, metrics=accuracy).to_fp16()
learner.fine_tune(4, 1e-2)
```

### Language Model fine-tune (si el enunciado pide "obtener el codificador")

```python
# Paso previo al clasificador: fine-tune del language model
# Esto puede durar horas/días — dejarlo indicado como comentario si no da tiempo
lm_db = DataBlock(
    blocks=TextBlock.from_df("text", is_lm=True),   # CAMBIAR col
    get_x=ColReader("text"),
    splitter=RandomSplitter(0.1, seed=random_seed))
lm_dls = lm_db.dataloaders(df_tv, bs=64)

lm_learner = language_model_learner(lm_dls, AWD_LSTM, metrics=accuracy).to_fp16()
# lm_learner.fine_tune(1, 1e-2)   # Descomentar si hay tiempo (puede durar mucho)
# lm_learner.save_encoder("codificador_lm")
print("Language model configurado. Entrenamiento omitido por tiempo.")
print("En producción: descomenta fine_tune y save_encoder antes de entrenar el clasificador.")
```

---

## PROHIBICIONES EN ESTE RA

- **Nunca** usar Keras ni TensorFlow para embeddings o visión si el enunciado no lo pide. El stack del curso es gensim (embeddings propios) y FastAI (clasificadores deep).
- **Nunca** usar Keras `Embedding` layer cuando el enunciado pide "entrena un embedding propio". Usa gensim Word2Vec.
- **Nunca** usar `aug_transforms()` de FastAI. Siempre `TransformPipeline` con albumentations.
- **Nunca** omitir `fastprogress==1.0.3` en la instalación.
- **Nunca** llamar a `learn.validate()` sin haber eliminado todos los callbacks antes.
- **Nunca** usar `SentenceTransformers` si el enunciado pide "entrenar un embedding desde cero".
- **Nunca** omitir `.to_fp16()` en el learner de texto.
- **Nunca** ejecutar el fine_tune del language model si el enunciado dice que podría durar días. La celda de entrenamiento debe estar completamente comentada. Si empieza a ejecutar accidentalmente → Kernel → Interrupt inmediatamente.
- **Nunca** usar `TextDataLoaders.from_df` con columna `is_valid`. El patrón del curso usa `DataBlock` con `ColSplitter("set")` donde la columna "set" vale False en train y True en validación.

---

*Fin del documento RA4*
