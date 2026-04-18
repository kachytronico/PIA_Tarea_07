# DOCUMENTO RA3 — MODELOS SUPERVISADOS Y ENSEMBLE
**Cuándo cargarlo:** hitos de modelos (KNN, árbol, SVM, red neuronal, ensemble, semisupervisado, fiabilidad).
**Instrucción para el agente:** Lee este documento completo antes de proponer cualquier modelo. Las decisiones de implementación aquí descritas tienen prioridad sobre tu conocimiento general de sklearn.

---

## IMPORTS OBLIGATORIOS RA3

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error,
    confusion_matrix, ConfusionMatrixDisplay
)
import numpy as np, pandas as pd, matplotlib.pyplot as plt

random_seed = 33
```

---

## DECISIÓN DE MÉTRICA

```
¿Es regresión? (variable objetivo continua: precio, temperatura, coste)
└─ RMSE o MAE — nunca accuracy ni F1

¿Es clasificación con desbalanceo? (clase minoritaria < 30%)
└─ F1-score weighted — nunca accuracy sola

¿Es clasificación balanceada?
└─ Accuracy — también mostrar F1 como comprobación
```

---

## DECISIÓN: ¿ESCALAR O NO?

```
KNN       → SÍ escalar (sensible a magnitudes)
SVM       → SÍ escalar
MLP       → SÍ escalar
KMeans    → SÍ escalar
DBSCAN    → SÍ escalar
Árbol     → NO escalar
Random Forest → NO escalar
```

---

## RED NEURONAL TABULAR — sklearn, NO TensorFlow ni Keras

```python
# Para tabular siempre MLPRegressor o MLPClassifier de sklearn
# Solo usar FastAI/TensorFlow si el enunciado lo pide explícitamente

# Regresión
mlp_reg = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=random_seed
)
mlp_reg.fit(X_train_scaled, y_train)
pred = mlp_reg.predict(X_test_scaled)
rmse = mean_squared_error(y_test, pred, squared=False)
print(f"RMSE MLP: {rmse:.4f}")

# Clasificación
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=random_seed
)
mlp_clf.fit(X_train_scaled, y_train)
print(f"F1 MLP: {f1_score(y_test, mlp_clf.predict(X_test_scaled), average='weighted'):.4f}")
```

---

## OPTIMIZACIÓN CON GRIDSEARCHCV

```python
# GridSearchCV evalúa solo sobre X_train — nunca toca X_test
param_grid = {
    "n_neighbors": range(1, 20, 2),      # CAMBIAR según el modelo
    "weights": ["uniform", "distance"]
}
grid = GridSearchCV(KNeighborsRegressor(),   # CAMBIAR modelo
    param_grid, cv=5, scoring="neg_root_mean_squared_error",  # CAMBIAR scoring
    n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print("Mejores parámetros:", grid.best_params_)
best_model = grid.best_estimator_

# Scoring según métrica del enunciado:
# Regresión RMSE → "neg_root_mean_squared_error"
# Regresión MAE  → "neg_mean_absolute_error"
# Clasificación F1 → "f1_weighted"
# Clasificación accuracy → "accuracy"
```

---

## ENSEMBLE TABULAR — promedio manual, NO VotingRegressor

```python
# El curso usa promedio manual de predicciones, no VotingRegressor
# No uses VotingRegressor ni StackingRegressor salvo que el enunciado lo pida

# Para REGRESIÓN: promedio de predicciones
pred_knn   = best_knn.predict(X_test_scaled)
pred_arbol = best_arbol.predict(X_test)       # árboles no necesitan escalado
pred_mlp   = best_mlp.predict(X_test_scaled)

pred_ensemble = (pred_knn + pred_arbol + pred_mlp) / 3
rmse_ens = mean_squared_error(y_test, pred_ensemble, squared=False)
print(f"RMSE Ensemble: {rmse_ens:.4f}")

# Para CLASIFICACIÓN con criterio OR (si cualquier modelo predice positivo → positivo):
CLASE_POSITIVA = 1   # CAMBIAR según el enunciado
pred_ensemble_clf = []
for i in range(len(X_test)):
    votos = [pred_knn[i], pred_arbol[i], pred_mlp[i]]
    pred_ensemble_clf.append(CLASE_POSITIVA if CLASE_POSITIVA in votos else votos[0])
pred_ensemble_clf = np.array(pred_ensemble_clf)
print(f"F1 Ensemble OR: {f1_score(y_test, pred_ensemble_clf, average='weighted'):.4f}")
```

---

## IMPORTANCIA DE VARIABLES — feature_importances_, NO shap

```python
# El curso usa el atributo nativo del modelo
# NUNCA uses shap, LIME, permutation_importance ni ninguna librería externa de XAI

# Para modelos con feature_importances_ (árboles, RF):
importancias = pd.Series(
    best_rf.feature_importances_,
    index=X_train.columns          # CAMBIAR si usaste escalado/PCA: usar X_train_scaled
).sort_values(ascending=False)
importancias.plot(kind="bar", figsize=(10, 4))
plt.title("Importancia de variables"); plt.tight_layout(); plt.show()
print(importancias)

# Si el modelo NO tiene feature_importances_ (KNN, SVM, MLP):
# Entrena un RF auxiliar SOLO para obtener importancias
from sklearn.ensemble import RandomForestRegressor   # o Classifier
rf_aux = RandomForestRegressor(n_estimators=100, random_state=random_seed)
rf_aux.fit(X_train, y_train)   # sin escalar para interpretar en espacio original
importancias_aux = pd.Series(rf_aux.feature_importances_, index=X_train.columns)
importancias_aux.sort_values(ascending=False).plot(kind="bar")
plt.title("Importancia (RF auxiliar)"); plt.show()

# Si el enunciado pide que una variable NO sea la más importante:
print("Top 3 variables más importantes:")
print(importancias_aux.sort_values(ascending=False).head(3))
# Si la variable prohibida aparece en top 2: documentarlo y justificar
# qué acción tomas (ej: eliminarla y reentrenar, o añadir penalización)
```

---

## EVALUACIÓN FINAL EN TEST

```python
# Evaluar en test UNA SOLA VEZ al final, con el mejor modelo ya elegido

# Para regresión:
pred_final = best_model.predict(X_test_scaled)  # CAMBIAR: usar escalado si aplica
rmse_final = mean_squared_error(y_test, pred_final, squared=False)
mae_final  = mean_absolute_error(y_test, pred_final)
print(f"RMSE final: {rmse_final:.4f}")
print(f"MAE final:  {mae_final:.4f}")

# Para clasificación:
pred_final = best_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, pred_final):.4f}")
print(f"F1:       {f1_score(y_test, pred_final, average='weighted'):.4f}")
cm = confusion_matrix(y_test, pred_final)
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.title("Matriz de confusión — Test"); plt.show()
```

---

## SEMISUPERVISADO (si el enunciado lo pide)

```python
from sklearn.semi_supervised import SelfTrainingClassifier

# Marcar datos sin etiquetar con -1
y_semi = y_train.copy()
idx_unlab = y_semi.sample(frac=0.7, random_state=random_seed).index
y_semi.loc[idx_unlab] = -1

# Threshold alto = pseudoetiquetas de alta confianza (mínimo 0.8)
base = KNeighborsClassifier(n_neighbors=5)
st   = SelfTrainingClassifier(base, threshold=0.9)
st.fit(X_train_scaled, y_semi)
print(f"F1 semi: {f1_score(y_test, st.predict(X_test_scaled), average='weighted'):.4f}")
```

---

## FIABILIDAD DEL SISTEMA (si el enunciado lo pide)

```python
# "Patrón con fiabilidad superior al X%" = % predicciones con confianza > umbral
probas     = best_model.predict_proba(X_test_scaled)  # requiere probability=True en SVC
max_probas = probas.max(axis=1)

print(f"Fiabilidad media: {max_probas.mean()*100:.1f}%")
print(f"Predicciones con confianza > 80%: {(max_probas > 0.8).mean()*100:.1f}%")
```

---

## TABLA COMPARATIVA DE MODELOS

```python
# Mostrar al final del bloque de modelos para justificar la elección del mejor
resultados = []
modelos = {
    "KNN":   (best_knn,   X_test_scaled),
    "Árbol": (best_arbol, X_test),
    "SVM":   (best_svm,   X_test_scaled),
    "RF":    (best_rf,    X_test),
    "MLP":   (best_mlp,   X_test_scaled),
}
for nombre, (m, X) in modelos.items():
    pred = m.predict(X)
    rmse = mean_squared_error(y_test, pred, squared=False)  # CAMBIAR para clasificación
    resultados.append({"Modelo": nombre, "RMSE": round(rmse, 4)})
display(pd.DataFrame(resultados).sort_values("RMSE"))
```

---

## PROHIBICIONES EN ESTE RA

- **Nunca** usar TensorFlow, Keras ni PyTorch para problemas tabulares. Solo `sklearn.neural_network`.
- **Nunca** usar `VotingRegressor` ni `StackingRegressor` como ensemble por defecto.
- **Nunca** usar `shap`, `LIME` ni `permutation_importance` para importancia de variables.
- **Nunca** evaluar en test más de una vez para comparar modelos (test leakage).
- **Nunca** usar accuracy como métrica principal si hay desbalanceo.

---

*Fin del documento RA3*
