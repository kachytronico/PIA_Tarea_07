# Guía de depuración — PROBLEMA 1 (Células malignas)

Este documento recoge errores frecuentes al ejecutar `PIA_07_P1_Celulas_Malignas.ipynb` en Google Colab y el fix mínimo recomendado.

---

## Checklist antes de ejecutar

- Dataset de P1 cargado en la ruta correcta.  
- Celdas ejecutadas en orden, de arriba a abajo.  
- No continuar tras un error sin arreglarlo.  
- Variables clave (`df`, `X`, `y`, `X_train`, `y_train`) creadas antes de entrenar modelos.

---

## Errores típicos

### Carga de datos

**Error:** `FileNotFoundError`  
**Causa:** ruta incorrecta o Drive no montado.  
**Fix:** comprobar con `!ls` y ajustar la ruta en `pd.read_csv()`.

---

### Split y K-Fold

**Error:** `ValueError: The least populated class in y has only 1 member`  
**Causa:** stratify o K-Fold con clases demasiado pequeñas.  
**Fix:** comprobar distribución de `anomaly_label` y, si es necesario, reducir `n_splits` (ej. de 5 a 3).

**Error:** `ValueError: Found input variables with inconsistent numbers of samples`  
**Causa:** X e y tienen longitudes distintas.  
**Fix:** imprimir `X.shape` y `len(y)` y revisar cualquier filtrado para aplicarlo a ambos.

---

### Nulos y codificación

**Error:** `ValueError: Input contains NaN, infinity or a value too large`  
**Causa:** quedan nulos o valores infinitos sin tratar.  
**Fix:** añadir imputación (SimpleImputer o `fillna`) antes de entrenar.

---

### PCA y escalado

**Error:** `ValueError: n_components=XX must be between 0 and min(n_samples, n_features)`  
**Causa:** número de componentes incompatible.  
**Fix:** usar `PCA(n_components=0.95)` o ajustar el número.

---

### Modelos (KNN, otros)

**Error:** `NotFittedError` al llamar a `predict`  
**Causa:** el modelo no se ha entrenado.  
**Fix:** ejecutar primero la celda de `fit` y confirmar que no da errores.

**Error:** `ValueError: Unknown label type: 'continuous'`  
**Causa:** la variable objetivo `y` se ha convertido en continua por error.  
**Fix:** revisar que `y` contiene solo clases (0/1) y no se modifica en el preprocesado.

---

### Ensemble

**Error:** longitudes distintas de predicciones  
**Causa:** cada modelo predice sobre conjuntos diferentes.  
**Fix:** asegurarse de que todos predicen sobre el mismo conjunto (mismas filas y orden).

---

## Si el runtime se reinicia

- Ejecutar todo el notebook desde la primera celda.  
- Si pasa a menudo, reducir el tamaño del dataset para pruebas y simplificar modelos.
