# DOCUMENTO RA1 — AED Y EXPLORACIÓN DE DATOS
**Cuándo cargarlo:** hitos de carga de datos, exploración inicial, análisis exploratorio.
**Instrucción para el agente:** Lee este documento completo antes de proponer cualquier celda de este hito. Usa exactamente los patrones aquí descritos.

---

## IMPORTS OBLIGATORIOS

```python
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

random_seed = 33  # SIEMPRE 33, nunca 42
```

---

## PIPELINE AED — ORDEN EXACTO

El AED siempre cubre estos cinco bloques en este orden. No omitas ninguno aunque parezca obvio, el enunciado suele pedir explícitamente cada uno.

### Bloque 1 — Carga e inspección inicial

```python
# Cargar el dataset
df = pd.read_csv("ruta/al/archivo.csv")  # CAMBIAR ruta real

# Inspección básica obligatoria — estas tres líneas siempre
print(df.shape)
display(df.head())
df.info()           # tipos de datos y recuento de no-nulos
display(df.describe())  # estadísticos numéricos
display(df.isnull().sum())  # nulos por columna
```

### Bloque 2 — Distribución de la variable objetivo

```python
# CAMBIAR "target" por el nombre real de la columna objetivo
TARGET = "charges"  # CAMBIAR

print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True).round(3))

# Visualizar distribución
df[TARGET].value_counts().plot(kind="bar", figsize=(6, 4))
plt.title(f"Distribución de {TARGET}")
plt.tight_layout(); plt.show()

# Detectar desbalanceo (para clasificación)
# Si la clase más frecuente > 70% del total → desbalanceo → usar F1, no accuracy
```

### Bloque 3 — Detección visual de outliers

```python
# Boxplots de todas las columnas numéricas
num_cols = df.select_dtypes(include="number").columns.tolist()
df[num_cols].plot(kind="box", subplots=True, layout=(2, -1), figsize=(15, 6))
plt.suptitle("Detección de outliers")
plt.tight_layout(); plt.show()
```

### Bloque 4 — Correlaciones

```python
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Mapa de correlaciones")
plt.tight_layout(); plt.show()

# Interpretar:
# Correlación alta con el TARGET → variable relevante para el modelo
# Correlación alta entre dos variables (>0.8) → posible redundancia → marcar para Hito 2
# Correlación perfecta (1.0) con TARGET → DATA LEAKAGE → eliminar antes de modelar
```

### Bloque 5 — Detectar valores sin sentido

```python
# Valores sin sentido: imposibles según el dominio del problema
# Ejemplos: edad negativa, precio = 0, nombre con solo números
# Inspeccionarlos manualmente y anotar qué columnas afectan
# El tratamiento se hace en RA2, aquí solo se identifican
print("Valores únicos por columna categórica:")
for col in df.select_dtypes(include="object").columns:
    print(f"  {col}: {df[col].unique()}")
```

---

## DECISIONES QUE TOMAR TRAS EL AED

Documenta en la celda de conclusión:

| Hallazgo | Decisión | Hito donde se trata |
|----------|----------|---------------------|
| Columna X tiene correlación 1.0 con target | Eliminar → leakage | Hito 2 |
| Columna Y es un ID | Eliminar → irrelevante | Hito 2 |
| Columna Z tiene outliers en boxplot | Aplicar outlier_eliminator | RA2 (tras split) |
| Columna W tiene >50% nulos | Eliminar columna | Hito 2 |
| Columnas A y B tienen correlación 0.95 | Eliminar una | Hito 2 |
| Variable objetivo desbalanceada | Usar F1, añadir stratify en split | RA3 |

---

## LO QUE EL CURSO DEFINE COMO CADA TIPO DE ANOMALÍA

- **Valor sin sentido:** imposible en el dominio real. Ejemplo: llamarse "5353456", tener -3 hijos.
- **Valor atípico (outlier):** posible pero extremo. Ejemplo: BMI de 80, sueldo de 10M. Se trata con `outlier_eliminator` en RA2, DESPUÉS del split.
- **Dato correlado:** columna que es combinación de otras o que revela directamente el target. Se elimina.
- **Dato irrelevante:** ID, timestamp, texto libre sin valor predictivo. Se elimina.

---

## PROHIBICIONES EN ESTE HITO

- No hagas el split en el hito del AED si el enunciado no lo pide explícitamente en este hito.
- No trates outliers en el AED. Solo identifícalos. El tratamiento es siempre DESPUÉS del split.
- No elimines columnas sin justificarlo en la conclusión.

---

*Fin del documento RA1*
