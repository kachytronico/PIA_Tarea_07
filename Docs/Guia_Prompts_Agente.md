# GUÍA DE PROMPTS PARA EL AGENTE PIA

Documento de referencia para el usuario. Copia y pega los prompts según el momento de la sesión.

---

## 1. PROMPT DE INICIO DE SESIÓN

Usar después de cargar los documentos de contexto (DOC1, DOC2, DOC3) y el enunciado del examen.

```
He cargado los documentos de contexto y el enunciado del examen.
Lee todos los documentos antes de responder.

Cuando termines, dime en el chat:
1. Tipo de problema de cada parte (tabular / imágenes / texto)
2. Tarea de cada parte (clasificación / regresión / clustering)
3. Métrica que pide el enunciado literalmente
4. Patrón de validación de cada parte y por qué (cita el texto
   exacto del enunciado que lo justifica)
5. Riesgos que detectas antes de empezar

No escribas código. Espera mi confirmación del análisis.
```

---

## 2. PROMPT DE INICIO DE HITO

Usar antes de cada hito nuevo. Carga primero el documento RA correspondiente.

```
Cargado el documento [RA1 / RA2 / RA3 / RA4]. Léelo completo.

Vamos al Hito [N]: [título breve del hito].

Antes de escribir código, cítame en el chat:
- Qué sección del documento RA aplica a este hito
- Qué regla o función del curso usarás
- Qué está prohibido hacer en este hito según el documento

Cuando lo hayas citado, propón el código en el chat
(sin añadirlo al cuaderno todavía) y espera mi OK.
```

---

## 3. PROMPT DE CONTROL DE FORMATO DEL CUADERNO

Usar si el agente no respeta la estructura o el idioma.

```
Recuerda la estructura obligatoria del cuaderno:
1. Celda Markdown de encabezado: título del hito en español,
   qué se hace (1-2 frases), qué RA trabaja.
2. Celdas de código necesarias. Comentarios en español,
   escritos en primera persona ("Cargo los datos", "Elimino
   las columnas", "Escalo las variables").
3. Celda Markdown de conclusión: escrita en primera persona
   y en español. IMPORTANTE: la conclusión se escribe DESPUÉS
   de que yo ejecute el código y te muestre el output.
   No la escribas antes de ver el resultado.

Prohibido: celdas de Reasoning, Subtask, Final Task,
Summary, Q&A. Eso va en el chat, no en el cuaderno.
```

---

## 4. PROMPT DE CONCLUSIÓN (después de ejecutar el código)

Usar cuando has ejecutado el código y tienes el output. Pégalo junto con el output.

```
Output del Hito [N]:
[pega aquí el output o describe lo que ves]

Con este resultado:
1. Escribe la celda Markdown de conclusión del hito en
   primera persona y en español. Interpreta el output
   y explica qué decisión tomas para el siguiente hito.
2. Dime qué estrategia seguimos en el Hito [N+1] teniendo
   en cuenta lo que acabamos de ver.
```

---

## 5. PROMPT DE CORRECCIÓN DE REGLA

Usar cuando el agente hace algo que contradice los documentos de contexto.

```
Esto contradice el documento [RA2 / RA3 / RA4], sección
[nombre de la sección]. La regla del curso dice:
[cita la regla exacta].

Corrige solo esa parte sin reescribir el resto del hito.
Propón la corrección en el chat antes de tocar el cuaderno.
```

---

## 6. PROMPT DE CORRECCIÓN DE SPLIT

Usar cuando el agente no respeta los porcentajes del enunciado.

```
El enunciado dice exactamente "[cita literal del enunciado]".
Eso significa test_size=[X], un solo train_test_split.
No añadas conjunto de validación extra si el enunciado
no lo pide. GridSearchCV hace la validación interna.
Corrige el split y propón el código corregido en el chat.
```

---

## 7. PROMPT DE CORRECCIÓN DE OUTLIERS

Usar cuando el agente usa clip, cap o IQR estándar para outliers.

```
Los outliers en este curso se tratan con la función
outlier_eliminator del documento RA2. Esa función:
- Elimina FILAS (no recorta valores con clip ni cap)
- Usa la MEDIA como centro del IQR, no Q1 ni Q3
- Solo elimina si los outliers son < 5% del total
- Se aplica SOLO sobre el conjunto de entrenamiento

Nunca sobre el test (el test no se toca, representa
datos reales de producción).

Reescribe la celda de outliers usando esa función.
```

---

## 8. PROMPT DE CORRECCIÓN DE ENSEMBLE

Usar cuando el agente usa VotingClassifier o VotingRegressor.

```
El curso usa promedio manual de predicciones como ensemble,
no VotingClassifier ni VotingRegressor. La razón es que
el promedio manual es más transparente y permite justificar
la decisión. Reescribe el ensemble así:

pred_ensemble = (pred_modelo1 + pred_modelo2 + pred_modelo3) / 3
```

---

## 9. PROMPT DE REORIENTACIÓN (si el agente se desvía)

Usar cuando el agente mezcla hitos, da demasiada teoría o avanza sin confirmación.

```
Para. Estamos en el Hito [N] y solo en ese.
No avances al siguiente hito hasta que yo confirme el output.
Dame solo el código mínimo para este hito, con la cita
del documento RA que lo respalda.
```

---

## 10. PROMPT DE REINICIO SUAVE

Usar si la conversación se ha descontrolado y quieres retomar el filo.

```
Ignora las últimas respuestas. Volvemos al Hito [N].

Tienes cargados los documentos de contexto.
Recuérdame brevemente qué pide este hito según el enunciado,
qué regla del documento RA aplica, y propón el código
en el chat esperando mi OK antes de añadirlo al cuaderno.
```

---

## RECORDATORIO RÁPIDO — CÓMO LEER EL OUTPUT ANTES DE LA CONCLUSIÓN

Cuando ejecutes una celda y veas el output, comprueba:

| Lo que ves | Pregunta al agente |
|-----------|-------------------|
| F1 = 1.0 exacto | "¿Hay data leakage? Revisa las columnas" |
| Accuracy = % de la clase mayoritaria | "El modelo no aprende. ¿Por qué?" |
| std > 0.10 entre folds | "¿Está bien estratificado?" |
| RMSE/MAE muy alto | "¿El modelo necesita más optimización o es el dato?" |
| Clases desequilibradas | "¿Estás usando F1 y stratify?" |

---

*Fin de la guía de prompts*
