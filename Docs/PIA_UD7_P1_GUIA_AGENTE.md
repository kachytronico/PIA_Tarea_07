# Guía del agente — PROBLEMA 1 (Células malignas)

Esta guía define cómo debe comportarse el agente cuando me ayuda a construir y depurar el notebook `PIA_07_P1_Celulas_Malignas.ipynb`.

---

## Contexto general

- Estamos resolviendo el PROBLEMA 1 de la UD7 en Google Colab.  
- El notebook está organizado en 8 hitos que siguen el enunciado oficial.  
- El objetivo del agente es ayudar a escribir código mínimo y correcto, interpretar outputs y redactar conclusiones.

---

## Estilo de escritura en Markdown

- **Primera persona singular:** "He cargado el dataset", "Entreno un KNN".  
- **Estilo directo, sin adornos.**  
- **Longitud:** introducciones de 3–5 líneas; conclusiones de 4–8 líneas.  
- **Datos reales siempre:** las conclusiones deben incluir métricas o tamaños reales del output.  
- Si no hay output todavía, usar `[COMPLETAR cuando vea el output de esta celda]` en lugar de inventar números.

---

## Estructura de cada hito

Cada hito debe seguir este patrón:

1. **Introducción (Markdown H1):** qué voy a hacer y por qué.  
2. **Código:** mínimo necesario, comentado cuando haga falta, sin imports duplicados.  
3. **Conclusión (Markdown H2):** resumen de resultados numéricos y decisión sobre el siguiente paso.

Ejemplo de conclusión correcta para el KNN:

> ## Modelo KNN optimizado  
> He probado varios valores de vecinos y pesos con validación cruzada (k=5). El mejor modelo usa 11 vecinos y pesos uniformes, con una F1 media de 0.81 y un accuracy de 0.87 en validación. Con estos resultados, considero que el KNN es una buena base para el ensemble.

---

## Cómo actuar con outputs y errores

Cuando pegue el output de una celda:

1. Identifica el hito (AED, split, KNN, ensemble, etc.).  
2. Di si el resultado parece razonable o sospechoso.  
3. Redacta la conclusión en primera persona y dime dónde pegarla.

Cuando haya un error:

1. Resume la causa probable en una frase.  
2. Da el fix mínimo (líneas concretas a cambiar).  
3. Indica qué celda re-ejecutar y qué debería verse si el fix funciona.

---

## Lo que NO debes hacer

- No inventar métricas ni resultados.  
- No mezclar train y test para entrenar modelos o elegir hiperparámetros.  
- No cambiar nombres de variables clave sin avisar.  
- No reestructurar todo el notebook sin que se lo pida expresamente.
