# UD7_P2 – Guía de depuración (para el usuario)

Este documento está pensado para ti, no para el agente.

Te da frases y estrategias para **reconducir al agente** si se desvía cuando trabajéis en el Problema 2 de la UD7 (`¿Esto es real?`).[file:44]

---

## 1. Si el agente “se va por las ramas” (demasiada teoría)

Frase tipo que puedes usar:

> “Céntrate solo en el Problema 2 de la UD7, hito X. No quiero teoría general, solo las celdas de código mínimas para este hito y una explicación muy corta.”

Otro ejemplo:

> “Esto es demasiado teórico. Dame solo el código mínimo para el Hito X del P2 y qué cosas tengo que cambiar (rutas, nombres, etc.).”

---

## 2. Si el agente ignora los hitos

Si empieza a hacer cosas de otros hitos o mezcla varios:

> “Recuerda que el Problema 2 está dividido en 8 hitos. Ahora estamos en el Hito X. No avances al siguiente hito hasta que este quede claro.”

Si mezcla RA2 y RA4 cuando aún estás en la parte de datos:

> “Todavía estamos en los hitos RA2 (1–3). No quiero código de ResNet ni entrenamiento deep hasta que hayamos terminado Hitos 1–3.”[file:44]

---

## 3. Si el código no coincide con la estructura de carpetas real

Cuando te dé código que asume rutas que tú no tienes:

> “Muéstrame primero un plan para inspeccionar la estructura de `P2.zip` en Colab (por ejemplo con `os.listdir` o `!tree`). Después, adapta el código a esa estructura concreta.”

Si el agente inventa nombres de carpetas:

> “No inventes nombres de carpetas. Usa exactamente los nombres que se ven en `os.listdir` y pregúntame si dudas. Revisa la estructura antes de escribir el `DataBlock`.”

---

## 4. Si se olvida de FastAI y se va a otro stack

Si empieza a usar Keras/TensorFlow sin venir a cuento:

> “Quiero que uses FastAI (`fastai.vision.all`) como en UD5, con `DataBlock` y `vision_learner`. No cambies a Keras/TensorFlow salvo que te lo pida explícitamente.”

---

## 5. Si no respeta el enunciado de los hitos

Ejemplo: intenta saltarse progressive resizing o la matriz de confusión.[file:44]

> “Asegúrate de seguir al pie de la letra los 8 hitos del Problema 2 del documento PIA_07_Tarea. Ahora mismo estás ignorando [nombre del hito]. Vuelve a centrarte en ese hito.”

Puedes copiarle el hito concreto del enunciado y añadir:

> “Este es el hito que quiero resolver ahora. Proponme solo el código y explicación para este punto.”

---

## 6. Si el entrenamiento tarda demasiado o usa modelos demasiado pesados

> “Proponme una versión más ligera del modelo (menos épocas, batch size razonable, arquitectura tipo ResNet18/34) que se pueda entrenar en Colab en pocos minutos.”

> “No busques el mejor modelo posible, solo uno razonable que cumpla el hito sin tardar una eternidad.”

---

## 7. Si no explica qué hay que adaptar

Cuando te da código “cerrado” sin parámetros claros:

> “Marca explícitamente qué partes del código tengo que adaptar al enunciado: rutas, columnas, nombres de carpetas, tamaños de imagen. Usa comentarios `# CAMBIAR` en esas líneas.”

---

## 8. Si quieres que conecte con las chuletas (RA2/RA4)

> “Indícame qué parte de la chuleta del RA2 o RA4 estás reutilizando en este código, para tener claro a qué bloque corresponde.”

---

## 9. Si quieres un resumen final por hito

Al acabar un hito, puedes pedirle:

> “Hazme un resumen rápido del Hito X: objetivo, RA implicado, fragmentos de código clave y errores que debería evitar en un examen.”

Esto te ayudará a transformar el trabajo del cuaderno en apuntes de examen.

---

## 10. Si el agente se rompe o hace cosas incoherentes

Siempre puedes “reiniciar” la conversación de forma suave:

> “Ignora las últimas respuestas y vuelve a centrarte en el Problema 2 de la UD7. Recuérdame brevemente los 8 hitos y pregúntame en cuál estamos.”

Con estas frases tipo podrás reconducir al agente cuando se desvíe y mantener el foco en resolver la tarea de forma útil para el examen.

Fin del archivo `UD7_P2_GuiaDepuracion`.
