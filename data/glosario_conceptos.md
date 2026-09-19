\# Glosario de conceptos — Categorización basada en esquemas



\## Esquema (schema) / categoría como modelo generativo

Un esquema no es solo una etiqueta de clasificación, sino un modelo interno

estructurado de las regularidades compartidas por los miembros de una

categoría. A diferencia de un clasificador que solo traza una frontera de

decisión, un esquema permite: reconstruir ejemplos incompletos, predecir

características ausentes, estimar la tipicidad de un miembro, detectar

casos atípicos, y generar nuevos ejemplos plausibles de la categoría

(imaginación).



\## Autoencoder variacional (VAE)

Modelo generativo de variable latente propuesto por Kingma y Welling (2014).

Aprende una representación comprimida (espacio latente) de los datos de

entrada, de forma que se puedan generar nuevas muestras plausibles

muestreando ese espacio. Se entrena maximizando una cota inferior de la

verosimilitud (ELBO), que combina un término de reconstrucción con un

término de regularización (divergencia KL) que fuerza al espacio latente a

seguir una distribución conocida (típicamente gaussiana).



\## Memoria auto-asociativa

Modelo de memoria (p. ej. redes de Hopfield) que almacena patrones de forma

que, dado un patrón incompleto o con ruido como entrada, la red converge

hacia el patrón completo almacenado más parecido. Se relaciona con la

categorización porque un "prototipo" de categoría puede funcionar como un

patrón almacenado hacia el que convergen ejemplos atípicos o incompletos.



\## Tipicidad (typicality)

Medida de cuánto se ajusta un ejemplo concreto al esquema/prototipo

aprendido de su categoría. En el espacio latente de un modelo generativo,

los ejemplos típicos ocupan regiones centrales y densas; los atípicos caen

en regiones periféricas o de baja densidad.



\## Paisaje de energía (energy landscape)

Descripción, tomada de la física estadística, de un sistema en términos de

una función de energía sobre sus posibles configuraciones. Los mínimos de

energía corresponden a estados estables (patrones almacenados, prototipos);

las fluctuaciones alrededor de un mínimo describen variabilidad dentro de

una categoría. Es el puente conceptual entre memoria/cognición y física

teórica que usa este TFM.



\## Variables colectivas

En física estadística, variables agregadas que resumen el comportamiento de

un sistema de muchos componentes (p. ej. la magnetización media en un

sistema de espines). En este TFM, la idea se traslada a resumir el estado

de una red neuronal o de un espacio latente mediante magnitudes agregadas

que capturan la estructura aprendida de una categoría.



\## Consolidación de memoria (Spens \& Burgess, 2024)

Modelo según el cual las memorias episódicas (eventos específicos,

codificados rápidamente) se usan para entrenar, de forma más lenta, un

modelo generativo capaz de reconstruir y organizar la experiencia — un

mecanismo propuesto para explicar cómo el cerebro pasa de recuerdos

puntuales a conocimiento generalizado (esquemas).

