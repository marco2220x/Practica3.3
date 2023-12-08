# Clasificador Gaussiano Ingenuo (Naive Bayes Classifier) 📸

En el transcurso de este proyecto, se llevará a cabo la implementación de un clasificador Gaussiano ingenuo. El propósito central de esta implementación es alcanzar el reconocimiento efectivo de objetos. El clasificador Gaussiano ingenuo es un modelo probabilístico que asume independencia entre las características de los datos, haciendo que sea "ingenuo" en su enfoque, pero a menudo es utilizado con éxito en diversas tareas de clasificación. En este contexto, se busca emplear esta técnica para lograr la identificación y clasificación precisa de objetos en un conjunto de datos, aprovechando las propiedades estadísticas y de probabilidad que ofrece el modelo Gaussiano. 

## Etapas del clasificador Gaussiano ingenuo
A continuación se describen las fases de procesamiento del modelo probabilístico. 

1. **Recopilación de datos:**  El proceso comienza con la recopilación de un conjunto de imágenes que representan diferentes clases o categorías. Cada imagen en el conjunto tiene asociada una etiqueta que indica a qué clase pertenece.

2. **Preprocesamiento de datos:** El preprocesamiento de los datos incluye la carga y el ajuste de las imágenes, así como la extracción de características utilizando el método de HOG.

3. **Asunciones del Modelo:** El clasificador Gaussiano ingenuo hace dos suposiciones clave:

   a. **Independencia condicional:**
      - Supone que las características utilizadas para describir las instancias son independientes entre sí dado el valor de la variable de clase. Esta es la razón por la cual se le llama "ingenuo" o "naive".

   b. **Distribución Gaussiana:**
      - Supone que las características siguen una distribución normal (Gaussiana) dentro de cada clase. Esto significa que los valores de las características se distribuyen de acuerdo con una curva de campana.

4. **Estimación de Parámetros:** El siguiente paso es estimar los parámetros del modelo. Para cada clase, se calculan la media y la desviación estándar de cada característica en el conjunto de entrenamiento.

5. **Cálculo de Probabilidades:** Dado el nuevo conjunto de características para una instancia no etiquetada, el clasificador calcula la probabilidad de que pertenezca a cada clase. Esto se hace utilizando la función de densidad de probabilidad de la distribución normal para cada característica.

6. **Entrenamiento y Evaluación:** El modelo se entrena utilizando un conjunto de datos etiquetado y se evalúa en un conjunto de datos de prueba para medir su rendimiento en términos de precisión y otras métricas de evaluación.

7. **Aplicación del Modelo:** Una vez entrenado y evaluado, el modelo se puede utilizar para predecir la clase de nuevas imágenes desconocidas.


## Cómo Usar el Programa
Aquí te proporcionamos instrucciones sobre cómo utilizar nuestro programa:
1. Clona este repositorio en tu máquina local.
2. Asegúrate de tener Python y las bibliotecas necesarias instaladas.
3. Ejecuta el programa y proporciona una imagen en escala de grises como entrada.
4. El programa aplicará las técnicas de segmentación y mostrará los resultados utilizando Matplotlib.

## Autores
Este proyecto fue realizado por un equipo de estudiantes:
| [<img src="https://avatars.githubusercontent.com/u/113084234?v=4" width=115><br><sub>Aranza Michelle Gutierrez Jimenez</sub>](https://github.com/AranzaMich) |  [<img src="https://avatars.githubusercontent.com/u/113297618?v=4" width=115><br><sub>Evelyn Solano Portillo</sub>](https://github.com/Eveeelyyyn) |  [<img src="https://avatars.githubusercontent.com/u/112792541?v=4" width=115><br><sub>Marco Castelan Rosete</sub>](https://github.com/marco2220x) | [<img src="https://avatars.githubusercontent.com/u/113079687?v=4" width=115><br><sub>Daniel Vega Rodríguez</sub>](https://github.com/DanVer2002) |
| :---: | :---: | :---: | :---: |
