import os
import numpy as np
from skimage import io, color
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from caracteristicas import extraer_caracteristicas
from clasificador import NaiveBayesGaussiano
from sklearn.naive_bayes import GaussianNB

if __name__ == "__main__":
    # Ruta a la carpeta principal en Google Drive
    ruta_carpeta_principal = 'complete_ms_data'
    imagenes = []
    etiquetas = []
    for subcarpeta in os.listdir(ruta_carpeta_principal):
        ruta_subcarpeta = os.path.join(ruta_carpeta_principal, subcarpeta)
        if os.path.isdir(ruta_subcarpeta):
            for archivo_imagen in os.listdir(ruta_subcarpeta):
                ruta_imagen = os.path.join(ruta_subcarpeta, archivo_imagen)
                imagen = io.imread(ruta_imagen) # Aquí se carga la imagen

                if len(imagen.shape) == 3 and imagen.shape[2] == 3:
                    imagen = color.rgb2gray(imagen)

                imagenes.append(imagen)
                etiquetas.append(subcarpeta)
                caracteristicas = extraer_caracteristicas(imagenes)

    # Dividir los datos en conjuntos de entrenamiento y pruebas de manera estratificada
    X_train, X_test, y_train, y_test = train_test_split(caracteristicas, etiquetas, test_size=0.25, stratify=etiquetas, random_state=42)

    '''Clasificador Manual'''
    # Crear el clasificador gaussiano ingenuo
    clasificador_manual = NaiveBayesGaussiano()

    # Entrenar el clasificador con el conjunto de entrenamiento
    clasificador_manual.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de pruebas
    predicciones_manual = clasificador_manual.predict(X_test)

    print("Precisión (manual):", accuracy_score(y_test, predicciones_manual))
    print("Matriz de Confusión (manual):")
    print(confusion_matrix(y_test, predicciones_manual))
    print("Reporte de Clasificación (manual):")
    print(classification_report(y_test, predicciones_manual))


    '''Clasificador con la biblioteca'''
    # Crear el clasificador gaussiano ingenuo
    clasificador = GaussianNB()

    # Entrenar el clasificador con el conjunto de entrenamiento
    clasificador.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de pruebas
    predicciones = clasificador.predict(X_test)


    precisión = accuracy_score(y_test, predicciones)
    print(f"Precisión: {precisión}")
    matriz_confusion = confusion_matrix(y_test, predicciones)
    print("Matriz de Confusión:")
    print(matriz_confusion)
    reporte_clasificacion = classification_report(y_test, predicciones)
    print("Reporte de Clasificación:")
    print(reporte_clasificacion)

                  