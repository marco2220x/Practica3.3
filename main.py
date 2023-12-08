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
