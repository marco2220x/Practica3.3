import numpy as np
from sklearn.preprocessing import LabelEncoder

class NaiveBayesGaussiano:
    def __init__(self):
        self.medias = None
        self.desviaciones = None
        self.probabilidades_clases = None

    def fit(self, X, y):
        # Utilizar LabelEncoder para convertir etiquetas de cadena a números
        self.label_encoder = LabelEncoder()
        y_numerico = self.label_encoder.fit_transform(y)

        clases = np.unique(y_numerico)
        num_clases = len(clases)
        num_caracteristicas = X.shape[1]

        # Calcular medias y desviaciones estándar para cada clase y característica
        self.medias = np.zeros((num_clases, num_caracteristicas))
        self.desviaciones = np.zeros((num_clases, num_caracteristicas))
        self.probabilidades_clases = np.zeros(num_clases)

        for i, clase in enumerate(clases):
            datos_clase = X[y_numerico == clase]
            self.medias[i] = np.mean(datos_clase, axis=0)
            self.desviaciones[i] = np.std(datos_clase, axis=0)
            self.probabilidades_clases[i] = len(datos_clase) / len(y_numerico)
            
    def calcular_probabilidad(self, x, media, desviacion):
        # Asegurar que la desviación estándar no sea cero
        desviacion = max(desviacion, 1e-10)
        exponente = np.exp(-((x - media) ** 2) / (2 * desviacion ** 2))
        return (1 / (np.sqrt(2 * np.pi) * desviacion)) * exponente

    def predict(self, X):
        predicciones = []

        for dato in X:
            probabilidades = []

            for i in range(len(self.medias)):
                prob_clase = np.log(self.probabilidades_clases[i])

                for j in range(len(dato)):
                    # Corregir el manejo de casos donde la desviación estándar es cero
                    prob_clase += np.log(self.calcular_probabilidad(dato[j], self.medias[i][j], self.desviaciones[i][j]))

                probabilidades.append(prob_clase)

            clase_predicha = np.argmax(probabilidades)
            predicciones.append(clase_predicha)

        # Utilizar inverse_transform para obtener las etiquetas originales
        predicciones = self.label_encoder.inverse_transform(predicciones)
        return np.array(predicciones)