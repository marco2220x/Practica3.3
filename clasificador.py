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
