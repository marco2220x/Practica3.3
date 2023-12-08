from skimage.feature import hog
import numpy as np

def extraer_caracteristicas(imagenes):
    caracteristicas = []
    for imagen in imagenes:
        caracteristica_hog = hog(imagen, orientations=8, pixels_per_cell=(16, 16),
                                 cells_per_block=(1, 1), block_norm='L2-Hys')
        caracteristicas.append(caracteristica_hog)
    return np.array(caracteristicas)

