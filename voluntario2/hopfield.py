import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from IPython.display import Image, display

# --- CONFIGURACIÓN DE RUTA ---
PATH = r"C:\Users\Usuario\Desktop\2526_Computacional\voluntario2"
if not os.path.exists(PATH):
    os.makedirs(PATH)

class HopfieldNet:
    def __init__(self, N):
        self.N = N
        self.num_neurons = N * N
        self.w = None
        self.theta = None
        self.patterns = []
        self.a = []

    def store_patterns(self, p_list):
        # Construye la matriz de pesos sinápticos usando la regla de Hebb.
        self.patterns = [p.flatten() for p in p_list]
        self.a = [np.mean(p) for p in self.patterns] # a^mu: actividad media
        self.w = np.zeros((self.num_neurons, self.num_neurons)) #w: peso sináptico

        # Optimización: Calculamos la matriz de pesos de una sola vez usando operaciones vectorizadas
        # Nota: El producto exterior (outer) suma la correlación de cada par de neuronas para cada patrón.
        for mu in range(len(self.patterns)):
            p_c = self.patterns[mu] - self.a[mu]
            self.w += np.outer(p_c, p_c)

        # Normalización y eliminación de autoconexiones
        self.w /= self.num_neurons
        np.fill_diagonal(self.w, 0) # autoconexiones wij,ij = 0
        self.theta = 0.5 * np.sum(self.w, axis=1) # Umbral de disparo theta_ij
        # determina la sensibilidad de la neurona basándose en la suma de sus conexiones,

    # El solapamiento (m^mu) mide la similitud entre el estado actual de la red y un patrón guardado.
    def overlap(self, state, mu):
        # Calcula el solapamiento m^mu (fidelidad del recuerdo)
        s = state.flatten()
        p = self.patterns[mu]
        a = self.a[mu]
        den = self.num_neurons * a * (1 - a)
        if den == 0: return 0
        return np.sum((p - a) * (s - a)) / den

    # Algoritmo de Metrópolis. En vez de calcular el Hamiltoniano total, calculamos solo
    # el cambio al invertir una neurona (s_i -> 1 - s_i).
    def metropolis_step(self, state, T):
        # Dinámica de Metrópolis para minimizar el Hamiltoniano H
        s = state.flatten().astype(float)
        for _ in range(self.num_neurons):
            i = np.random.randint(0, self.num_neurons)
            delta_s = 1.0 - 2.0 * s[i] # Cambio de estado s: 0 <-> 1

            # Campo local h_i: influencia de la red sobre la neurona i
            h_i = np.dot(self.w[i], s)

            # Variación de energía Delta H
            delta_E = delta_s * (self.theta[i] - h_i)

            # Criterio de Metrópolis
            if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / T):
                s[i] = 1.0 - s[i]
        return s.reshape((self.N, self.N))
    


def obtener_diccionario_letras(N):
    # Definimos plantillas 5x7 para las letras. 'X' es píxel activo, '.' es inactivo.
    # Esto permite tener todo el abecedario de forma compacta.
    plantillas = {
        'A': ["..X..", ".X.X.", "X...X", "XXXXX", "X...X", "X...X", "X...X"],
        'B': ["XXXX.", "X...X", "XXXX.", "X...X", "X...X", "XXXX.", "....."],
        'C': [".XXXX", "X....", "X....", "X....", "X....", ".XXXX", "....."],
        'D': ["XXXX.", "X...X", "X...X", "X...X", "X...X", "XXXX.", "....."],
        'E': ["XXXXX", "X....", "XXXX.", "X....", "X....", "XXXXX", "....."],
        'F': ["XXXXX", "X....", "XXXX.", "X....", "X....", "X....", "....."],
        'G': [".XXXX", "X....", "X..XX", "X...X", "X...X", ".XXXX", "....."],
        'H': ["X...X", "X...X", "XXXXX", "X...X", "X...X", "X...X", "....."],
        'I': ["XXXXX", "..X..", "..X..", "..X..", "..X..", "XXXXX", "....."],
        'J': ["..XXX", "...X.", "...X.", "...X.", "X..X.", ".XX..", "....."],
        'K': ["X...X", "X..X.", "XXX..", "X..X.", "X...X", "X...X", "....."],
        'L': ["X....", "X....", "X....", "X....", "X....", "XXXXX", "....."],
        'M': ["X...X", "XX.XX", "X.X.X", "X...X", "X...X", "X...X", "....."],
        'N': ["X...X", "XX..X", "X.X.X", "X..XX", "X...X", "X...X", "....."],
        'O': [".XXX.", "X...X", "X...X", "X...X", "X...X", ".XXX.", "....."],
        'P': ["XXXX.", "X...X", "XXXX.", "X....", "X....", "X....", "....."],
        'Q': [".XXX.", "X...X", "X...X", "X.X.X", "X..X.", ".XX.X", "....."],
        'R': ["XXXX.", "X...X", "XXXX.", "X.X..", "X..X.", "X...X", "....."],
        'S': [".XXXX", "X....", ".XXX.", "....X", "XXXX.", ".....", "....."],
        'T': ["XXXXX", "..X..", "..X..", "..X..", "..X..", "..X..", "....."],
        'U': ["X...X", "X...X", "X...X", "X...X", "X...X", ".XXX.", "....."],
        'V': ["X...X", "X...X", "X...X", "X...X", ".X.X.", "..X..", "....."],
        'W': ["X...X", "X...X", "X...X", "X.X.X", "XX.XX", "X...X", "....."],
        'X': ["X...X", ".X.X.", "..X..", ".X.X.", "X...X", "X...X", "....."],
        'Y': ["X...X", ".X.X.", "..X..", "..X..", "..X..", "..X..", "....."],
        'Z': ["XXXXX", "....X", "...X.", "..X..", ".X...", "XXXXX", "....."]
    }

    diccionario = {}
    for letra, lineas in plantillas.items():
        # Convertimos la plantilla de texto a una matriz numérica
        matriz_pequeña = np.array([[1 if c == 'X' else 0 for c in fila] for fila in lineas])
        # Escalamos la matriz pequeña para que encaje en el tamaño N x N de la red
        diccionario[letra] = np.kron(matriz_pequeña, np.ones((N // 7 + 1, N // 5 + 1)))[:N, :N]
    return diccionario

