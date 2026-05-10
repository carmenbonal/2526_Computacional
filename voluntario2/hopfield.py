import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

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
        matriz_pequena = np.array([[1 if c == 'X' else 0 for c in fila] for fila in lineas])
        diccionario[letra] = np.kron(matriz_pequena, np.ones((N // 7 + 1, N // 5 + 1)))[:N, :N]
    return diccionario

def run_all_tasks():
    T_low = 1e-4

    # --- TAREA 1: ANIMACIÓN (N=50) ---
    print("Generando GIF de la Tarea 1...")
    N1 = 50
    p1 = np.zeros((N1, N1))
    p1[N1//2-2:N1//2+2, :] = 1
    p1[:, N1//2-2:N1//2+2] = 1 # Patrón en cruz
    net1 = HopfieldNet(N1)
    net1.store_patterns([p1])
    
    
    state = np.random.randint(0, 2, (N1, N1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im = ax1.imshow(state, cmap='binary')
    line, = ax2.plot([], [], 'r-')
    ax2.set_xlim(0, 20); ax2.set_ylim(-1.1, 1.1)
    ax2.set_title("Solapamiento m(t)")
    ax2.grid(True)
    m_hist = []

    def update(frame):
        nonlocal state
        m_hist.append(net1.overlap(state, 0))
        im.set_data(state)
        line.set_data(range(len(m_hist)), m_hist)
        state = net1.metropolis_step(state, T_low)
        return im, line

    ani = FuncAnimation(fig, update, frames=21, blit=True)
    ani.save(os.path.join(PATH, "evolucion_hopfield.gif"), writer='pillow')
    plt.close()

    # --- TAREA 2: INFLUENCIA DE LA TEMPERATURA ---
    print("Ejecutando Tarea 2: Solapamiento vs Temperatura...")
    temps = np.linspace(0.001, 1.5, 25)
    m_finales = []
    
    for t in temps:
        s_t = p1.copy()
        
        # 1. Termalización: Dejamos que la red evolucione durante 100 pasos para alcanzar un estado estacionario a la temperatura dada.
        for _ in range(100):
            s_t = net1.metropolis_step(s_t, t)
            
        # 2. Promediado de la medida (reducción de fluctuaciones térmicas)
        m_avg = 0
        for _ in range(50):
            s_t = net1.metropolis_step(s_t, t)
            m_avg += abs(net1.overlap(s_t, 0))
            
        m_finales.append(m_avg / 50.0)

    plt.figure()
    plt.plot(temps, m_finales, 'o-')
    plt.title("Tarea 2: |m| medio final vs Temperatura")
    plt.xlabel("T"); plt.ylabel("|m|")
    plt.grid(); plt.savefig(os.path.join(PATH, "tarea2_m_vs_T.png"))
    print(f"Gráfica guardada en {PATH}")

    # --- TAREA 3: INTERACCIÓN CON EL USUARIO (N=30) ---
    print("\n--- INICIANDO TAREA 3 ---")
    N3 = 30
    dict_letras = obtener_diccionario_letras(N3)

    while True:
        entrada = input("Introduce 3 letras de la A a la Z (ej: ABC): ").upper()
        if len(entrada) == 3 and all(c in dict_letras for c in entrada):
            break
        print("Entrada no válida. Por favor, introduce exactamente 3 letras.")

    letras_elegidas = [dict_letras[c] for c in entrada]
    net3 = HopfieldNet(N3)
    net3.store_patterns(letras_elegidas)

    print(f"Simulando recuperación de la letra '{entrada[0]}'...")
    s_test = letras_elegidas[0].copy()
    ruido = np.random.rand(N3, N3) < 0.3
    s_test[ruido] = 1 - s_test[ruido]

    m_hist_3 = {i: [] for i in range(3)}
    
    for _ in range(30):
        for i in range(3):
            m_hist_3[i].append(net3.overlap(s_test, i))
        s_test = net3.metropolis_step(s_test, T_low)

    plt.figure()
    for i in range(3):
        plt.plot(m_hist_3[i], label=f"Letra {entrada[i]}")
    plt.title(f"Tarea 3: Evolución con letras {entrada}")
    plt.xlabel("Pasos Monte Carlo"); plt.ylabel("Solapamiento m")
    plt.legend(); plt.grid()
    plt.savefig(os.path.join(PATH, "tarea3_letras_usuario.png"))
    print(f"Gráfica guardada en {PATH}")

    # --- TAREA 4: CAPACIDAD (N=20) ---
    print("\nCalculando Tarea 4: Capacidad Crítica...")
    N4 = 20
    
    #  Para N=20 (400 neuronas), la capacidad teórica máxima es
    # P_c = 0.138 * 400 = 55. Probamos hasta 120 para ver el comportamiento empírico.
    P_vals = np.arange(1, 120, 5) 
    recuperados = []
    Pc = 0
    
    for P in P_vals:
        p_list = [(np.random.rand(N4, N4) > 0.5).astype(int) for _ in range(P)]
        net4 = HopfieldNet(N4)
        net4.store_patterns(p_list)
        
        count = 0
        for mu in range(P):
            s_test = p_list[mu].copy()
            
            # Para saber si el estado es un atractor y no un
            # máximo inestable, hay que darle un empujón diminuto (ruido) y 
            # dejarlo evolucionar. 
            ruido = np.random.rand(N4, N4) < 0.05
            s_test[ruido] = 1 - s_test[ruido]
            
            # Evolucionamos para verificar la estabilidad de la cuenca de atracción.
            for _ in range(25):
                s_test = net4.metropolis_step(s_test, T_low)
                
            if net4.overlap(s_test, mu) > 0.75:
                count += 1
                
        recuperados.append(count)
        if count >= P * 0.9: 
            Pc = P

    alpha_c = Pc / (N4*N4)
    print(f"Resultado T4: alpha_c = {alpha_c:.4f}")

    plt.figure()
    plt.plot(P_vals, recuperados, 's-', label="Recordados")
    plt.plot(P_vals, P_vals, '--k', alpha=0.3, label="Límite Ideal")
    plt.axvline(x=Pc, color='r', label=f'Pc={Pc}')
    plt.xlabel("Patrones almacenados (P)")
    plt.ylabel("Patrones recordados")
    plt.legend(); plt.grid()
    plt.savefig(os.path.join(PATH, "tarea4_capacidad.png"))

if __name__ == "__main__":
    run_all_tasks()