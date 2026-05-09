import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# --- CONFIGURACIÓN DE RUTA ---
# Usamos r"" para evitar problemas con las backslashes de Windows
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
        self.a = [np.mean(p) for p in self.patterns] 
        self.w = np.zeros((self.num_neurons, self.num_neurons))

        # Optimización: Calculamos la matriz de pesos de una sola vez usando operaciones vectorizadas
        # Nota: El producto exterior (outer) suma la correlación de cada par de neuronas para cada patrón.
        for mu in range(len(self.patterns)):
            p_c = self.patterns[mu] - self.a[mu]
            self.w += np.outer(p_c, p_c)
            
        # Normalización y eliminación de autoconexiones
        self.w /= self.num_neurons
        np.fill_diagonal(self.w, 0) # Autoconexiones prohibidas 
        self.theta = 0.5 * np.sum(self.w, axis=1) # Umbral de disparo 

    def overlap(self, state, mu):
        # Calcula cuánto se parece el estado actual (s) a un patrón almacenado (p)
        s = state.flatten()
        p = self.patterns[mu]
        a = self.a[mu]
        den = self.num_neurons * a * (1 - a)
        # Si den es 0 (patrón uniforme), devolvemos 0 para evitar error
        if den == 0: return 0
        return np.sum((p - a) * (s - a)) / den 

    def metropolis_step(self, state, T):
        # Realiza un paso de actualización de la red usando el criterio de Metrópolis
        #  para aceptar o rechazar cambios. Si la energía disminuye, se acepta el cambio.
        #  Si aumenta, se acepta con probabilidad exp(-delta_E/T).
        s = state.flatten().astype(float)
        for _ in range(self.num_neurons):
            i = np.random.randint(0, self.num_neurons)
            delta_s = 1.0 - 2.0 * s[i] # Cambio s -> 1-s 
            
            # El campo local h_i es la influencia de todas las demás neuronas sobre la neurona i
            h_i = np.dot(self.w[i], s)
            
            # delta_E representa la variación de energía del sistema si cambiamos la neurona i
            delta_E = delta_s * (self.theta[i] - h_i) # Optimización 
            
            # Criterio de Metrópolis 
            if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / T):
                s[i] = 1.0 - s[i]
        return s.reshape((self.N, self.N))

# --- GENERACIÓN DE RESULTADOS ---

def run_all_tasks():
    # --- TAREA 1: ANIMACIÓN (N=30, T=1e-4)  ---
    N1 = 50
    T_low = 1e-4
    # Creamos un patrón visual (una cruz)
    p_task1 = np.zeros((N1, N1))
    p_task1[N1//2-2:N1//2+2, :] = 1
    p_task1[:, N1//2-2:N1//2+2] = 1
    
    net1 = HopfieldNet(N1)
    net1.store_patterns([p_task1])
    
    # Condición inicial aleatoria 
    state = np.random.randint(0, 2, (N1, N1))
    
    # Listas para almacenar la evolución del solapamiento y los fotogramas de la animación
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im = ax1.imshow(state, cmap='binary')
    ax1.set_title("Evolución de la Red (N=30)")
    
    # Gráfica de solapamiento m(t)
    line, = ax2.plot([], [], 'r-')
    ax2.set_xlim(0, 20); ax2.set_ylim(-1.1, 1.1)
    ax2.set_title("Solapamiento m(t)")
    ax2.grid(True)
    
    m_history = []

    def update(frame):
        nonlocal state
        m = net1.overlap(state, 0)
        m_history.append(m)
        im.set_data(state)
        line.set_data(range(len(m_history)), m_history)
        state = net1.metropolis_step(state, T_low)
        return im, line

    print("Generando GIF de la Tarea 1...")
    ani = FuncAnimation(fig, update, frames=21, blit=True, repeat=False)
    ani.save(os.path.join(PATH, "evolucion_hopfield.gif"), writer='pillow')
    plt.close() # Cerramos para seguir con las otras tareas

    # --- TAREA 2: TEMPERATURA ---
    # Analizamos cómo afecta el "ruido termico" (temperatura) a la estabilida de la memoria.
    print("Calculando Tarea 2: Solapamiento vs T...")
    temps = np.linspace(0.001, 1, 30)
    m_vs_t = []
    for t in temps:
        s_test = p_task1.copy()
        # Dejamos que el sistema evolucione suficiente para alcanzar el equilibrio térmico
        for _ in range(15): s_test = net1.metropolis_step(s_test, t)
        m_vs_t.append(abs(net1.overlap(s_test, 0)))
    
    plt.figure()
    plt.plot(temps, m_vs_t, 'o-')
    plt.title("Tarea 2: |m| vs Temperatura")
    plt.xlabel("T"); plt.ylabel("|m|")
    plt.grid()
    plt.savefig(os.path.join(PATH, "tarea2_m_vs_T.png"))

    # --- TAREA 4: CAPACIDAD (N=20, T=1e-4)  ---
    print("Calculando Tarea 4: Capacidad Crítica...")
    N4 = 50
    # Ampliamos el rango de P para asegurar que vemos el colapso (P_max > 0.15 * N^2)
    P_vals = np.arange(1, 501, 20) 
    recuperados = []
    Pc = 0
    
    for P in P_vals:
        p_list = [(np.random.rand(N4, N4) > 0.5).astype(int) for _ in range(P)] 
        net4 = HopfieldNet(N4)
        net4.store_patterns(p_list)
        
        count = 0
        for mu in range(P):
            # Partimos del propio patrón para ver si es un mínimo estable
            s_test = p_list[mu].copy()
            for _ in range(10): s_test = net4.metropolis_step(s_test, T_low)
            # Criterio del guion: se recuerda si m > 0.75
            if net4.overlap(s_test, mu) > 0.75: 
                count += 1
        
        recuperados.append(count)
        
        # Guardamos el último P donde la red aún recuerda casi todo (ej. > 90%)
        if count >= P * 0.9:
            Pc = P
    
    # Cálculo de alpha_c (capacidad máxima relativa)
    alpha_c = Pc / (N4 * N4)
    
    print(f"--- RESULTADOS TAREA 4 ---")
    print(f"P crítico (Pc) detectado: {Pc}")
    print(f"Capacidad crítica calculada: alpha_c = {alpha_c:.4f}")
    print(f"--------------------------")

    plt.figure()
    plt.plot(P_vals, recuperados, 's-', label="Patrones Recordados")
    plt.plot(P_vals, P_vals, '--k', alpha=0.3, label="Límite Ideal (100%)")
    plt.axvline(x=Pc, color='r', linestyle=':', label=f'Pc ≈ {Pc} (alpha_c={alpha_c:.3f})')
    plt.title(f"Tarea 4: Capacidad de la Red (N={N4})")
    plt.xlabel("Patrones Almacenados (P)")
    plt.ylabel("Patrones Recordados")
    plt.legend(); plt.grid()
    plt.savefig(os.path.join(PATH, "tarea4_capacidad.png"))

if __name__ == "__main__":
    run_all_tasks()