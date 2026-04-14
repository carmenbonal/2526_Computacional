import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================================================================
# 1. DATOS Y REESCALADO 
# ==============================================================================
M_SOLAR = 1988500.0  # (10^24 kg) [cite: 145]
AU_KM = 149.6        # 1 UA (10^6 km) [cite: 149]

# [Nombre, Masa (10^24kg), Perihelio (10^6km), Excentricidad, Periodo Real (días)] 
datos_planetas = [
    ["Sol",      M_SOLAR, 0.0,    0.0,   0.0],
    ["Mercurio", 0.330,   46.0,   0.205, 88.0],
    ["Venus",    4.87,    107.5,  0.007, 224.7],
    ["Tierra",   5.97,    147.1,  0.017, 365.2],
    ["Marte",    0.642,   206.6,  0.094, 687.0],
    ["Jupiter",  1898,    740.5,  0.049, 4331],
    ["Saturno",  568,     1352.6, 0.057, 10747],
    ["Urano",    86.8,    2741.3, 0.046, 30589],
    ["Neptuno",  102,     4444.5, 0.011, 59800]
]

n = len(datos_planetas)
m = np.array([p[1] / M_SOLAR for p in datos_planetas]) # m' [cite: 148]
r = np.zeros((n, 2))
v = np.zeros((n, 2))

for i, p in enumerate(datos_planetas):
    if i == 0: continue
    r_ua = p[2] / AU_KM
    r[i, 0] = r_ua
    # v' para órbita elíptica [cite: 188]
    v[i, 1] = np.sqrt((1 + p[3]) / r_ua)

# Configuración: t' reescalado (1 t' = 58.1 días) 
h = 0.01  
pasos = 250000 # Elevado para que Neptuno complete órbitas
t_conv = 58.1 

# ==============================================================================
# 2. MOTOR FÍSICO: VERLET EN VELOCIDAD [cite: 117, 137]
# ==============================================================================
def calcular_fisica(pos, vel, masas):
    acc = np.zeros_like(pos)
    ep = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = pos[j] - pos[i]
            dist = np.linalg.norm(diff)
            # Ecuación (4) del PDF [cite: 153]
            f = masas[j] * diff / dist**3
            acc[i] += f
            acc[j] -= (masas[i]/masas[j]) * f 
            ep -= masas[i] * masas[j] / dist
    ek = 0.5 * np.sum(masas * np.sum(vel**2, axis=1))
    return acc, ek + ep

historico_r = []
energias = []
tiempos_orbitales = [[] for _ in range(n)]

a_actual, e_ini = calcular_fisica(r, v, m)

# Bucle de integración temporal
for t_step in range(pasos):
    # Guardamos 1 de cada 500 para que la animación no pese demasiado
    if t_step % 500 == 0: historico_r.append(r.copy())
    
    # Verlet: Paso 1 (Posición) 
    r_nuevo = r + h * v + 0.5 * h**2 * a_actual
    
    # Verlet: Paso 2 (Aceleración y Energía [Objetivo 2])
    a_nueva, etot = calcular_fisica(r_nuevo, v, m)
    energias.append(etot)
    
    # Verlet: Paso 3 (Velocidad) 
    v_nueva = v + 0.5 * h * (a_actual + a_nueva)
    
    # Objetivo 3: Detección de periodos
    for i in range(1, n):
        if r[i, 1] < 0 and r_nuevo[i, 1] >= 0: # Cruce de meta
            tiempos_orbitales[i].append(t_step * h * t_conv)
            
    r, v, a_actual = r_nuevo, v_nueva, a_nueva

# ==============================================================================
# 3. RESULTADOS (Objetivos 2, 3 y 4) 
# ==============================================================================
print(f"\n{'Planeta':12} | {'Simulado':10} | {'Real':10} | {'Error Relat.'}")
print("-" * 55)
for i in range(1, n):
    if len(tiempos_orbitales[i]) > 1:
        p_sim = np.mean(np.diff(tiempos_orbitales[i]))
        p_real = datos_planetas[i][4]
        error = abs(p_sim - p_real) / p_real
        print(f"{datos_planetas[i][0]:12} | {p_sim:8.2f}d | {p_real:8.2f}d | {error:.2e}")
    else:
        print(f"{datos_planetas[i][0]:12} | Sin datos (necesita más tiempo)")

e_media = np.mean(energias)
fluct = np.std(energias) / abs(e_media)
print(f"\nConservación Energía: Media={e_media:.6f}, Fluct. Relativa={fluct:.2e}")

# ==============================================================================
# 4. ANIMACIÓN (Objetivo 1 )
# ==============================================================================
fig, ax = plt.subplots(figsize=(8,8))
ax.set_facecolor('black')
ax.set_xlim(-35, 35) 
ax.set_ylim(-35, 35)
ax.set_aspect('equal')

colores = ['yellow', 'grey', 'orange', 'blue', 'red', 'brown', 'tan', 'cyan', 'royalblue']
puntos = [ax.plot([], [], 'o', color=colores[i], label=datos_planetas[i][0])[0] for i in range(n)]

def animar(i):
    for j in range(n):
        pos = historico_r[i][j]
        puntos[j].set_data([pos[0]], [pos[1]])
    return puntos

ani = FuncAnimation(fig, animar, frames=len(historico_r), interval=1, blit=True)
plt.legend(loc='upper right', fontsize='x-small')
plt.show()