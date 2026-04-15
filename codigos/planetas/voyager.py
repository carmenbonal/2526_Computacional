import numpy as np

# ==============================================================================
# 1. CONSTANTES Y DATOS (NASA Fact Sheet)
# ==============================================================================
M_SOLAR = 1988500.0  
AU_KM = 149.6        

# [Nombre, Masa (10^24kg), Perihelio (10^6km), Excentricidad, Periodo Real (días)]
datos_planetas = [
    ["Sol",      M_SOLAR, 0.0,    0.0,   0.0],
    ["Mercurio", 0.330,   46.0,   0.205, 88.0],
    ["Venus",    4.87,    107.5,  0.007, 224.7],
    ["Tierra",   5.97,    147.1,  0.017, 365.2],
    ["Marte",    0.642,   206.6,  0.094, 687.0],
    ["Jupiter",  1898,    740.5,  0.049, 4331.0],
    ["Saturno",  568,     1352.6, 0.057, 10747.0],
    ["Urano",    86.8,    2741.3, 0.046, 30589.0],
    ["Neptuno",  102,     4444.5, 0.011, 59800.0],
    ["Voyager",  0.0000008, 147.1, 0.0,   0.0] # Masa despreciable
]

# ==============================================================================
# 2. CONFIGURACIÓN DE LA SIMULACIÓN
# ==============================================================================
file_out = "trayectorias_sistema_solar.dat"
h = 0.01             # Paso de tiempo
pasos = 250000       # Suficiente para que Neptuno y Voyager avancen
t_conv = 58.1        # 1 unidad t' = 58.1 días
frame_skip = 500     # Guardar 1 posición de cada 500 para el archivo

# ==============================================================================
# 3. INICIALIZACIÓN DE ESTADOS
# ==============================================================================
n = len(datos_planetas)
m = np.array([p[1] / M_SOLAR for p in datos_planetas])
r = np.zeros((n, 2))
v = np.zeros((n, 2))

for i, p in enumerate(datos_planetas):
    if i == 0: continue
    
    r_ua = p[2] / AU_KM
    e = p[3]
    r[i, 0] = r_ua
    
    # Velocidades iniciales (Jupiter normal y Voyager a escape)
    if p[0] == "Voyager":
        v[i, 1] = 1.5 # Hiperbólica
    else:
        v[i, 1] = np.sqrt((1.0 + e) / r_ua)

# ==============================================================================
# 4. MOTOR FÍSICO (VERLET EN VELOCIDAD)
# ==============================================================================
def calcular_fisica(pos, vel, masas):
    acc = np.zeros_like(pos)
    ep = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = pos[j] - pos[i]
            dist = np.linalg.norm(diff)
            if dist == 0: continue
            
            f = masas[j] * diff / dist**3
            acc[i] += f
            acc[j] -= (masas[i]/masas[j]) * f 
            ep -= masas[i] * masas[j] / dist
    ek = 0.5 * np.sum(masas * np.sum(vel**2, axis=1))
    return acc, ek + ep

historico_r = []
energias = []
tiempos_orbitales = [[] for _ in range(n)]

a_actual, _ = calcular_fisica(r, v, m)

print("Calculando trayectorias...")

for t_step in range(pasos):
    if t_step % frame_skip == 0:
        historico_r.append(r.copy())
    
    # Integración de Verlet
    r_nuevo = r + h * v + 0.5 * h**2 * a_actual
    a_nueva, etot = calcular_fisica(r_nuevo, v, m)
    v_nueva = v + 0.5 * h * (a_actual + a_nueva)
    
    # Detección de periodos
    for i in range(1, n):
        if r[i, 1] < 0 and r_nuevo[i, 1] >= 0:
            tiempos_orbitales[i].append(t_step * h * t_conv)
            
    r, v, a_actual = r_nuevo, v_nueva, a_nueva
    energias.append(etot)

# ==============================================================================
# 5. SALIDA DE DATOS (POSIX)
# ==============================================================================
print(f"Escribiendo datos en {file_out}...")
with open(file_out, "w") as f:
    for frame in historico_r:
        for pos in frame:
            f.write(f"{pos[0]:.6f} {pos[1]:.6f}\n")
        f.write("\n") # Separador de frames

# Resultados por consola
print(f"\n{'Planeta':12} | {'Simulado':10} | {'Real':10} | {'Error Relat.'}")
print("-" * 55)
for i in range(1, n):
    if len(tiempos_orbitales[i]) > 1:
        p_sim = np.mean(np.diff(tiempos_orbitales[i]))
        p_real = datos_planetas[i][4]
        if p_real > 0:
            error = abs(p_sim - p_real) / p_real
            print(f"{datos_planetas[i][0]:12} | {p_sim:8.2f}d | {p_real:8.2f}d | {error:.2e}")
        else:
            print(f"{datos_planetas[i][0]:12} | {p_sim:8.2f}d | Traject. abierta")

e_media = np.mean(energias)
fluct = np.std(energias) / abs(e_media)
print(f"\nConservación Energía: Media={e_media:.6f}, Fluct. Relativa={fluct:.2e}")