import numpy as np

# ==============================================================================
# 1. DATOS Y REESCALADO
# ==============================================================================
M_SOLAR = 1988500.0   # masa solar en unidades de 10^24 kg
AU_KM = 149.6         # 1 UA en unidades de 10^6 km

# [Nombre, Masa (10^24 kg), Perihelio (10^6 km), Excentricidad, Periodo real (días)]
datos_planetas = [
    ["Sol",      M_SOLAR, 0.0,    0.0,   0.0],
    ["Mercurio", 0.330,   46.0,   0.205, 88.0],
    ["Venus",    4.87,    107.5,  0.007, 224.7],
    ["Tierra",   5.97,    147.1,  0.017, 365.2],
    ["Marte",    0.642,   206.6,  0.094, 687.0],
    ["Jupiter",  1898,    740.5,  0.049, 4331.0],
    ["Saturno",  568,     1352.6, 0.057, 10747.0],
    ["Urano",    86.8,    2741.3, 0.046, 30589.0],
    ["Neptuno",  102,     4444.5, 0.011, 59800.0]
]

# ==============================================================================
# 2. PARÁMETROS DE SIMULACIÓN
# ==============================================================================
file_out = "planets_data.dat"

# Paso temporal reescalado
h = 0.002

# Número de pasos de simulación
pasos = 200000

# Conversión de tiempo reescalado a días
# 1 t' = 58.1 días
t_conv = 58.1

# Guardar 1 frame cada frame_skip pasos
frame_skip = 300

# ==============================================================================
# 3. CONSTRUCCIÓN DE MAGNITUDES INICIALES
# ==============================================================================
n = len(datos_planetas)

# Masas reescaladas en masas solares
m = np.array([p[1] / M_SOLAR for p in datos_planetas])

# Posiciones y velocidades en 2D
r = np.zeros((n, 2))
v = np.zeros((n, 2))

# Condiciones iniciales simplificadas:
# - todos los planetas arrancan en el perihelio sobre el eje x
# - velocidad inicial tangencial sobre el eje y
for i, p in enumerate(datos_planetas):
    if i == 0:
        continue  # el Sol se queda inicialmente en el origen

    r_ua = p[2] / AU_KM      # perihelio en UA
    e = p[3]                 # excentricidad

    r[i, 0] = r_ua
    r[i, 1] = 0.0

    # velocidad inicial reescalada en perihelio
    v[i, 0] = 0.0
    v[i, 1] = np.sqrt((1.0 + e) / r_ua)

# ==============================================================================
# 4. FUNCIÓN FÍSICA: ACELERACIONES Y ENERGÍA
# ==============================================================================
def calcular_fisica(pos, vel, masas):
    n_local = len(masas)
    acc = np.zeros_like(pos)
    ep = 0.0

    for i in range(n_local):
        for j in range(i + 1, n_local):
            diff = pos[j] - pos[i]
            dist = np.linalg.norm(diff)

            # Evita divisiones por cero por seguridad
            if dist == 0:
                continue

            # Contribución gravitatoria reescalada
            f = masas[j] * diff / dist**3

            # Aceleración del cuerpo i
            acc[i] += f

            # Aceleración del cuerpo j
            acc[j] -= (masas[i] / masas[j]) * f

            # Energía potencial gravitatoria
            ep -= masas[i] * masas[j] / dist

    # Energía cinética
    ek = 0.5 * np.sum(masas * np.sum(vel**2, axis=1))

    return acc, ek + ep

# ==============================================================================
# 5. BUCLE DE INTEGRACIÓN TEMPORAL (VELOCITY VERLET)
# ==============================================================================
historico_r = []
energias = []
tiempos_orbitales = [[] for _ in range(n)]

a_actual, _ = calcular_fisica(r, v, m)

for t_step in range(pasos):
    # Guardar frame para la animación
    if t_step % frame_skip == 0:
        historico_r.append(r.copy())

    # Paso 1: actualizar posición
    r_nuevo = r + h * v + 0.5 * h**2 * a_actual

    # Paso 2: nueva aceleración
    a_nueva, _ = calcular_fisica(r_nuevo, v, m)

    # Paso 3: actualizar velocidad
    v_nueva = v + 0.5 * h * (a_actual + a_nueva)

    # Energía del estado ya actualizado
    _, etot = calcular_fisica(r_nuevo, v_nueva, m)
    energias.append(etot)

    # Detección aproximada de periodos orbitales:
    # cruce del eje x desde y<0 hasta y>=0
    for i in range(1, n):
        if r[i, 1] < 0 and r_nuevo[i, 1] >= 0:
            tiempos_orbitales[i].append(t_step * h * t_conv)

    # Avanzar al siguiente estado
    r, v, a_actual = r_nuevo, v_nueva, a_nueva

# ==============================================================================
# 6. ESCRITURA DEL ARCHIVO .DAT (OPCIÓN A: DESDE historico_r)
# ==============================================================================
with open(file_out, "w") as f:
    for i_frame, frame in enumerate(historico_r):
        for planeta_pos in frame:
            x, y = planeta_pos
            f.write(f"{x}, {y}\n")

        # Línea en blanco entre frames, excepto quizá al final
        if i_frame != len(historico_r) - 1:
            f.write("\n")

print(f"\nArchivo de datos generado: {file_out}")

# ==============================================================================
# 7. RESULTADOS NUMÉRICOS
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
print(f"Frames guardados para animación: {len(historico_r)}")