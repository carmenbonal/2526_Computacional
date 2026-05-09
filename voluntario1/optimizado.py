import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
from numba import njit

# =============================================================================
# PARÁMETROS FÍSICOS (VALORES REALISTAS)
# =============================================================================
N_SISTEMAS = 80
G = 1.0 
M_AGUJERO_NEGRO = 4.1e6  # Masa de Sagitario A* en masas solares
M_SISTEMA = 1.0          # Masa de un sistema estelar (1 masa solar)
R_COLISION = 0.06
R_ABSORCION = 0.5
R_FRONTERA = 15.0        # Ampliado para la escala galáctica
DIST_INTERACCION = 5.0
DT = 0.0005              # Reducido para manejar la enorme fuerza central
SOFTENING = 0.2          # Aumentado para evitar divergencias numéricas
PASOS_ESTABILIZACION = 5000 
PASOS_MEDICION = 5000
COLA = 150

# RUTA DE GUARDADO
OUTPUT_DIR = r"C:\Users\Usuario\Desktop\2526_Computacional\voluntario1"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# =============================================================================
# FUNCIONES OPTIMIZADAS
# =============================================================================

@njit(fastmath=True)
def get_acc_numba(pos, mass, m_bh, r_limit, softening, G_const):
    n = pos.shape[0]
    acc = np.zeros_like(pos)
    for i in range(n):
        rx, ry = pos[i, 0], pos[i, 1]
        r_sq = rx**2 + ry**2
        r_mag = np.sqrt(r_sq)
        # Fuerza del Agujero Negro Central
        factor_bh = -G_const * m_bh / (r_sq * r_mag + softening)
        acc[i, 0], acc[i, 1] = rx * factor_bh, ry * factor_bh

        # Interacciones entre sistemas (N-cuerpos limitado por r_limit)
        for j in range(i + 1, n):
            dx, dy = pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1]
            d_sq = dx**2 + dy**2
            dist = np.sqrt(d_sq)
            if dist < r_limit and dist > 0:
                f_mag = G_const * mass / (d_sq * dist + softening)
                acc[i, 0] += f_mag * dx; acc[i, 1] += f_mag * dy
                acc[j, 0] -= f_mag * dx; acc[j, 1] -= f_mag * dy
    return acc

@njit(fastmath=True)
def resolver_colisiones_numba(pos, vel, r_col):
    n, r_double = pos.shape[0], 2 * r_col
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1]
            d_sq = dx**2 + dy**2
            if d_sq < r_double**2 and d_sq > 0:
                dist = np.sqrt(d_sq)
                nx, ny = dx / dist, dy / dist
                rvx, rvy = vel[j, 0] - vel[i, 0], vel[j, 1] - vel[i, 1]
                v_imp = rvx * nx + rvy * ny
                if v_imp < 0:
                    vel[i, 0] += v_imp * nx; vel[i, 1] += v_imp * ny
                    vel[j, 0] -= v_imp * nx; vel[j, 1] -= v_imp * ny

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():
    # Inicialización de posiciones y velocidades circulares
    pos = np.random.uniform(-R_FRONTERA * 0.7, R_FRONTERA * 0.7, (N_SISTEMAS, 2))
    vel = np.zeros_like(pos)
    for i in range(N_SISTEMAS):
        r = np.sqrt(pos[i,0]**2 + pos[i,1]**2) or 1e-6
        v_mag = np.sqrt(G * M_AGUJERO_NEGRO / r) * np.random.uniform(0.7, 0.9)
        vel[i] = np.array([-pos[i, 1], pos[i, 0]]) / r * v_mag + np.random.normal(0, 0.1, 2)

    acc = get_acc_numba(pos, M_SISTEMA, M_AGUJERO_NEGRO, DIST_INTERACCION, SOFTENING, G)
    total_steps = PASOS_ESTABILIZACION + PASOS_MEDICION
    
    historial_inercia = []
    historial_densidad = []
    fotogramas = []
    
    fig_anim, ax_anim = plt.subplots(figsize=(7, 7))
    print(f"Calculando {total_steps} pasos")

    # Bucle Temporal (Verlet en Velocidad)
    for step in range(total_steps):
        # 1. Actualización de posición
        v_half = vel + acc * DT / 2.0
        pos += v_half * DT
        
        # 2. Gestión de fronteras y absorción
        for i in range(N_SISTEMAS):
            r_mag = np.sqrt(pos[i,0]**2 + pos[i,1]**2)
            if r_mag < R_ABSORCION or r_mag > R_FRONTERA:
                theta = np.random.uniform(0, 2 * np.pi)
                pos[i] = R_FRONTERA * np.array([np.cos(theta), np.sin(theta)])
                v_reg = np.sqrt(G * M_AGUJERO_NEGRO / R_FRONTERA) * np.random.uniform(0.8, 0.95)
                vel[i] = np.array([-pos[i, 1], pos[i, 0]]) / R_FRONTERA * v_reg
                v_half[i] = vel[i]

        # 3. Colisiones y nueva aceleración
        resolver_colisiones_numba(pos, v_half, R_COLISION)
        acc_new = get_acc_numba(pos, M_SISTEMA, M_AGUJERO_NEGRO, DIST_INTERACCION, SOFTENING, G)
        vel = v_half + acc_new * DT / 2.0
        acc = acc_new

        # 4. Toma de datos en fase de medición
        if step >= PASOS_ESTABILIZACION:
            # Momento de inercia: I = sum(m * r^2)
            I = np.sum(M_SISTEMA * np.sum(pos**2, axis=1))
            historial_inercia.append(I)
            
            # Densidad radial
            radios = np.sqrt(np.sum(pos**2, axis=1))
            counts, bins = np.histogram(radios, bins=25, range=(0, R_FRONTERA))
            areas = np.pi * (bins[1:]**2 - bins[:-1]**2)
            densidad = counts * M_SISTEMA / areas
            historial_densidad.append(densidad)

        # Captura de fotogramas para el GIF
        if step % 80 == 0:
            p_pos = ax_anim.scatter(pos[:, 0], pos[:, 1], s=8, color='crimson', animated=True)
            p_bh = ax_anim.scatter(0, 0, s=100, color='black', animated=True)
            fotogramas.append([p_pos, p_bh])

    # --- GUARDADO DE RESULTADOS ---
    
    # 1. Guardar GIF
    print("Generando simulación...")
    ax_anim.set_xlim(-R_FRONTERA, R_FRONTERA)
    ax_anim.set_ylim(-R_FRONTERA, R_FRONTERA)
    ax_anim.set_title("Dinámica Galáctica (Verlet + Numba)")
    ani = animation.ArtistAnimation(fig_anim, fotogramas, interval=40, blit=True)
    ani.save(os.path.join(OUTPUT_DIR, "simulacion_opt.gif"), writer="pillow")
    plt.close(fig_anim)

    # 2. Generar y guardar Gráfica de Análisis
    print("Generando gráficas de análisis...")
    densidad_media = np.mean(historial_densidad, axis=0)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    fig_res, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(bin_centers, densidad_media, color='blue', lw=2)
    ax1.set_title("Perfil de Densidad Radial")
    ax1.set_xlabel("Distancia al Centro (r)"); ax1.set_ylabel("Densidad")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(historial_inercia, color='green')
    ax2.set_title("Evolución del Momento de Inercia (I)")
    ax2.set_xlabel("Pasos de medición"); ax2.set_ylabel("I")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "analisis_galaxia_opt.png"))
    print(f"Éxito. Archivos guardados.")

if __name__ == "__main__":
    main()