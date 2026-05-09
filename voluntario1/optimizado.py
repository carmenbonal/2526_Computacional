import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from numba import njit

# =============================================================================
# PARÁMETROS FÍSICOS
# =============================================================================
N_SISTEMAS = 80
G = 1.0
M_AGUJERO_NEGRO = 2000.0
M_SISTEMA = 1.0
R_COLISION = 0.06
R_ABSORCION = 0.8
R_FRONTERA = 8.0
DIST_INTERACCION = 3.0
DT = 0.005
SOFTENING = 0.1
PASOS_ESTABILIZACION = 5000  # Ajustado para que no tarde una eternidad
PASOS_MEDICION = 5000
COLA = 200

# =============================================================================
# FUNCIONES OPTIMIZADAS (CÁLCULO PURO)
# =============================================================================

@njit(fastmath=True)
def get_acc_numba(pos, mass, m_bh, r_limit, softening, G_const):
    n = pos.shape[0]
    acc = np.zeros_like(pos)
    for i in range(n):
        rx, ry = pos[i, 0], pos[i, 1]
        r_sq = rx**2 + ry**2
        r_mag = np.sqrt(r_sq)
        factor_bh = -G_const * m_bh / (r_sq * r_mag + softening)
        acc[i, 0], acc[i, 1] = rx * factor_bh, ry * factor_bh

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
    pos = np.random.uniform(-R_FRONTERA * 0.6, R_FRONTERA * 0.6, (N_SISTEMAS, 2))
    vel = np.zeros_like(pos)
    for i in range(N_SISTEMAS):
        r = np.sqrt(pos[i,0]**2 + pos[i,1]**2) or 1e-6
        v_mag = np.sqrt(G * M_AGUJERO_NEGRO / r) * np.random.uniform(0.4, 0.8)
        vel[i] = np.array([-pos[i, 1], pos[i, 0]]) / r * v_mag + np.random.normal(0, 0.05, 2)

    acc = get_acc_numba(pos, M_SISTEMA, M_AGUJERO_NEGRO, DIST_INTERACCION, SOFTENING, G)
    total_steps = PASOS_ESTABILIZACION + PASOS_MEDICION
    pos_save = np.zeros((N_SISTEMAS, 2, COLA))
    
    # --- VARIABLES DE CRONÓMETRO ---
    tiempo_fisica_acumulado = 0.0
    fotogramas = []
    fig, ax_anim = plt.subplots(figsize=(6, 6))

    print(f"Calculando {total_steps} pasos...")

    for step in range(total_steps):
        
        # >>> INICIO CRONÓMETRO FÍSICA >>>
        t0 = time.time()
        
        # 1. Integración
        v_half = vel + acc * DT / 2.0
        pos += v_half * DT
        
        # 2. Absorción y Fronteras
        for i in range(N_SISTEMAS):
            r_mag = np.sqrt(pos[i,0]**2 + pos[i,1]**2)
            if r_mag < R_ABSORCION or r_mag > R_FRONTERA:
                theta = np.random.uniform(0, 2 * np.pi)
                pos[i] = R_FRONTERA * np.array([np.cos(theta), np.sin(theta)])
                v_reg = np.sqrt(G * M_AGUJERO_NEGRO / R_FRONTERA) * np.random.uniform(0.4, 0.8)
                vel[i] = np.array([-pos[i, 1], pos[i, 0]]) / R_FRONTERA * v_reg
                v_half[i] = vel[i]

        # 3. Colisiones y Aceleración nueva
        resolver_colisiones_numba(pos, v_half, R_COLISION)
        acc_new = get_acc_numba(pos, M_SISTEMA, M_AGUJERO_NEGRO, DIST_INTERACCION, SOFTENING, G)
        vel = v_half + acc_new * DT / 2.0
        acc = acc_new
        
        # <<< FIN CRONÓMETRO FÍSICA <<<
        tiempo_fisica_acumulado += (time.time() - t0)

        # GESTIÓN DE FOTOGRAMAS (Esto NO se cuenta en el cronómetro de física)
        if step % 50 == 0:
            idx = step % COLA
            pos_save[:, :, idx] = pos.copy()
            elementos = []
            p_pos = ax_anim.scatter(pos[:, 0], pos[:, 1], s=10, color='red', animated=True)
            p_bh = ax_anim.scatter(0, 0, s=80, color='black', animated=True)
            elementos.extend([p_pos, p_bh])
            fotogramas.append(elementos)

    print(f"\n--- INFORME DE RENDIMIENTO ---")
    print(f"Tiempo invertido en CÁLCULO FÍSICO: {tiempo_fisica_acumulado:.4f} segundos")
    
    t_ani_start = time.time()
    print("Generando GIF (esto es lento y NO es culpa de la física)...")
    ani = animation.ArtistAnimation(fig, fotogramas, interval=50, blit=True)
    ani.save("simulacion_opt.gif", writer="pillow")
    print(f"Tiempo invertido en GENERAR GIF: {time.time() - t_ani_start:.2f} segundos")

if __name__ == "__main__":
    main()