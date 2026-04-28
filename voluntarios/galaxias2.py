import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

PASOS_ESTABILIZACION = 10000
PASOS_MEDICION = 10000

# Longitud de la cola de trayectorias
COLA = 200


# =============================================================================
# ACELERACIONES
# =============================================================================

def get_acc(pos, mass, m_bh, r_limit):

    acc = np.zeros_like(pos)
    r_mag_bh = np.linalg.norm(pos, axis=1).reshape(-1, 1)

    # Gravedad del agujero negro
    acc -= G * m_bh * pos / (r_mag_bh**3 + SOFTENING)

    # Interacciones locales
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):

            diff = pos[j] - pos[i]
            dist = np.linalg.norm(diff)

            if dist < r_limit and dist > 0:
                force = G * mass * diff / (dist**3 + SOFTENING)
                acc[i] += force
                acc[j] -= force

    return acc


# =============================================================================
# COLISIONES ELÁSTICAS
# =============================================================================

def resolver_colisiones(pos, vel, r_col):

    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):

            diff = pos[j] - pos[i]
            dist = np.linalg.norm(diff)

            if dist < 2 * r_col and dist > 0:
                normal = diff / dist
                rel_vel = vel[j] - vel[i]
                v_impulse = np.dot(rel_vel, normal)

                if v_impulse < 0:
                    vel[i] += v_impulse * normal
                    vel[j] -= v_impulse * normal


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():

    # -------------------------------------------------
    # INICIALIZACIÓN
    # -------------------------------------------------

    pos = np.random.uniform(
        -R_FRONTERA * 0.6,
        R_FRONTERA * 0.6,
        (N_SISTEMAS, 2)
    )

    vel = np.zeros_like(pos)

    for i in range(N_SISTEMAS):
        r = np.linalg.norm(pos[i])
        if r == 0:
            r = 1e-6

        v_circular = np.sqrt(G * M_AGUJERO_NEGRO / r)
        factor = np.random.uniform(0.4, 0.8)
        v_mag = v_circular * factor
        direccion = np.array([-pos[i, 1], pos[i, 0]]) / r
        
        vel[i] = direccion * v_mag
        vel[i] += np.random.normal(0, 0.05, 2)

    acc = get_acc(
        pos,
        M_SISTEMA,
        M_AGUJERO_NEGRO,
        DIST_INTERACCION
    )

    absorciones = 0
    historial_inercia = []
    historial_densidad = []
    total_steps = PASOS_ESTABILIZACION + PASOS_MEDICION

    pos_save = np.zeros((N_SISTEMAS, 2, COLA))

    print("Fase 1: Calculando simulación y fotogramas...")

    # Configuración de la figura para la animación
    fig, ax_anim = plt.subplots(figsize=(6, 6))
    ax_anim.set_xlim(-R_FRONTERA, R_FRONTERA)
    ax_anim.set_ylim(-R_FRONTERA, R_FRONTERA)
    ax_anim.set_aspect("equal")
    
    fotogramas = []

    # -------------------------------------------------
    # BUCLE TEMPORAL (VERLET)
    # -------------------------------------------------

    for step in range(total_steps):

        v_half = vel + acc * DT / 2.0
        pos += v_half * DT

        idx = step % COLA
        pos_save[:, :, idx] = pos

        # -------------------------------------------------
        # ABSORCIÓN Y REGENERACIÓN
        # -------------------------------------------------

        for i in range(N_SISTEMAS):
            r_mag = np.linalg.norm(pos[i])

            if r_mag < R_ABSORCION:
                if step >= PASOS_ESTABILIZACION:
                    absorciones += 1

                theta = np.random.uniform(0, 2 * np.pi)
                pos[i] = R_FRONTERA * np.array([np.cos(theta), np.sin(theta)])
                v_circular = np.sqrt(G * M_AGUJERO_NEGRO / R_FRONTERA)
                factor = np.random.uniform(0.4, 0.8)
                v_reg = v_circular * factor
                direccion = np.array([-pos[i, 1], pos[i, 0]]) / R_FRONTERA
                
                vel[i] = direccion * v_reg
                vel[i] += np.random.normal(0, 0.2, 2)
                v_half[i] = vel[i]

        resolver_colisiones(
            pos,
            v_half,
            R_COLISION
        )

        acc_new = get_acc(
            pos,
            M_SISTEMA,
            M_AGUJERO_NEGRO,
            DIST_INTERACCION
        )

        vel = v_half + acc_new * DT / 2.0
        acc = acc_new

        # -------------------------------------------------
        # VISUALIZACIÓN DE TRAYECTORIAS (Guardado de fotogramas)
        # -------------------------------------------------

        if step % 40 == 0:  # Reducida la frecuencia para no generar vídeos excesivamente pesados

            elementos_fotograma = []

            for i in range(N_SISTEMAS):
                if step < COLA:
                    trayectoria = pos_save[i, :, :step]
                else:
                    trayectoria = np.concatenate((
                        pos_save[i, :, idx:],
                        pos_save[i, :, :idx]
                    ), axis=1)

                if trayectoria.shape[1] > 0:
                    pts_tray = ax_anim.scatter(trayectoria[0], trayectoria[1], s=3, color='blue', alpha=0.3)
                    elementos_fotograma.append(pts_tray)

            pts_pos = ax_anim.scatter(pos[:, 0], pos[:, 1], s=10, color='red')
            pts_bh = ax_anim.scatter(0, 0, s=80, color='black')
            
            elementos_fotograma.extend([pts_pos, pts_bh])
            fotogramas.append(elementos_fotograma)

        # -------------------------------------------------
        # MEDICIONES
        # -------------------------------------------------

        if step >= PASOS_ESTABILIZACION:

            I = np.sum(M_SISTEMA * np.sum(pos**2, axis=1))
            historial_inercia.append(I)

            radios = np.linalg.norm(pos, axis=1)
            counts, bins = np.histogram(radios, bins=20, range=(0, R_FRONTERA))
            areas = np.pi * (bins[1:]**2 - bins[:-1]**2)
            densidad = counts * M_SISTEMA / areas
            historial_densidad.append(densidad)

    # -------------------------------------------------
    # COMPILACIÓN DEL VÍDEO
    # -------------------------------------------------
    print("Compilando animación...")
    ani = animation.ArtistAnimation(fig, fotogramas, interval=50, blit=True)
    ani.save("simulacion.gif", writer="pillow")
    plt.close(fig) # Evita que se muestre una figura vacía al final

    # -------------------------------------------------
    # RESULTADOS
    # -------------------------------------------------

    tiempo_total = PASOS_MEDICION * DT
    flujo_masa = (absorciones * M_SISTEMA) / tiempo_total
    densidad_media = np.mean(historial_densidad, axis=0)
    inercia_media = np.mean(historial_inercia)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    print("\n--- RESULTADOS ---")
    print("Absorciones totales:", absorciones)
    print("Tiempo total:", tiempo_total)
    print("Flujo medio de masa:", flujo_masa)
    print("Momento de inercia medio:", inercia_media)

    # -------------------------------------------------
    # GRÁFICAS FINALES
    # -------------------------------------------------

    fig_res, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(bin_centers, densidad_media)
    ax1.set_title("Distribución radial de densidad")
    ax1.set_xlabel("Radio")
    ax1.set_ylabel("Densidad")

    ax2.plot(historial_inercia)
    ax2.set_title("Evolución del momento de inercia")
    ax2.set_xlabel("Pasos de medición")
    ax2.set_ylabel("I")

    plt.tight_layout()
    plt.savefig("analisis_galaxia.png")
    plt.show()

if __name__ == "__main__":
    main()