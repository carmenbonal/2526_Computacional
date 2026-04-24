import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARÁMETROS FÍSICOS
# =============================================================================

N_SISTEMAS = 80

G = 1.0

M_AGUJERO_NEGRO = 2000.0
M_SISTEMA = 1.0

R_COLISION = 0.06

# AUMENTADO: facilita absorciones reales
R_ABSORCION = 0.8

R_FRONTERA = 8.0

# AUMENTADO: más interacciones → difusión dinámica
DIST_INTERACCION = 3.0

DT = 0.005

SOFTENING = 0.1

PASOS_ESTABILIZACION = 10000
PASOS_MEDICION = 10000


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

    # Velocidades con dispersión realista
    for i in range(N_SISTEMAS):

        r = np.linalg.norm(pos[i])

        if r == 0:
            r = 1e-6

        v_circular = np.sqrt(G * M_AGUJERO_NEGRO / r)

        # ALEATORIO: evita órbitas cerradas
        factor = np.random.uniform(0.4, 0.8)

        v_mag = v_circular * factor

        direccion = np.array([-pos[i, 1], pos[i, 0]]) / r

        vel[i] = direccion * v_mag

        # perturbación aleatoria
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

    print("Fase 1: Alcanzando estado estacionario...")

    # -------------------------------------------------
    # BUCLE TEMPORAL (VERLET)
    # -------------------------------------------------

    for step in range(total_steps):

        # Kick
        v_half = vel + acc * DT / 2.0

        # Drift
        pos += v_half * DT

        # -------------------------------------------------
        # ABSORCIÓN Y REGENERACIÓN
        # -------------------------------------------------

        for i in range(N_SISTEMAS):

            r_mag = np.linalg.norm(pos[i])

            if r_mag < R_ABSORCION:

                if step >= PASOS_ESTABILIZACION:

                    absorciones += 1

                # regenerar en frontera

                theta = np.random.uniform(0, 2 * np.pi)

                pos[i] = R_FRONTERA * np.array([
                    np.cos(theta),
                    np.sin(theta)
                ])

                v_circular = np.sqrt(
                    G * M_AGUJERO_NEGRO / R_FRONTERA
                )

                factor = np.random.uniform(0.4, 0.8)

                v_reg = v_circular * factor

                direccion = np.array([
                    -pos[i, 1],
                    pos[i, 0]
                ]) / R_FRONTERA

                vel[i] = direccion * v_reg

                vel[i] += np.random.normal(0, 0.2, 2)

                v_half[i] = vel[i]

        # Colisiones

        resolver_colisiones(
            pos,
            vel,
            R_COLISION
        )

        # Aceleraciones nuevas

        acc_new = get_acc(
            pos,
            M_SISTEMA,
            M_AGUJERO_NEGRO,
            DIST_INTERACCION
        )

        vel = v_half + acc_new * DT / 2.0

        acc = acc_new

        # -------------------------------------------------
        # MEDICIONES
        # -------------------------------------------------

        if step >= PASOS_ESTABILIZACION:

            # Momento de inercia

            I = np.sum(
                M_SISTEMA *
                np.sum(pos**2, axis=1)
            )

            historial_inercia.append(I)

            # Densidad radial

            radios = np.linalg.norm(pos, axis=1)

            counts, bins = np.histogram(
                radios,
                bins=20,
                range=(0, R_FRONTERA)
            )

            areas = np.pi * (
                bins[1:]**2 - bins[:-1]**2
            )

            densidad = counts * M_SISTEMA / areas

            historial_densidad.append(densidad)

    # -------------------------------------------------
    # RESULTADOS
    # -------------------------------------------------

    tiempo_total = PASOS_MEDICION * DT

    flujo_masa = (
        absorciones *
        M_SISTEMA
    ) / tiempo_total

    densidad_media = np.mean(
        historial_densidad,
        axis=0
    )

    inercia_media = np.mean(
        historial_inercia
    )

    bin_centers = (
        bins[:-1] + bins[1:]
    ) / 2

    print("\n--- RESULTADOS ---")

    print(
        "Absorciones totales:",
        absorciones
    )

    print(
        "Tiempo total:",
        tiempo_total
    )

    print(
        "Flujo medio de masa:",
        flujo_masa
    )

    print(
        "Momento de inercia medio:",
        inercia_media
    )

    # -------------------------------------------------
    # GRÁFICAS
    # -------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    ax1.plot(
        bin_centers,
        densidad_media
    )

    ax1.set_title(
        "Distribución radial de densidad"
    )

    ax1.set_xlabel("Radio")

    ax1.set_ylabel("Densidad")

    ax2.plot(
        historial_inercia
    )

    ax2.set_title(
        "Evolución del momento de inercia"
    )

    ax2.set_xlabel(
        "Pasos de medición"
    )

    ax2.set_ylabel("I")

    plt.tight_layout()

    plt.savefig(
        "analisis_galaxia.png"
    )

    plt.show()


if __name__ == "__main__":
    main()