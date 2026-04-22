import numpy as np # Para el cálculo científico
import matplotlib.pyplot as plt # Para visualización

# Dudas: como al ser absorbidos los sistemas, y luego son todos generados justo en la frontera, 
# si la tasa de nacimiento es mayor que la tasa de muerte se quedan orbitando cerca de la frontera, 
# lo que genera un pico en la densidad radial. Sería mejor que, en vez de que nazcan todos en R=25,
# nacieran en un rango entre R=20 y R=25, para evitar esa acumulación?



# --- Configuración del Modelo  ---
N_SISTEMAS = 100    #Cantidad de sistemas solares simulados
G = 1.0             # Constante gravitacional en unidades naturales
M_BH = 5000.0        # Masa del agujero negro central
M_SYS = 1.0          # Masa de cada sistema solar
DT = 0.01               # Paso de tiempo
PASOS_RELAJACION = 5000  # Tiempo para alcanzar el estado estacionario
PASOS_MEDICION = 5000    # Mediciones en estado estacionario
R_MAX = 25.0           # Radio máximo de la simulación
R_ABS = 1          # Radio de absorción del agujero negro

# Añadimos un tiempo de relajación. Esto se debe a que los sistemas solares
# necesitan tiempo para alcanzar el estado estacionario. Durante la relajación.
# el sistema es intestable y los observables oscilan bruscamente.

# --- Inicialización ---
pos = np.random.uniform(-R_MAX/2, R_MAX/2, (N_SISTEMAS, 2))
vel = np.zeros_like(pos)

# Damos velocidad inicial para que no caigan directamente en el agujero negro, 
# usando la fórmula de v obtenida de igualar la fuerza centrípeta y la gravitatoria
# (órbita circular al menos al principio para q no colapse directamente). La excentricidad
# se da de forma natural con el tiempo con la interacción gravitatoria. 
# Las posiciones iniciales también se dan de forma aleatoria.
for i in range(N_SISTEMAS):
    r_mag = np.linalg.norm(pos[i])
    v_mag = np.sqrt(G * M_BH / r_mag)
    vel[i] = np.array([-pos[i, 1], pos[i, 0]]) / r_mag * v_mag
    # [-y, x] es un vector perpendicular, que le da velocidad tangencial al
    # cuerpo para que sea capaz de orbitar. Al dividir por r_mag unitarizamos
    # el vector, y al multiplicar por v_mag le damos la v justa para mantener
    # una órbita circular. 


# Calculo aceleración con la ley de Newton en cada instante. 
def calcular_aceleracion(p):
    acc = np.zeros_like(p)
    # Gravedad central (Agujero Negro)
    dist_sq = np.sum(p**2, axis=1).reshape(-1, 1)
    dist = np.sqrt(dist_sq)
    acc -= G * M_BH * p / (dist**3)
    
    # Interacción entre sistemas cercanos: Solo calculamos la gravedad 
    # entre 2 sistemas si están a una distancia menor a 3 unidades.
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            diff = p[j] - p[i]
            d_ij = np.linalg.norm(diff)
            if d_ij < 3.0: # Umbral de cercanía
                f = G * M_SYS * diff / (d_ij**3)
                acc[i] += f
                acc[j] -= f
    return acc

# --- Contenedores para el análisis ---
historial_inercia = []
radios_estacionarios = []
absorciones = 0

# --- Bucle de Evolución (Verlet en Velocidad) ---
total_pasos = PASOS_RELAJACION + PASOS_MEDICION

for paso in range(total_pasos):
    a_t = calcular_aceleracion(pos)
    
    # Actualización con algoritmo Verlet
    pos_nueva = pos + DT * vel + 0.5 * (DT**2) * a_t
    w = vel + 0.5 * DT * a_t
    a_t_h = calcular_aceleracion(pos_nueva)
    vel_nueva = w + 0.5 * DT * a_t_h
    
    pos, vel = pos_nueva, vel_nueva

    # Gestión de absorción y regeneración: la masa que entra debe ser igual a 
    # la que sale para mantener el número constante de sistemas solares.    
    for i in range(N_SISTEMAS):
        r_i = np.linalg.norm(pos[i])
        # Si el sistema solar se acerca demasiado al agujero o se aleja demasiado, 
        # lo consideramos absorbido o perdido.
        if r_i < R_ABS or r_i > R_MAX * 1.5:
            if r_i < R_ABS: absorciones += 1
            # Regenerar en la frontera con órbita cerrada
            ang = np.random.uniform(0, 2*np.pi)
            pos[i] = np.array([np.cos(ang), np.sin(ang)]) * R_MAX
            v_orb = np.sqrt(G * M_BH / R_MAX)
            vel[i] = np.array([-np.sin(ang), np.cos(ang)]) * v_orb

    # Análisis post-relajación 
    inercia = np.sum(M_SYS * np.sum(pos**2, axis=1))
    historial_inercia.append(inercia)
    
    if paso > PASOS_RELAJACION:
        radios_estacionarios.extend(np.linalg.norm(pos, axis=1))

# --- Resultados del Análisis ---
# 1. Flujo medio de masa: cuanto material es absorbido por el agujero negro por unidad de tiempo en estado estacionario.
flujo = (absorciones * M_SYS) / (total_pasos * DT)

# 2. Densidad radial media 
# Cuento cuántos sistemas solares hay en cada anillo radial y divido por
#  el área de ese anillo para obtener la densidad.
counts, bins = np.histogram(radios_estacionarios, bins=30, range=(0, R_MAX))
areas = np.pi * (bins[1:]**2 - bins[:-1]**2)
densidad_radial = (counts / (PASOS_MEDICION)) / areas

# --- Visualización ---
plt.figure(figsize=(12, 5))


# Si el sistema es estable, el valor del momento de inercia no debería crecer ni
# disminuir sistemáticamente. Si esto ocurre es que el sistema no ha llegado a 
# alcanzar el estado estacionario. En este caso, aumentar el valor de pasos de relajación.
# Si la tasa de 'nacimientos' es mayor que la tasa de 'muertes', la inercia total no va a dejar de subir.
plt.subplot(1, 3, 1)
plt.plot(historial_inercia)
plt.axvline(PASOS_RELAJACION, color='r', linestyle='--', label='Fin Relajación')
plt.title("Momento de Inercia ")
plt.legend()


# Muestra masa por ud. de área. La mayor parte de la masa debería estar concentrada 
# cerca del agujero (a la izquierda del gráfico) y luego decaer a medida que nos alejamos.
# Si vemos un pico a la derecha, se debe a que cuando un sistema solar es absorbido, 
# se inserta uno nuevo en la frontera con órbita circular, lo que genera una acumulación de sistemas solares en esa región.
# Al 'nacer' con velocidad circular en ese radio, los sistemas tienden a quedarse orbitando cerca de donde nacieron.
plt.subplot(1, 3, 2)
plt.bar(bins[:-1], densidad_radial, width=np.diff(bins), align='edge')
plt.title("Densidad de Masa Radial ")


# Mapa de posiciones finales de los sistemas solares. Deberíamos ver una concentración de puntos
#  cerca del centro (agujero negro) y luego una distribución más dispersa a medida que nos alejamos. 
# Subplot 3: Mapa de posiciones con el Agujero Negro
plt.subplot(1, 3, 3)
plt.scatter(pos[:, 0], pos[:, 1], s=10, alpha=0.6, label='Sistemas Solares') 
plt.title("Mapa de Posiciones Final")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# Añadir el Flujo Medio como texto en la figura para que no se pierda
info_texto = f"Flujo Medio de Masa: {flujo:.6f} M_sol/dt"
plt.figtext(0.5, 0.01, info_texto, ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.tight_layout()
plt.show()

print(f"Flujo medio de masa absorbida: {flujo:.4f} M_sol/unidad_tiempo ")