from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np

# ==============================================================================
# ANIMACIÓN DEL SISTEMA SOLAR A PARTIR DE UN FICHERO DE DATOS
# ==============================================================================

# Parámetros
# ==============================================================================
file_in = "trayectorias_sistema_solar.dat"   # Debe coincidir con el nombre del generador
file_out = "planetas"          

# Límites de los ejes (Neptuno está a ~30 UA, Voyager irá más allá)
x_min, x_max = -40, 40
y_min, y_max = -40, 40

interval = 20        
show_trail = True    
trail_width = 0.45    
trail_length = 50    
save_to_file = False 
dpi = 150

# 1. Ajuste de Radios (10 elementos: Sol + 8 planetas + Voyager)
planet_radius = [0.25, 0.05, 0.07, 0.07, 0.1, 0.18, 0.12, 0.12, 0.12, 0.05]

# 2. Ajuste de Colores (Corregida la coma faltante en "royalblue")
planet_colors = [
    "yellow",      # Sol
    "grey",        # Mercurio
    "orange",      # Venus
    "deepskyblue", # Tierra
    "red",         # Marte
    "saddlebrown", # Júpiter
    "tan",         # Saturno
    "cyan",        # Urano
    "royalblue",   # Neptuno
    "white"        # Voyager
]

# 3. Ajuste de Nombres
planet_names = [
    "Sol", "Mercurio", "Venus", "Tierra",
    "Marte", "Jupiter", "Saturno", "Urano", "Neptuno", "Voyager"
]

# ==============================================================================
# LECTURA DEL FICHERO DE DATOS
# ==============================================================================
with open(file_in, "r") as f:
    data_str = f.read()

frames_data = []

# Separar por doble salto de línea (frames)
for frame_data_str in data_str.strip().split("\n\n"):
    frame_data = []
    # Separar por salto de línea (planetas)
    for planet_pos_str in frame_data_str.split("\n"):
        # IMPORTANTE: sep=" " porque el generador guarda con espacios, no con comas
        planet_pos = np.fromstring(planet_pos_str, sep=" ")
        if planet_pos.size > 0:
            frame_data.append(planet_pos)
    if len(frame_data) > 0:
        frames_data.append(frame_data)

nplanets = len(frames_data[0])

# ==============================================================================
# CREACIÓN DE LA FIGURA
# ==============================================================================
fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_facecolor("black")

# Validaciones de seguridad
if nplanets != len(planet_colors) or nplanets != len(planet_names):
    raise ValueError(f"Error: El archivo tiene {nplanets} cuerpos, pero las listas de colores/nombres no coinciden.")

planet_points = []
planet_trails = []

for planet_pos, radius, color, name in zip(frames_data[0], planet_radius, planet_colors, planet_names):
    x, y = planet_pos
    # Punto del planeta/sonda
    planet_point = Circle((x, y), radius, color=color, label=name)
    ax.add_artist(planet_point)
    planet_points.append(planet_point)

    # Estela
    if show_trail:
        planet_trail, = ax.plot([x], [y], "-", linewidth=trail_width, color=color, alpha=0.6)
        planet_trails.append(planet_trail)

# ==============================================================================
# FUNCIONES DE ANIMACIÓN
# ==============================================================================
def update(j_frame, frames_data, planet_points, planet_trails, show_trail):
    artists = []
    for j_planet, planet_pos in enumerate(frames_data[j_frame]):
        x, y = planet_pos
        planet_points[j_planet].center = (x, y)
        artists.append(planet_points[j_planet])

        if show_trail:
            xs_old, ys_old = planet_trails[j_planet].get_data()
            xs_new = np.append(xs_old, x)
            ys_new = np.append(ys_old, y)

            if len(xs_new) > trail_length:
                xs_new = xs_new[-trail_length:]
                ys_new = ys_new[-trail_length:]

            planet_trails[j_planet].set_data(xs_new, ys_new)
            artists.append(planet_trails[j_planet])
    return artists

def init_anim():
    return planet_points + planet_trails

# Leyenda
ax.legend(loc="upper right", fontsize="x-small", facecolor="white", framealpha=0.5)

# Generar animación
nframes = len(frames_data)
animation = FuncAnimation(
    fig, update, frames=nframes, init_func=init_anim,
    fargs=(frames_data, planet_points, planet_trails, show_trail),
    blit=True, interval=interval
)

plt.show()