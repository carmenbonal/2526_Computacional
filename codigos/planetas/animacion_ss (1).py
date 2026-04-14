from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np

# ==============================================================================
# ANIMACIÓN DEL SISTEMA SOLAR A PARTIR DE UN FICHERO DE DATOS
# ==============================================================================

# Parámetros
# ==============================================================================
file_in = "planets_data.dat"   # fichero generado por simulador_ss.py
file_out = "planetas"          # nombre del fichero de salida (sin extensión)

# Límites de los ejes
x_min = -35
x_max = 35
y_min = -35
y_max = 35

interval = 20        # tiempo entre fotogramas en ms
show_trail = True    # mostrar estela
trail_width = 0.45    # estela más fina
trail_length = 40    # número de posiciones que conserva la estela
save_to_file = False # True -> guarda mp4/pdf, False -> muestra por pantalla
dpi = 150

# Radio visual de los planetas
# Puede ser un número o una lista con un radio por planeta
planet_radius = [0.25, 0.05, 0.07, 0.07, 0.1, 0.18, 0.12, 0.12, 0.12]

# Colores de los planetas
planet_colors = [
    "yellow",      # Sol
    "grey",        # Mercurio
    "orange",      # Venus
    "deepskyblue", # Tierra
    "red",         # Marte
    "saddlebrown", # Júpiter
    "tan",         # Saturno
    "cyan",        # Urano
    "royalblue"    # Neptuno
]

# Nombres para la leyenda
planet_names = [
    "Sol", "Mercurio", "Venus", "Tierra",
    "Marte", "Jupiter", "Saturno", "Urano", "Neptuno"
]

# ==============================================================================
# LECTURA DEL FICHERO DE DATOS
# ==============================================================================
with open(file_in, "r") as f:
    data_str = f.read()

frames_data = []

for frame_data_str in data_str.split("\n\n"):
    frame_data = []

    for planet_pos_str in frame_data_str.split("\n"):
        planet_pos = np.fromstring(planet_pos_str, sep=",")

        if planet_pos.size > 0:
            frame_data.append(planet_pos)

    if len(frame_data) > 0:
        frames_data.append(frame_data)

nplanets = len(frames_data[0])

# ==============================================================================
# CREACIÓN DE LA FIGURA
# ==============================================================================
fig, ax = plt.subplots()
ax.axis("equal")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_facecolor("black")

# Si se da un único radio, lo convierte en array
if not hasattr(planet_radius, "__iter__"):
    planet_radius = planet_radius * np.ones(nplanets)
else:
    if nplanets != len(planet_radius):
        raise ValueError(
            "El número de radios especificados no coincide con el número de planetas"
        )

# Comprobar número de colores
if nplanets != len(planet_colors):
    raise ValueError(
        "El número de colores especificados no coincide con el número de planetas"
    )

# Comprobar número de nombres
if nplanets != len(planet_names):
    raise ValueError(
        "El número de nombres especificados no coincide con el número de planetas"
    )

# Crear puntos y estelas
planet_points = []
planet_trails = []

for planet_pos, radius, color, name in zip(frames_data[0], planet_radius, planet_colors, planet_names):
    x, y = planet_pos

    # Punto del planeta
    planet_point = Circle((x, y), radius, color=color, label=name)
    ax.add_artist(planet_point)
    planet_points.append(planet_point)

    # Estela del mismo color
    if show_trail:
        planet_trail, = ax.plot(
            [x], [y], "-",
            linewidth=trail_width,
            color=color
        )
        planet_trails.append(planet_trail)

# ==============================================================================
# FUNCIONES DE ANIMACIÓN
# ==============================================================================
def update(j_frame, frames_data, planet_points, planet_trails, show_trail):
    for j_planet, planet_pos in enumerate(frames_data[j_frame]):
        x, y = planet_pos
        planet_points[j_planet].center = (x, y)

        if show_trail:
            xs_old, ys_old = planet_trails[j_planet].get_data()

            xs_new = np.append(xs_old, x)
            ys_new = np.append(ys_old, y)

            # Conservar solo los últimos trail_length puntos
            if len(xs_new) > trail_length:
                xs_new = xs_new[-trail_length:]
                ys_new = ys_new[-trail_length:]

            planet_trails[j_planet].set_data(xs_new, ys_new)

    return planet_points + planet_trails

def init_anim():
    if show_trail:
        for j_planet in range(nplanets):
            x0, y0 = frames_data[0][j_planet]
            planet_trails[j_planet].set_data([x0], [y0])

    return planet_points + planet_trails

# ==============================================================================
# LEYENDA
# ==============================================================================
ax.legend(
    handles=planet_points,
    loc="upper right",
    fontsize="small",
    facecolor="white"
)

# ==============================================================================
# GENERAR ANIMACIÓN O IMAGEN
# ==============================================================================
nframes = len(frames_data)

if nframes > 1:
    animation = FuncAnimation(
        fig,
        update,
        init_func=init_anim,
        fargs=(frames_data, planet_points, planet_trails, show_trail),
        frames=nframes,
        blit=True,
        interval=interval
    )

    if save_to_file:
        animation.save(f"{file_out}.mp4", dpi=dpi)
    else:
        plt.show()

else:
    if save_to_file:
        fig.savefig(f"{file_out}.pdf")
    else:
        plt.show()