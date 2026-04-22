import numpy as np

# --- Parámetros de la simulación ---
N = 32              # Tamaño del retículo (N x N) 
T = 2.27            # Temperatura (T_c aprox 2.269) 
pasos_mc = 1000     # Pasos Monte Carlo totales 
frecuencia_guardado = 10 # Guardar configuración cada X pasos
fichero_salida = "ising_data.dat"

def inicializar_sistema(n, ordenada=False):
    """Crea una configuración inicial de espines"""
    if ordenada:
        return np.ones((n, n), dtype=int)
    else:
        # Genera 1 o -1 con probabilidad 1/2 
        return np.random.choice([-1, 1], size=(n, n))

def calcular_delta_e(grid, i, j, n):
    """Calcula el cambio de energía si se girara el espín en (i, j)"""
    s = grid[i, j]
    # Condiciones de contorno periódicas usando el operador módulo % 
    vecinos = (grid[(i + 1) % n, j] + grid[(i - 1) % n, j] +
               grid[i, (j + 1) % n] + grid[i, (j - 1) % n])
    return 2 * s * vecinos

def paso_metropolis(grid, n, temp):
    """Realiza un paso Monte Carlo (N^2 intentos de cambio)."""
    for _ in range(n**2):
        # 1. Elegir un punto al azar 
        i, j = np.random.randint(0, n, size=2)
        
        # 2. Calcular diferencia de energía 
        de = calcular_delta_e(grid, i, j, n)
        
        # 3. Probabilidad de aceptación 
        if de <= 0:
            grid[i, j] *= -1
        elif np.random.rand() < np.exp(-de / temp):
            grid[i, j] *= -1
    return grid

def guardar_configuracion(f, grid):
    """Escribe la matriz en el formato que lee animacion_ising.py."""
    np.savetxt(f, grid, fmt='%d', delimiter=',')
    f.write("\n") # El script de animación requiere doble salto de línea entre bloques

# --- Ejecución principal ---
def simular():
    grid = inicializar_sistema(N, ordenada=False)
    
    with open(fichero_salida, "w") as f:
        for paso in range(pasos_mc):
            grid = paso_metropolis(grid, N, T)
            
            if paso % frecuencia_guardado == 0:
                guardar_configuracion(f, grid)
                print(f"Progreso: {paso}/{pasos_mc} pasos completados.", end="\r")

if __name__ == "__main__":
    simular()
    print("\nSimulación finalizada. Datos guardados en:", fichero_salida)