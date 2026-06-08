import numpy as np
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg') # Forces matplotlib to run without a display (crucial for headless cluster nodes)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
from numba import njit, prange, float32, int64

# =============================================================================
# 1. MPI CONFIGURATION (Distributed Memory)
# =============================================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # Who am I? (0 is the Boss, 1, 2, 3 are Workers)
size = comm.Get_size() # How many processes are running in total?

# =============================================================================
# 2. PHYSICAL PARAMETERS (Statically Typed for HPC)
# Using np.float32 halves memory bandwidth requirements and allows the CPU's 
# vector registers to process twice as many numbers per clock cycle.
# =============================================================================
N_SISTEMAS = 160
G = np.float32(1.0)
M_AGUJERO_NEGRO_INICIAL = np.float32(4.1e6)
M_SISTEMA = np.float32(1.0)
R_COLISION = np.float32(0.06)
R_ABSORCION = np.float32(0.5)
R_FRONTERA = np.float32(15.0)
DIST_INTERACCION = np.float32(5.0)
DT = np.float32(0.0005)
SOFTENING = np.float32(0.2)
PASOS_ESTABILIZACION = 1000
PASOS_MEDICION = 1000

OUTPUT_DIR = "resultados_mpi"

# =============================================================================
# 3. WORKLOAD DISTRIBUTION
# The Boss divides the total number of stars by the number of MPI processes.
# Each worker only calculates the forces for its specific assigned "chunk".
# =============================================================================
n_local = N_SISTEMAS // size
start_idx = rank * n_local
end_idx = (rank + 1) * n_local if rank != size - 1 else N_SISTEMAS

# =============================================================================
# 4. MATHEMATICAL KERNEL (HPC Optimized)
# - Eager Compilation: The string tells Numba exactly what data types to expect, 
#   forcing it to compile to C machine code immediately during startup.
# - prange: OpenMP-style loop parallelization using shared memory threads.
# =============================================================================
@njit('float32[:,:](float32[:,:], int64, int64, float32, float32, float32, float32, float32)', fastmath=True, parallel=True)
def compute_local_acc(pos, start, end, m_bh, m_sys, r_limit_sq, softening, G_const):
    n_total = pos.shape[0]
    n_local = end - start
    
    # We allocate the local acceleration array in float32
    local_acc = np.zeros((n_local, 2), dtype=np.float32)
    
    soft_sq = softening**2 # Computed once to save CPU cycles
    
    # prange distributes this loop across the physical cores of the current machine
    for local_i in prange(n_local):
        global_i = start + local_i
        
        rx = pos[global_i, 0]
        ry = pos[global_i, 1]
        r_sq = rx**2 + ry**2
        
        # Strict Analytical Plummer Softening: M / (r^2 + e^2)^(3/2)
        # Prevents mathematical singularities (division by zero) when objects overlap.
        factor_bh = -G_const * m_bh / ((r_sq + soft_sq)**np.float32(1.5))
        local_acc[local_i, 0] = rx * factor_bh
        local_acc[local_i, 1] = ry * factor_bh

        # N-Body Pairwise Interactions
        for j in range(n_total):
            if global_i == j:
                continue # Skip self-interaction
                
            dx = pos[j, 0] - pos[global_i, 0]
            dy = pos[j, 1] - pos[global_i, 1]
            d_sq = dx**2 + dy**2
            
            # Distance filter: only compute gravity if within interaction range
            if d_sq < r_limit_sq:
                f_mag = G_const * m_sys / ((d_sq + soft_sq)**np.float32(1.5))
                local_acc[local_i, 0] += f_mag * dx