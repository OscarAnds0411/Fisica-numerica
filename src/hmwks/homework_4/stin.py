#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulación 1D de una cuerda mediante diferencias finitas (esquema explícito)
Se agrega animación del movimiento de la cuerda en función del tiempo.

Autor: [Tu nombre]
Fecha: Octubre 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------
# Parámetros de la malla
# ----------------------------
Nx = 100  # número de puntos espaciales
Delta_x = 0.01  # paso espacial
Nt = 250  # número de pasos en tiempo
Delta_t = 0.01  # paso temporal

# velocidad característica y número de Courant
c_phys = 0.5 * (Delta_x / Delta_t)
c0 = Delta_x / Delta_t
courant = c_phys / c0
cons = (courant) ** 2

print(
    f"Delta_x={Delta_x}, Delta_t={Delta_t}, c0={c0:.3f}, c={c_phys:.3f}, Courant={courant:.3f}"
)

# ----------------------------
# Inicializar arreglo Yc[i, j] --> posición x tiempo
# ----------------------------
Yc = np.zeros((Nx, Nt), dtype=float)

# Condición inicial en t=0
for i in range(1, Nx - 1):
    x = i * Delta_x
    Yc[i, 0] = np.sin(2.0 * np.pi * x)  # forma inicial (puedes cambiarla)

# Paso inicial
for i in range(1, Nx - 1):
    Yc[i, 1] = Yc[i, 0] + 0.5 * cons * (Yc[i + 1, 0] + Yc[i - 1, 0] - 2.0 * Yc[i, 0])

# Evolución temporal
for n in range(2, Nt):
    for i in range(1, Nx - 1):
        Yc[i, n] = (
            2.0 * Yc[i, n - 1]
            - Yc[i, n - 2]
            + cons * (Yc[i + 1, n - 1] + Yc[i - 1, n - 1] - 2.0 * Yc[i, n - 1])
        )

# ----------------------------
# Animación 2D
# ----------------------------
x = np.linspace(0, (Nx - 1) * Delta_x, Nx)

fig, ax = plt.subplots(figsize=(8, 4))
(line,) = ax.plot([], [], lw=2, color="tab:blue")
ax.set_xlim(0, (Nx - 1) * Delta_x)
ax.set_ylim(-1.2 * np.max(np.abs(Yc)), 1.2 * np.max(np.abs(Yc)))
ax.set_xlabel("Posición (m)")
ax.set_ylabel("Desplazamiento y(x,t)")
ax.set_title("Animación de la cuerda vibrante")


# Función de inicialización
def init():
    line.set_data([], [])
    return (line,)


# Función que actualiza cada frame
def update(frame):
    line.set_data(x, Yc[:, frame])
    ax.set_title(f"Tiempo = {frame * Delta_t:.2f} s")
    return (line,)


# Crear animación
anim = FuncAnimation(
    fig, update, frames=Nt, init_func=init, blit=True, interval=30, repeat=True
)

plt.tight_layout()
plt.show()

# Si quieres guardar la animación como video:
# anim.save("cuerda_animacion.mp4", fps=30)
