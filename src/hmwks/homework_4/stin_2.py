#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulación 1D de una cuerda mediante diferencias finitas (esquema explícito)
Se construye una superficie espacio-tiempo y se muestra en 3D.

Autor: [Tu nombre]
Fecha: Octubre 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parámetros de la malla
# ----------------------------
Nx = 100  # número de puntos espaciales
Delta_x = 0.01  # paso espacial
Nt = 250  # número de pasos en tiempo
Delta_t = 0.01  # paso temporal

# velocidad característica y número de Courant
c_phys = 0.5 * (Delta_x / Delta_t)  # ejemplo; igual que en tu código original
c0 = Delta_x / Delta_t  # velocidad de la malla
courant = c_phys / c0
cons = (courant) ** 2  # coeficiente en la ecuación en diferencias

print(
    f"Delta_x={Delta_x}, Delta_t={Delta_t}, c0={c0:.3f}, c={c_phys:.3f}, Courant={courant:.3f}"
)

# ----------------------------
# Inicializar arreglo (espacio x tiempo)
# Yc[i, j] --> i: posición (0..Nx-1), j: tiempo (0..Nt-1)
# ----------------------------
Yc = np.zeros((Nx, Nt), dtype=float)

# Condición inicial en t=0 (ejemplo: seno)
for i in range(1, Nx - 1):
    x = i * Delta_x
    Yc[i, 0] = np.sin(2.0 * np.pi * x)  # forma inicial

# Condición para t=Delta_t (usamos un paso inicial centrado)
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
# Seleccionar submalla para graficar (reducir densidad para wireframe)
# ----------------------------
x_idx = np.arange(0, Nx, 2)  # índices espaciales (cada 2)
t_idx = np.arange(0, Nt, 10)  # índices temporales (cada 10)

# Crear mallas de índices
X_idx, T_idx = np.meshgrid(x_idx, t_idx)  # shape: (len(t_idx), len(x_idx))

# Convertir índices a coordenadas físicas para etiquetado
X_pos = X_idx * Delta_x
T_time = T_idx * Delta_t

# Extraer Z desde Yc usando índices (X_idx: posición, T_idx: tiempo)
Z = Yc[X_idx, T_idx]  # recuerda Yc[space_index, time_index]

# ----------------------------
# Graficar wireframe 3D
# ----------------------------
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

# superficie de malla
ax.plot_wireframe(X_pos, T_time, Z, rcount=50, ccount=50, linewidth=0.8)

ax.set_xlabel("Posición (m)")
ax.set_ylabel("Tiempo (s)")
ax.set_zlabel("Desplazamiento y(x,t)")
ax.set_title("Evolución espacio-tiempo de la cuerda (wireframe)")

plt.tight_layout()
plt.show()
