"""
Simulación de Cuerda Vibrante - Versión Optimizada
==================================================

Usa curve.modify() para actualizar puntos de forma eficiente.
"""

from vpython import *
import numpy as np

# ==============================================================================
# PARÁMETROS FÍSICOS
# ==============================================================================

rho = 0.01  # Densidad lineal (kg/m)
ten = 40.0  # Tensión (N)
c = np.sqrt(ten / rho)  # Velocidad de onda
c1 = c
ratio = (c / c1) ** 2

print(f"Velocidad de onda: c = {c:.2f} m/s")
print(f"Ratio: {ratio:.4f}")

# ==============================================================================
# INICIALIZACIÓN
# ==============================================================================

N = 101  # Número de puntos

# Matriz de posiciones
xi = np.zeros((N, 3), float)

# Condición inicial: forma triangular
for i in range(0, 81):
    xi[i, 0] = 0.00125 * i

for i in range(81, N):
    xi[i, 0] = 0.1 - 0.005 * (i - 80)

# ==============================================================================
# CONFIGURACIÓN VISUAL
# ==============================================================================

# Crear puntos iniciales
puntos = []
for i in range(N):
    x = 2.0 * i - 100.0
    y = 300.0 * xi[i, 0]
    puntos.append(vector(x, y, 0))

# Crear la curva
vibst = curve(puntos, color=color.yellow, radius=0.3)

# Esferas en los extremos
ball1 = sphere(pos=vector(-100, 0, 0), color=color.red, radius=2)
ball2 = sphere(pos=vector(100, 0, 0), color=color.red, radius=2)

# Etiquetas
label(
    pos=vector(0, 40, 0),
    text="Simulación de Cuerda Vibrante",
    height=10,
    color=color.cyan,
)

# ==============================================================================
# PRIMER PASO
# ==============================================================================

for i in range(1, N - 1):
    xi[i, 1] = xi[i, 0] + 0.5 * ratio * (xi[i + 1, 0] + xi[i - 1, 0] - 2 * xi[i, 0])

xi[0, 1] = 0
xi[N - 1, 1] = 0

# ==============================================================================
# BUCLE PRINCIPAL
# ==============================================================================

print("\nSimulación iniciada.")

while True:
    rate(60)

    # Calcular siguiente paso
    for i in range(1, N - 1):
        xi[i, 2] = (
            2 * xi[i, 1]
            - xi[i, 0]
            + ratio * (xi[i + 1, 1] + xi[i - 1, 1] - 2 * xi[i, 1])
        )

    xi[0, 2] = 0
    xi[N - 1, 2] = 0

    #  MÉTODO ALTERNATIVO: Usar modify() para actualizar cada punto
    for i in range(N):
        x = 2.0 * i - 100.0
        y = 300.0 * xi[i, 2]
        vibst.modify(i, pos=vector(x, y, 0))

    # Rotar tiempos
    xi[:, 0] = xi[:, 1]
    xi[:, 1] = xi[:, 2]
