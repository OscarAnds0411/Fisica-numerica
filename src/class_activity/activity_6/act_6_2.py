# -*- coding: utf-8 -*-
"""
@author: O Valencia
"""
import matplotlib.pyplot as plt
import numpy as np

# Definición de constantes para el oscilador armónico amortiguado y forzado
# Caso A: dominado por fricción
N = 4000  # Número de pasos
x0 = 1.0  # Posición inicial en el eje x
v0 = 0.0  # Velocidad inicial
tau = 250.0  # Tiempo en segundos de la simulación
h = tau / float(N - 1)  # Paso del tiempo
k = 2.0  # Constante elástica del resorte
m = 0.5  # Masa de la partícula
b = 0.05  # Coeficiente de fricción
f_0 = 0.2  # Amplitud de la fuerza externa
omega = 0.5  # Frecuencia de la fuerza externa

# Generamos un arreglo de Nx2 para almacenar posición y velocidad
y = np.zeros([N, 2])
# tomamos los valores del estado inicial
y[0, 0] = x0
y[0, 1] = v0

# Generamos tiempos igualmente espaciados
tiempo = np.linspace(0, tau, N)


# Definimos nuestra ecuación diferencial
def EDO(estado, tiempo):
    f0 = estado[1]
    f1 = -(k / m) * estado[0] - (b / m) * estado[1] + (f_0 / m) * np.sin(omega * tiempo)
    return np.array([f0, f1])


# Método de Euler para  resolver numéricamente la EDO
def Euler(y, t, h, f):
    y_p = y + h * f(y, t)  # Calculamos el valor siguiente de y predictor
    y_c = (
        y + h * (f(y, t) + f(y_p, t + h)) / 2.0
    )  # Calculamos el valor siguiente de y corregido
    return y_c


# Ahora calculamos!
for j in range(N - 1):
    y[j + 1] = Euler(y[j], tiempo[j], h, EDO)

# Ahora graficamos
xdatos = [y[j, 0] for j in range(N)]
vdatos = [y[j, 1] for j in range(N)]

# Create a figure and two subplots arranged in 1 row and 2 columns
fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 1 column
axes = axes.flatten()
# caso A: dominado por fricción
axes[0].plot(xdatos, vdatos, "-b")
axes[0].set_title("Fase espacio caso A")
axes[0].set_xlabel("Posición (m)")
axes[0].set_ylabel("Velocidad (m/s)")
# ----------------------------------------------------------
axes[1].plot(tiempo, vdatos, "-r")
axes[1].set_title("Velocidad vs Tiempo caso A")
axes[1].set_xlabel("Tiempo (s)")
axes[1].set_ylabel("Velocidad (m/s)")
# ----------------------------------------------------------
axes[2].plot(tiempo, xdatos, "-g")
axes[2].set_title("Posición vs Tiempo caso A")
axes[2].set_xlabel("Tiempo (s)")
axes[2].set_ylabel("Posición (m)")

# caso B: dominado por crecimiento de amplitud
# Cambiamos la unica constante de
# interes del oscilador armónico amortiguado y forzado
omega = 2.0  # Frecuencia de la fuerza externa

# Ahora calculamos!
for j in range(N - 1):
    y[j + 1] = Euler(y[j], tiempo[j], h, EDO)

# Ahora graficamos
xdatos = [y[j, 0] for j in range(N)]
vdatos = [y[j, 1] for j in range(N)]
# Caso B: dominado por crecimiento de amplitud
axes[3].plot(xdatos, vdatos, "-b")
axes[3].set_title("Fase espacio caso B")
axes[3].set_xlabel("Posición (m)")
axes[3].set_ylabel("Velocidad (m/s)")
# ----------------------------------------------------------
axes[4].plot(tiempo, vdatos, "-r")
axes[4].set_title("Velocidad vs Tiempo caso B")
axes[4].set_xlabel("Tiempo (s)")
axes[4].set_ylabel("Velocidad (m/s)")
# ----------------------------------------------------------
axes[5].plot(tiempo, xdatos, "-g")
axes[5].set_title("Posición vs Tiempo caso B")
axes[5].set_xlabel("Tiempo (s)")
axes[5].set_ylabel("Posición (m)")


plt.tight_layout()
plt.show()
