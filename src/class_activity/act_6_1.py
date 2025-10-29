# -*- coding: utf-8 -*-
"""
@author: O Valencia
"""
from matplotlib.pyplot import *
from numpy import *

# Definición de constantes
N = 1000  # Número de pasos
x0 = 1.0  # Posición inicial en el eje x
v0 = 0.0  # Velocidad inicial
tau = 30.0  # Tiempo en segundos de la simulación
h = tau / float(N - 1)  # Paso del tiempo
k = 2.0  # Constante elástica del resorte
m = 0.5  # Masa de la partícula
b = 0.1  # Coeficiente de fricción

# Generamos un arreglo de Nx2 para almacenar posición y velocidad
y = zeros([N, 2])
# tomamos los valores del estado inicial
y[0, 0] = x0
y[0, 1] = v0

# Generamos tiempos igualmente espaciados
tiempo = linspace(0, tau, N)


# Definimos nuestra ecuación diferencial
def EDO(estado, tiempo):
    f0 = estado[1]
    f1 = -(k / m) * estado[0] - (b / m) * estado[1]
    return array([f0, f1])


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

fig, axes = subplots(3, 1, figsize=(8, 10))  # 3 rows, 1 column
axes[0].plot(tiempo, xdatos, "-g")
axes[0].set_title("Posición vs Tiempo")
axes[0].set_xlabel("Tiempo (s)")
axes[0].set_ylabel("Posición (m)")
axes[1].plot(tiempo, vdatos, "-b")
axes[1].set_title("Velocidad vs Tiempo")
axes[1].set_xlabel("Tiempo (s)")
axes[1].set_ylabel("Velocidad (m/s)")
axes[2].plot(xdatos, vdatos, "-r")
axes[2].set_title("Fase espacio")
axes[2].set_xlabel("Posición (m)")
axes[2].set_ylabel("Velocidad (m/s)")
fig.tight_layout()
show()
