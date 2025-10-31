# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:59:19 2021

@author: Prof. Uriarte
"""
import matplotlib.pyplot as plt
import numpy as np

# Definición de constantes
N = 1000  # Número de pasos
x0 = 6.0  # Posición inicial
v0 = 2.0  # Velocidad inicial
tau = 6.0  # Tiempo en segundos de la simulación
h = tau / float(N - 1)  # Paso del tiempo
k = 2.5  # Constante elástica del resorte
m = 0.2  # Masa de la partícula

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
    f1 = -(k / m) * estado[0]
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

# plot(tiempo,xdatos,'-r')
# plot(tiempo,vdatos,'-b')
plt.plot(xdatos, vdatos, "-r")
plt.xlabel("Tiempo")
plt.ylabel("Posición y velocidad")
plt.show()
