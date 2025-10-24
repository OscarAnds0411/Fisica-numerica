"""
Lanzamiento de martillo. El record mundial para hombres en lanzamiento de martillo es de 86.74 m
por Yuri Sedykh y se ha mantenido desde 1986. El martillo pesa 7:26 kg, es esférico,
y tiene un radio de R = 6 cm. 

La fricciÛn en el martillo puede ser considerada proporcional
al cuadrado de la velocidad del martillo relativa al aire:
F_D =\frac{1}{2}\rho AC_Dv^2

donde \rho es la densidad del aire (1.2 kg/m^3 ) y A = \pi R^2 es la sección transver-
sal del martillo. 

El martillo puede experimentar, en principio, un flujo laminar con coeficiente de rozamiento CD = 0.5 
o un flujo inestable oscilante con CD = 0.75.

(a) Resuelva la ecuación de movimiento para el lanzamiento oblicuo
de martillo. Deberá transformar las EDOs para los moviemtos
en x y y en un sistema de cuatro ecuaciones de primer orden.
Considere lanzamientos desde una posición inicial x0 = 0 y y0 = 2
m, para un ángulo ideal \theta = 45 y encuentre la velocidad que
produce la distancia del lanzamiento del record mundial.

(b) Calcule y grafique la dependencia en el tiempo de la altitud del
martillo y su trayectoria y = y (x) en los tres régimenes:
i. Sin fricción
ii. Flujo laminar
iii. Flujo inestable oscilante

(c) En el inciso anterior, estime la cantidad en que es influenciada la
distancia del lanzamiento por la fricción.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 22:54:51 2021

@author: fernando
"""

from pylab import *
import numpy as np
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

# Definición de constantes
MAX = 5000  # -Número máximo de iteraciones
N = 500  # -Número de pasos
epsilon = 1.0e-3 # -Precisión requerida
x0=0.0    # -Posición inicial en x
y0=2.0    # -Pocisión inicial en y
vx0=10.0      # -Velocidad inicial en x
vy0=10.0      # -Velocidad 
rho=1.2      # -Densidad del aire
dt = 0.001 # -Longitud del pado en el tiempo
R=0.06          # -Radio del martillo
A=pi*R**2        # -Area de sección transversal
m=7.26        # -Masa en kg
g=9.81       # -Aceleración gravitacional en m/s**2
Cd=list([0., 0.5, 0.75])
record = 86.74

# -Definimos unestro sistema de EDO.
def launching(r, time, k):
    f0 = r[1]
    f1 = -k/m*r[1]*np.sqrt(r[1]**2 + r[3]**2)
    f2 = r[3]
    f3 = -g - k/m*r[3]*np.sqrt(r[1]**2 + r[3]**2)
    return array([f0, f1, f2, f3])


# -Dadas las condiciones iniciales de lanzamiento encontramos la 
#  distancia recorrida antes de llegar al suelo.
def FindGroundDistance(InitialPosition, v0, k):
    # -Velocidad en cada componente. En este caso, como theta = 45°, 
    #  vx = vy
    v = v0*sin(pi/4.)
    finalt = 5.        # -Tiempo final de la simulación.
    r0 = array([InitialPosition[0], v, InitialPosition[1], v])
    r = r0
    t = linspace(0., finalt, N)
    # -Tiempo donde el martillo "atravieza" el suelo
    tground = 0.
    
    i = 0
    while i < MAX:
        sol = odeint(launching, r, t, args = (k,))
        n = len(sol)-1
        
        # -En caso de no llegar al suelo.
        if sol[n, 2] > 0:
            finalt += 1.
            t = linspace(0., finalt, N)
        else:
            for j in range(n):
            
                # -'Atravezamos' el suelo
                if sol[j,2] <= 0:
                    tground += t[j-1]-t[0]
                    
                    # -Checamos que no estemos muy lejos del cero.
                    if abs(sol[j,2]) <= epsilon/2.:
                        
                        # -En caso de cumplir con la precisión 
                        #  reacemos nuestra simulación pero usando el 
                        #  tiempo 'casi exacto' para que toque el suelo.
                        t = linspace(0.,tground+t[j]-t[j-1], N*50**i)
                        sol = odeint(launching, r0, t, args = (k,))
                        n = len(sol)-1
                        
                        # -Regresamos el promedio de 'justo antes de tocar
                        #  el suelo' y 'justo despues de tocar el suelo'.
                        return (sol[n,0]+sol[n-1,0])/2., tground
                    else:
                        # -En caso de no cumplir con la presición, 
                        #  resolvemos nuestra ED pero nuestras condiciones 
                        #  iniciales serán las condiciones que nuestro 
                        #  martillo tenia justo antes de tocar el suelo.
                        r = array([sol[j-1][0], sol[j-1][1],\
                                    sol[j-1][2], sol[j-1][3]])
                        t = linspace(t[j-1], t[j], 50)
                        break            
        i += 1
        
    print("Distancia recorrida antes de llegar al suelo no encontrada\
          dentro de las iteraciones permitidas. Regresamos 0.")
    return 0.


# -Encontramos la velocidad necesaria para alcanzar la distancia requerida.
# -'distance' es la distancia requerida.
# -'v0' es la semilla inicial
def FindVelocity(distance, InitialPosition, v0, k):
    f = lambda v: FindGroundDistance(InitialPosition, v, k)[0] - distance
    dv = 1.0e-3
    
    i = 0
    #Newton-Raphson
    while i < MAX:
        #print(i)
        fv = f(v0)
        if abs(fv) <= epsilon:
            return v0
        df = (f(v0+dv/2.)-f(v0-dv/2.))/dv
        dv = -fv/df
        v0 += dv
            
        i += 1
    print("Velocidad no encontrada\
          dentro de las iteraciones permitidas. Regresamos 0.")
    return 0.


# -Encontramos velocidad de campeonato sin considerar fricción.
k = rho*A*Cd[0]/2.
v = FindVelocity(record, array([x0,y0]), 28., k)
print("La velocidad necesaria para alcanzar la distancia record sin\
 considerar la friccion es:",v)

# -Aprobechamos que theta = 45°
vy = v*sin(pi/4.)
r0 = array([x0, vy, y0, vy])
ground = zeros(N, float)

# -Calculamos
solution = list()

# -Hacemos las tres gráficas
for i in range(3):
    k = rho*A*Cd[i]/2.
    t = FindGroundDistance(array([x0,y0]), v,k)[1]
    time = linspace(0., t, N)
    solution.append(odeint(launching, r0, time, args = (k,)))

    xdata = solution[i][:,0]
    ydata = solution[i][:,2]
    print('Alcance con C_D = ', Cd[i], ' es ', xdata[N-1])
    
    # -Graficamos altura contra distancia
    figure(1)
    xlabel("Distancia"); ylabel("Altura"); title("Trayectoria del martillo")
    lb = 'C_D = ' + str(Cd[i])
    plot(xdata, ydata, '-', label = lb, lw=1.5,color=((i-1.)*(i-2.)/2.,\
                                       (i-2.)*i*(-1)*0.8, (i-1.)*i/2.))
    plot(xdata, ground,'-',lw=1,color=(0.2,0.2,0.2))
    grid(True)
    legend()
    show()
    
    # -Graficamos altura contra tiempo
    figure(2)
    xlabel("Tiempo"); ylabel("Altura"); title("Altura vs Tiempo")
    plot(time, ydata,'-', label = lb,lw=1.5,color=((i-1.)*(i-2.)/2.,\
                                       (i-2.)*i*(-1)*0.8, (i-1.)*i/2.))
    plot(time, ground,'-',lw=1,color=(0.2,0.2,0.2))
    grid(True)
    legend()
    show()
    