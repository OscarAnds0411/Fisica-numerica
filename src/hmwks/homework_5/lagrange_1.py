"""
Programa: interpolación de lagrange
Typed: 04/11/2002
By: Oscar en un día aburrido de SS
Que vamos a hacer?
- Escribir un programa que ajuste un polinomio según el algortimo 
de Lagrange, puede utilizar las bibliotecas de Python, a un conjunto de n puntos.
"""

import os #para guardar las imagenes
import numpy as np # numpy
import numpy.polynomial.polynomial as polipocket # numpy
from pylab import *
from rich.console import Console  #Hay que darle formato a la consola
from rich.table import Table #Hay que darle formato a la consola
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#plt.style.use('seaborn-poster')

x = [0,1,2]
y = [1,3,2]

P1_c= [1,-1.5,.5]
P2_c= [0,2,-1]
P3_c= [0,-.5,.5]

#obtenemos la polipocket:
P1= polipocket.Polynomial(P1_c)
P2= polipocket.Polynomial(P2_c)
P3= polipocket.Polynomial(P3_c)

x_new = np.arange(-1.,3.1,0.1)

fig = plt.figure(figsize=(10,8))
plt.plot(x_new, P1(x_new), 'b', label ='P1')
plt.plot(x_new, P2(x_new), 'r', label ='P2')
plt.plot(x_new, P3(x_new), 'g', label ='P3')

plt.plot(x, np.ones(len(x)), 'ko', x, np.zeros(len(x)), 'ko')
plt.title('Lagrange Basis Polynomials')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()