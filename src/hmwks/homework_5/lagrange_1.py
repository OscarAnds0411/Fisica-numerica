"""
Programa: interpolación de lagrange
Typed: 04/11/2002
By: Oscar en un día aburrido de SS
Que vamos a hacer?
- Escribir un programa que ajuste un polinomio según el algortimo 
de Lagrange, puede utilizar las bibliotecas de Python, a un conjunto de n puntos.
"""

import os  # para guardar las imagenes

import matplotlib.pyplot as plt
import numpy as np  # numpy
from numpy.polynomial import Polynomial  # numpy
from pylab import *
from rich.console import Console  # Hay que darle formato a la consola
from rich.table import Table  # Hay que darle formato a la consola
from scipy.integrate import odeint


output_dir = "resultados_tarea_5"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
console = Console()  # Para que se vea bonito

def lagrange_interpolation(points):
    """"
    realiza la interpolacion dados n puntos (un array)
    """
    grad = len(points)
    poly = Polynomial(np.zeros(grad))

    for j in range(grad):
        k = [i for i in range(grad) if i!= j]
        roots = -1* points[k,0]

        sub_poly = Polynomial.fromroots(points[k,0])
        scale = points[j,0]/ np.prod(points[j,0]-points[k,0])
        sub_poly.coef *= scale
        poly.coef += sub_poly.coef
    return poly

console.rule("[bold red]Programa de interpolación para datos dados:[/bold red]")
i = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
Ei = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])  # Energía (MeV)
fE = np.array(
    [10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7]
)  # Sección eficaz (MeV)
sigma = np.array(
    [9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14]
)  # Incertidumbre (MeV)

table = Table(title="[bold yellow]Tabla para la sección eficaz[/bold yellow]")
table.add_column("i", justify="left", style="cyan")
table.add_column(r'$E_i$', justify="center", style="red")
table.add_column(r'$f\left(E_i\right)$', justify="center", style="red")
table.add_column(r'$\sigma_i$', justify="center", style="magenta")
for s in range(len(i)):
    table.add_row(f"{i[s]}",f"{Ei[s]:.1f}",f"{fE[s]:.2f}",f"{sigma[s]:.2f}")

console.print(table)
#juntamos nuestros puntos para nuestra función:
puntos = np.vstack((Ei,fE)).T

lag = lagrange_interpolation(puntos)

E_range = np.linspace(0, 200, 1000)
f_eval = lag(E_range)

plt.figure(figsize=(10,6))
plt.plot(Ei, fE, 'o', label='Datos medidos (puntos)', color='r')
plt.plot(E_range, f_eval, '-', label='Ajuste (Lagrange)', color = 'b')
plt.title('Sección eficaz interpolada por lagrange')
plt.xlabel(r'Energía $E_i$ (MeV)')
plt.ylabel(r'Sección eficaz $f\left(E_i\right)$ (MeV)')
plt.grid(True)
plt.legend()
plt.show()