"""
EJERCICIO 2: Interpolación de Datos Experimentales
===================================================

Tienes mediciones de temperatura vs tiempo durante un experimento:

t (s):  0    5    10   15   20   25   30
T (°C): 20   35   48   59   68   75   80

a) Interpolar usando polinomios de Lagrange para t = 7.5 s
b) Interpolar usando splines cúbicos para t = 7.5 s
c) Comparar ambos resultados
d) Estimar la velocidad de calentamiento dT/dt en t = 10 s
e) Graficar datos originales y curvas interpoladas
f) ¿Cuál método es más apropiado y por qué?

Conceptos clave:
• Interpolación de Lagrange
• Splines cúbicos
• Derivación numérica
• Comparación de métodos
"""

import numpy as lp
import matplotlib.pyplot as gp
import scipy as sp
import os  # para guardar las imagenes
from scipy.interpolate import CubicSpline
from numpy.polynomial import Polynomial  # numpy
from rich.console import Console  # Hay que darle formato a la consola
from rich.table import Table  # Hay que darle formato a la consola
from rich.panel import Panel  # Hay que darle formato a la consola
from scipy.optimize import brentq
from rich import box

cs = Console()

output_dir = "exam_results"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cs.rule("[bold red] ejercicio 2: Interpolacion de datos experimentales :D")


def lagrange_interpolation(x_data, y_data):
    """
    Realiza la interpolación de Lagrange dados n puntos.

    Parámetros:
    -----------
    x_data : array-like
        Coordenadas x de los puntos
    y_data : array-like
        Coordenadas y de los puntos

    Retorna:
    --------
    poly : Polynomial
        Polinomio interpolador de Lagrange
    """
    n = len(x_data)  # Vemos la longitud del arreglo horizontal
    x_data = lp.array(x_data, dtype=float)
    y_data = lp.array(y_data, dtype=float)

    poly_total = Polynomial([0.0])  # inicializamos el polinomio en cero

    # Construimos el polinomio de lagrange
    for j in range(n):
        index = [
            i for i in range(n) if i != j
        ]  # indices de todos los puntos menos el j.ésimo
        # Construimos el polinomio base L_j(x)
        # Numerador Prod (x-x_i) para i neq j
        numerador = Polynomial.fromroots(x_data[index])

        denominador = lp.prod(x_data[j] - x_data[index])
        L_j = numerador / denominador
        poly_total = poly_total + y_data[j] * L_j
    return poly_total


cs.rule("[bold red] imprimimos los datos dados")
T = lp.array([20, 35, 48, 59, 68, 75, 80])
t = lp.array([0, 5, 10, 15, 20, 25, 30])

tab = Table(title="[bold yellow]Datos experimentales")
tab.add_column("T [C]", justify="center", style="blue")
tab.add_column("t [C]", justify="center", style="blue")

for n in range(len(T)):
    tab.add_row(f"{T[n]}", f"{t[n]}")
cs.print(tab)
cs.print(
    "[cyan] Si interpolamos usando los polinomios de lagrange para t=7.5s tenemos lo siguiente:\n"
)

poli = lagrange_interpolation(t, T)
cs.print("[bold green] polinomio generado correctamente")

t_range = lp.linspace(0, 30, 1000)
f_lagrange = poli(t_range)
esp = poli(7.5)

gp.figure(figsize=(12, 7))

gp.scatter(t, T, color="r", label="Datos  experimentales")
gp.scatter(7.5, esp, color="b", label=rf"Punto de ${poli}$ en t=7.5 s")
gp.plot(
    t_range,
    f_lagrange,
    "-",
    color="g",
    linewidth=2.5,
    label="Polinomio de Lagrange",
    alpha=0.8,
)
gp.xlabel(r"tiempo $t_i$ (s)", fontsize=13, fontweight="bold")
gp.ylabel(r"Temperatura $f(t_i)$ (Celcius)", fontsize=13, fontweight="bold")
gp.title(
    "Interpolación de Lagrange: hasta los 7.5 s",
    fontsize=15,
    fontweight="bold",
)
gp.legend(fontsize=11, loc="best")
gp.grid(True, alpha=0.3)
gp.tight_layout()

new_png = os.path.join(output_dir, "interpolacion_lagrange_t_T.png")
gp.savefig(new_png, dpi=300, bbox_inches="tight")
cs.print(f"[bold yellow] Gráfica guardada en: {new_png}")
gp.show()

# _____
cs.print("[bold green] O de otra manera")
t_range = lp.linspace(0, 7.5, 1000)
f_lagrange = poli(t_range)

T_sop = lp.array([20, 35])
t_sop = lp.array([0, 5])

gp.figure(figsize=(12, 7))

gp.scatter(t_sop, T_sop, color="r", label="Datos  experimentales")
gp.plot(
    t_range,
    f_lagrange,
    "-",
    color="g",
    linewidth=2.5,
    label="Polinomio de Lagrange hasta 7.5 s",
    alpha=0.8,
)
gp.xlabel(r"tiempo $t_i$ (s)", fontsize=13, fontweight="bold")
gp.ylabel(r"Temperatura $f(t_i)$ (Celcius)", fontsize=13, fontweight="bold")
gp.title(
    "Interpolación de Lagrange: hasta los 7.5 s",
    fontsize=15,
    fontweight="bold",
)
gp.legend(fontsize=11, loc="best")
gp.grid(True, alpha=0.3)
gp.tight_layout()

new_png = os.path.join(output_dir, "interpolacion_lagrange_t_T_75.png")
gp.savefig(new_png, dpi=300, bbox_inches="tight")
cs.print(f"[bold yellow] Gráfica guardada en: {new_png}")
gp.show()

cs.rule("[bold red] Hagamos lo mismo pero con spline cubic")

# funcion:


def cubic(x_data, y_data):
    return CubicSpline(x_data, y_data, bc_type="natural")


cs.print("[bold cyan] Ajustando por splines cubicos...")

spline = cubic(t, T)

cs.print("[bold cyan] spline cubico creado exitosamente")

t_range = lp.linspace(0, 30, 1000)
f_spline = spline(t_range)
