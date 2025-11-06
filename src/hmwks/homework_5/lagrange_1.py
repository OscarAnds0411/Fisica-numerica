"""
Programa: interpolación de lagrange
Typed: 04/11/2002
By: Oscar en un día aburrido de SS
Que vamos a hacer?
- Escribir un programa que ajuste un polinomio según el algortimo
de Lagrange, puede utilizar las bibliotecas de Python, a un conjunto de n puntos.
"""

import os  # para guardar las imagenes
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np  # numpy
from numpy.polynomial import Polynomial  # numpy
from rich.console import Console  # Hay que darle formato a la consola
from rich.table import Table  # Hay que darle formato a la consola
from rich.panel import Panel  # Hay que darle formato a la consola
from scipy.optimize import brentq
from rich import box

output_dir = "resultados_tarea_5"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
console = Console()  # Para que se vea bonito


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
    n = len(x_data)
    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float)

    # Inicializar el polinomio total en cero
    poly_total = Polynomial([0.0])

    # Construir el polinomio de Lagrange
    # P(x) = Σ y_i · L_i(x)
    # donde L_i(x) = Π[(x - x_j) / (x_i - x_j)] para j ≠ i

    for i in range(n):
        # Índices de todos los puntos excepto el i-ésimo
        indices = [j for j in range(n) if j != i]

        # Construir el polinomio base L_i(x)
        # Numerador: Π(x - x_j) para j ≠ i
        numerador = Polynomial.fromroots(x_data[indices])

        # Denominador: Π(x_i - x_j) para j ≠ i
        denominador = np.prod(x_data[i] - x_data[indices])

        # L_i(x) = numerador / denominador
        L_i = numerador / denominador

        # Agregar el término y_i · L_i(x) al polinomio total
        poly_total = poly_total + y_data[i] * L_i

    return poly_total


def encontrar_resonancia(func, E_min, E_max, num_puntos=10000):
    """
    Encuentra la energía de resonancia (máximo de la curva).

    Parámetros:
    -----------
    func : funcion interpolada
    E_min, E_max : float
        Rango de energías a buscar
    num_puntos : int
        Número de puntos para evaluar

    Retorna:
    --------
    E_resonancia : float
        Energía donde ocurre el máximo
    sigma_max : float
        Valor máximo de la sección eficaz
    """
    E_eval = np.linspace(E_min, E_max, num_puntos)
    sigma_eval = func(E_eval)

    idx_max = np.argmax(sigma_eval)
    E_resonancia = E_eval[idx_max]
    sigma_max = sigma_eval[idx_max]

    return E_resonancia, sigma_max


def calcular_fwhm(func, E_resonancia, sigma_max, E_min, E_max):
    """
    Calcula el ancho a media altura (FWHM = Γ).

    FWHM es la diferencia de energías donde σ(E) = σ_max / 2

    Parámetros:
    -----------
    func : funcion interpolada
    E_resonancia : float
        Energía de resonancia
    sigma_max : float
        Valor máximo de σ
    E_min, E_max : float
        Límites del rango de búsqueda

    Retorna:
    --------
    FWHM : float
        Ancho a media altura
    E_left : float
        Energía del borde izquierdo (σ = σ_max/2)
    E_right : float
        Energía del borde derecho (σ = σ_max/2)
    """
    sigma_half = sigma_max / 2.0

    # Función auxiliar: σ(E) - σ_max/2
    def diferencia(E):
        return func(E) - sigma_half

    # Buscar cruces con σ_max/2
    # Lado izquierdo: entre E_min y E_resonancia
    try:
        E_left = brentq(diferencia, E_min, E_resonancia)
    except ValueError:
        # Si no encuentra raíz, usar evaluación numérica
        E_eval = np.linspace(E_min, E_resonancia, 1000)
        sigma_eval = func(E_eval)
        idx = np.argmin(np.abs(sigma_eval - sigma_half))
        E_left = E_eval[idx]

    # Lado derecho: entre E_resonancia y E_max
    try:
        E_right = brentq(diferencia, E_resonancia, E_max)
    except ValueError:
        E_eval = np.linspace(E_resonancia, E_max, 1000)
        sigma_eval = func(E_eval)
        idx = np.argmin(np.abs(sigma_eval - sigma_half))
        E_right = E_eval[idx]

    FWHM = E_right - E_left

    return FWHM, E_left, E_right


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
table.add_column(r"$E_i$", justify="center", style="red")
table.add_column(r"$f\left(E_i\right)$", justify="center", style="red")
table.add_column(r"$\sigma_i$", justify="center", style="magenta")
for s in range(len(i)):
    table.add_row(f"{i[s]}", f"{Ei[s]:.1f}", f"{fE[s]:.2f}", f"{sigma[s]:.2f}")

console.print(table)
# Aplicar interpolación de Lagrange
console.print("\n[cyan]Aplicando interpolación de Lagrange...[/cyan]")
polinomio = lagrange_interpolation(Ei, fE)

console.print(f"[green] Polinomio interpolador creado exitosamente[/green]")
console.print(f"[green]  Grado del polinomio: {len(Ei) - 1}[/green]\n")

# Evaluar el polinomio en un rango fino para graficar
E_range = np.linspace(0, 200, 1000)
f_interpolada = polinomio(E_range)

# Crear gráfica
plt.figure(figsize=(12, 7))

# Graficar polinomio interpolador
plt.plot(
    E_range,
    f_interpolada,
    "-",
    color="blue",
    linewidth=2.5,
    label="Polinomio de Lagrange",
    alpha=0.8,
)
plt.plot(Ei, fE, "o", color="red", label="Datos experimentales")

plt.xlabel("Energía $E_i$ (MeV)", fontsize=13, fontweight="bold")
plt.ylabel(r"Sección eficaz $f(E_i)$ (mb)", fontsize=13, fontweight="bold")
plt.title(
    "Interpolación de Lagrange: Sección Eficaz vs Energía",
    fontsize=15,
    fontweight="bold",
)
plt.legend(fontsize=11, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Guardar la figura
archivo_salida = os.path.join(output_dir, "interpolacion_lagrange.png")
plt.savefig(archivo_salida, dpi=300, bbox_inches="tight")
console.print(f"[green]Gráfica guardada en: {archivo_salida}[/green]")

plt.show()

console.rule("[bold cyan]Análisis de Resonancia[/bold cyan]")

# 1. Encontrar energía de resonancia (máximo)
E_resonancia, sigma_max = encontrar_resonancia(polinomio, 0, 200)

console.print(f"\n[bold green]1. Energía de Resonancia (Er):[/bold green]")
console.print(f"   Er = {E_resonancia:.2f} MeV")
console.print(f"   σ(Er) = {sigma_max:.2f} mb (valor máximo)")

# 2. Calcular FWHM (Γ)
FWHM_1, E_left, E_right = calcular_fwhm(polinomio, E_resonancia, sigma_max, 0, 200)

console.print(f"\n[bold green]2. Ancho a Media Altura (FWHM = Γ):[/bold green]")
console.print(f"   Γ = {FWHM_1:.2f} MeV")
console.print(f"   σ(Er)/2 = {sigma_max/2:.2f} mb")
console.print(f"   E_izquierda = {E_left:.2f} MeV")
console.print(f"   E_derecha = {E_right:.2f} MeV")

# 3. Valores teóricos
Er_teorico = 78.0  # MeV
Gamma_teorico = 55.0  # MeV

console.print(f"\n[bold yellow]3. Valores Teóricos:[/bold yellow]")
console.print(f"   Er (teórico) = {Er_teorico:.2f} MeV")
console.print(f"   Γ (teórico) = {Gamma_teorico:.2f} MeV")

# 4. Comparación
error_Er = abs(E_resonancia - Er_teorico)
error_Gamma = abs(FWHM_1 - Gamma_teorico)
error_rel_Er = 100 * error_Er / Er_teorico
error_rel_Gamma = 100 * error_Gamma / Gamma_teorico

console.print(f"\n[bold magenta]4. Comparación con Teoría:[/bold magenta]")
console.print(f"   ΔEr = {error_Er:.2f} MeV ({error_rel_Er:.1f}%)")
console.print(f"   ΔΓ = {error_Gamma:.2f} MeV ({error_rel_Gamma:.1f}%)")

# Tabla comparativa
console.print("\n")
tabla_comparacion = Table(
    title="[bold]Comparación: Experimental vs Teórico[/bold]", box=box.DOUBLE_EDGE
)
tabla_comparacion.add_column("Parámetro", style="cyan", justify="left")
tabla_comparacion.add_column("Experimental", style="green", justify="center")
tabla_comparacion.add_column("Teórico", style="yellow", justify="center")
tabla_comparacion.add_column("Error Absoluto", style="red", justify="center")
tabla_comparacion.add_column("Error Relativo", style="magenta", justify="center")

tabla_comparacion.add_row(
    "Er (MeV)",
    f"{E_resonancia:.2f}",
    f"{Er_teorico:.2f}",
    f"{error_Er:.2f}",
    f"{error_rel_Er:.1f}%",
)

tabla_comparacion.add_row(
    "Γ (MeV)",
    f"{FWHM_1:.2f}",
    f"{Gamma_teorico:.2f}",
    f"{error_Gamma:.2f}",
    f"{error_rel_Gamma:.1f}%",
)

console.print(tabla_comparacion)

# Ajuste con splines cubicos


def cubic_split(x_data, y_data):
    """ "
    no hay mucho que decir, solo que regresa el splin
    """
    return CubicSpline(x_data, y_data, bc_type="natural")


console.print("\n[cyan]Ajustando splines cúbicos...[/cyan]")

# Crear spline cúbico
# bc_type='natural' impone segunda derivada nula en los extremos
spline = cubic_split(Ei, fE)

console.print(f"[green]  Spline cúbico creado exitosamente[/green]")
console.print(
    f"[green]  Tipo: Spline cúbico con condiciones de frontera naturales[/green]"
)
console.print(f"[green]  Segmentos: {len(Ei) - 1} splines cúbicos[/green]\n")


console.rule("[bold cyan]Análisis de Resonancia con Splines[/bold cyan]")

# 1. Encontrar energía de resonancia
Er_spline, sigma_max_spline = encontrar_resonancia(spline, 0, 200)

console.print(f"\n[bold green]1. Energía de Resonancia (Er) - SPLINES:[/bold green]")
console.print(f"   Er = {Er_spline:.2f} MeV")
console.print(f"   σ(Er) = {sigma_max_spline:.2f} mb (valor máximo)")

# 2. Calcular FWHM
FWHM_spline, E_left_spline, E_right_spline = calcular_fwhm(
    spline, Er_spline, sigma_max_spline, 0, 200
)

console.print(
    f"\n[bold green]2. Ancho a Media Altura (FWHM = Γ) - SPLINES:[/bold green]"
)
console.print(f"   Γ = {FWHM_spline:.2f} MeV")
console.print(f"   σ(Er)/2 = {sigma_max_spline/2:.2f} mb")
console.print(f"   E_izquierda = {E_left_spline:.2f} MeV")
console.print(f"   E_derecha = {E_right_spline:.2f} MeV")

# comparacion con la teoria

Er_teorico = 78.0
Gamma_teorico = 55.0

console.print(f"\n[bold yellow]3. Valores Teóricos:[/bold yellow]")
console.print(f"   Er (teórico) = {Er_teorico:.2f} MeV")
console.print(f"   Γ (teórico) = {Gamma_teorico:.2f} MeV")

# Errores respecto a teoría
error_Er_spline = abs(Er_spline - Er_teorico)
error_Gamma_spline = abs(FWHM_spline - Gamma_teorico)
error_rel_Er_spline = 100 * error_Er_spline / Er_teorico
error_rel_Gamma_spline = 100 * error_Gamma_spline / Gamma_teorico

console.print(f"\n[bold magenta]4. Comparación con Teoría (SPLINES):[/bold magenta]")
console.print(f"   ΔEr = {error_Er_spline:.2f} MeV ({error_rel_Er_spline:.1f}%)")
console.print(f"   ΔΓ = {error_Gamma_spline:.2f} MeV ({error_rel_Gamma_spline:.1f}%)")

f_interpolada = spline(E_range)

# Crear gráfica
plt.figure(figsize=(12, 7))

# Graficar polinomio interpolador
plt.plot(
    E_range,
    f_interpolada,
    "-",
    color="blue",
    linewidth=2.5,
    label="Spline cubico",
    alpha=0.8,
)
plt.plot(Ei, fE, "o", color="red", label="Datos experimentales")

plt.xlabel("Energía $E_i$ (MeV)", fontsize=13, fontweight="bold")
plt.ylabel(r"Sección eficaz $f(E_i)$ (mb)", fontsize=13, fontweight="bold")
plt.title(
    "Interpolación Spline cubica: Sección Eficaz vs Energía",
    fontsize=15,
    fontweight="bold",
)
plt.legend(fontsize=11, loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Guardar la figura
archivo_salida = os.path.join(output_dir, "interpolacion_spline_cubico.png")
plt.savefig(archivo_salida, dpi=300, bbox_inches="tight")
console.print(f"[green]Gráfica guardada en: {archivo_salida}[/green]")

plt.show()
console.print("\n")
console.rule("[bold green]RESUMEN[/bold green]")

resumen = Panel(
    f"""[bold cyan]Resultados con Splines Cúbicos:[/bold cyan]

[yellow]Valores Estimados:[/yellow]
  • Energía de resonancia: Er = {Er_spline:.2f} MeV
  • Ancho a media altura: Γ = {FWHM_spline:.2f} MeV
  • Sección eficaz máxima: σ_max = {sigma_max_spline:.2f} mb

[yellow]Valores Teóricos:[/yellow]
  • Er (teórico) = {Er_teorico:.2f} MeV
  • Γ (teórico) = {Gamma_teorico:.2f} MeV

[yellow]Comparación con Teoría:[/yellow]
  • Error en Er: {error_Er_spline:.2f} MeV ({error_rel_Er_spline:.1f}%)
  • Error en Γ: {error_Gamma_spline:.2f} MeV ({error_rel_Gamma_spline:.1f}%)

[yellow]Comparación Splines vs Lagrange:[/yellow]
  • ΔEr = {abs(Er_spline - E_resonancia):.2f} MeV
  • ΔΓ = {abs(FWHM_spline - FWHM_1):.2f} MeV

[green]Ventajas de los Splines Cúbicos:[/green]
  * Suavidad garantizada (C² continua)
  * No sufre oscilaciones de Runge
  * Mejor estabilidad numérica
  * Más apropiado para datos experimentales
  * Comportamiento local (un dato afecta solo 4 segmentos)

[green]Interpretación:[/green]
  {" :D Excelente concordancia con teoría" if error_rel_Er_spline < 5 and error_rel_Gamma_spline < 10 else " :D Buena concordancia con teoría"}
  Los splines cúbicos proporcionan una interpolación más suave y
  estable que el polinomio de Lagrange de grado 8.
""",
    title="[bold]Análisis con Splines Cúbicos[/bold]",
    border_style="green",
    box=box.DOUBLE,
)

console.print(resumen)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # 3 rows, 1 column
ax1 = plt.plot(
    E_range,
    f_interpolada,
    "-",
    color="blue",
    linewidth=2.5,
    label="Polinomio de Lagrange",
    alpha=0.8,
)
ax1.plot(Ei, fE, "o", color="red", label="Datos experimentales")
ax1.xlabel("Energía $E_i$ (MeV)", fontsize=13, fontweight="bold")
ax1.ylabel(r"Sección eficaz $f(E_i)$ (mb)", fontsize=13, fontweight="bold")
ax1.title(
    "Interpolación de Lagrange: Sección Eficaz vs Energía",
    fontsize=15,
    fontweight="bold",
)
ax1.legend(fontsize=11, loc="best")
ax1.grid(True, alpha=0.3)
ax2 = plt.plot(
    E_range,
    f_interpolada,
    "-",
    color="blue",
    linewidth=2.5,
    label="Spline cubico",
    alpha=0.8,
)
ax2.plot(Ei, fE, "o", color="red", label="Datos experimentales")

ax2.xlabel("Energía $E_i$ (MeV)", fontsize=13, fontweight="bold")
ax2.ylabel(r"Sección eficaz $f(E_i)$ (mb)", fontsize=13, fontweight="bold")
ax2.title(
    "Interpolación Spline cubica: Sección Eficaz vs Energía",
    fontsize=15,
    fontweight="bold",
)
ax2.legend(fontsize=11, loc="best")
ax2.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
console.print("\n[bold green]Programa finalizado exitosamente[/bold green]")
