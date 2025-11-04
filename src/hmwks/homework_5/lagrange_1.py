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
from rich.console import Console  # Hay que darle formato a la consola
from rich.table import Table  # Hay que darle formato a la consola
from scipy.optimize import brentq

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
def encontrar_resonancia(polinomio, E_min, E_max, num_puntos=10000):
    """
    Encuentra la energía de resonancia (máximo de la curva).
    
    Parámetros:
    -----------
    polinomio : Polynomial
        Polinomio interpolador
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
    sigma_eval = polinomio(E_eval)
    
    idx_max = np.argmax(sigma_eval)
    E_resonancia = E_eval[idx_max]
    sigma_max = sigma_eval[idx_max]
    
    return E_resonancia, sigma_max
def calcular_fwhm(polinomio, E_resonancia, sigma_max, E_min, E_max):
    """
    Calcula el ancho a media altura (FWHM = Γ).
    
    FWHM es la diferencia de energías donde σ(E) = σ_max / 2
    
    Parámetros:
    -----------
    polinomio : Polynomial
        Polinomio interpolador
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
        return polinomio(E) - sigma_half
    
    # Buscar cruces con σ_max/2
    # Lado izquierdo: entre E_min y E_resonancia
    try:
        E_left = brentq(diferencia, E_min, E_resonancia)
    except ValueError:
        # Si no encuentra raíz, usar evaluación numérica
        E_eval = np.linspace(E_min, E_resonancia, 1000)
        sigma_eval = polinomio(E_eval)
        idx = np.argmin(np.abs(sigma_eval - sigma_half))
        E_left = E_eval[idx]
    
    # Lado derecho: entre E_resonancia y E_max
    try:
        E_right = brentq(diferencia, E_resonancia, E_max)
    except ValueError:
        E_eval = np.linspace(E_resonancia, E_max, 1000)
        sigma_eval = polinomio(E_eval)
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

# Graficar puntos experimentales con barras de error
#plt.errorbar(Ei, fE, yerr=sigma, fmt='o', color='red', markersize=8,
#             capsize=5, capthick=2, elinewidth=2, 
#             label='Datos medidos', zorder=5)

# Graficar polinomio interpolador
plt.plot(E_range, f_interpolada, '-', color='blue', linewidth=2.5,
         label='Polinomio de Lagrange', alpha=0.8)
plt.plot(Ei, fE, 'o', color='red',
         label='Datos experimentales')

plt.xlabel('Energía $E_i$ (MeV)', fontsize=13, fontweight='bold')
plt.ylabel(r'Sección eficaz $f(E_i)$ (mb)', fontsize=13, fontweight='bold')
plt.title('Interpolación de Lagrange: Sección Eficaz vs Energía', 
          fontsize=15, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Guardar la figura
archivo_salida = os.path.join(output_dir, 'interpolacion_lagrange.png')
plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
console.print(f"[green]Gráfica guardada en: {archivo_salida}[/green]")

plt.show()

# ==============================================================================
# EJEMPLOS DE USO DEL POLINOMIO
# ==============================================================================

#console.print("\n[bold cyan]Ejemplos de evaluación del polinomio:[/bold cyan]")

# Evaluar en algunos puntos
#puntos_prueba = [30, 60, 90, 120, 150]
#console.print("\nValores interpolados:")
#for E_test in puntos_prueba:
#    sigma_test = polinomio(E_test)
#    console.print(f"  E = {E_test:3d} MeV  →  σ(E) = {sigma_test:.2f} mb")

# Verificar que el polinomio pasa por los puntos originales
#console.print("\n[bold green]Verificación (el polinomio debe pasar exactamente por los puntos):[/bold green]")
#errores_verificacion = []
#for E_orig, f_orig in zip(Ei, fE):
#    f_calculada = polinomio(E_orig)
#    error = abs(f_calculada - f_orig)
#    errores_verificacion.append(error)

#error_maximo = max(errores_verificacion)
#console.print(f"Error máximo en puntos originales: {error_maximo:.2e}")

#if error_maximo < 1e-8:
#    console.print("[green]✓ El polinomio pasa exactamente por todos los puntos[/green]")
#else:
#    console.print("[yellow]⚠ Hay pequeños errores numéricos (normales en punto flotante)[/yellow]")

console.print("\n[bold green]Programa finalizado exitosamente[/bold green]")
