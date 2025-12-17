"""
Normalización de la Distribución de Fermi-Dirac
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table
from scipy.integrate import quad
from scipy.optimize import brentq, fsolve

# Configuración de Rich para output elegante
cons = Console()

# Crear directorio para resultados
output_dir = "resultados_fermi_dirac"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    cons.print(f"[bold green]Directorio creado: {output_dir}[/bold green]\n")


# Creamos la distribucion
def fermi_dirac(E, mu, kT=0.025):
    """
    Distribución de Fermi-Dirac.

    Parámetros:
    -----------
    E : float o array
        Energía en eV
    mu : float
        Energía de Fermi (parámetro químico) en eV
    kT : float, opcional
        Energía térmica kT en eV (default: 0.025 eV ≈ temperatura ambiente)

    Retorna:
    --------
    f_FD : float o array
        Probabilidad de ocupación

    Notas:
    ------
    Para evitar overflow en exp(), usamos la forma estable:
    - Si (E-μ)/kT >> 0: f ≈ exp(-(E-μ)/kT)
    - Si (E-μ)/kT << 0: f ≈ 1
    """
    x = (E - mu) / kT

    # Manejo numérico estable para evitar overflow
    # Si x es muy positivo, exp(x) -> ∞, entonces f -> 0
    # Si x es muy negativo, exp(x) -> 0, entonces f -> 1

    # Usamos la forma: f = 1/(1 + exp(x))
    # Para x > 50, exp(x) es gigante -> f ≈ 0
    # Para x < -50, exp(x) ≈ 0 -> f ≈ 1

    return 1.0 / (np.exp(np.clip(x, -500, 500)) + 1.0)


# que vamos a normalizar?
def integral_fermi_dirac(mu, E_min=0.0, E_max=2.0, kT=0.025):
    """
    Calcula la integral int_{E_min}^{E_max} f_FD(E; μ) dE

    Parámetros:
    -----------
    mu : float
        Energía de Fermi (parámetro a ajustar)
    E_min, E_max : float
        Límites de integración (en eV)
    kT : float
        Energía térmica (en eV)

    Retorna:
    --------
    integral : float
        Valor de la integral
    error : float
        Estimación del error numérico
    """

    # Definir función integrando con μ fijo
    def integrand(E):
        return fermi_dirac(E, mu, kT)

    # Integración numérica adaptativa (cuadratura de Gauss-Legendre)
    integral, error = quad(integrand, E_min, E_max, limit=100)

    return integral, error


def objetivo_normalizacion(mu, E_min=0.0, E_max=2.0, kT=0.025):
    """
    Función objetivo F(μ) = int f_FD(E; μ) dE - 1

    Queremos encontrar μ* tal que F(μ*) = 0
    """
    integral, _ = integral_fermi_dirac(mu, E_min, E_max, kT)
    return integral - 1.0


# veamos que sigue
cons.rule("[bold cyan]EXPLORACIÓN DE LA FUNCIÓN OBJETIVO F(μ)")

# Explorar F(μ) en un rango amplio
mu_range = np.linspace(-1.0, 3.0, 50)
F_values = []

cons.print("[yellow]Calculando F(μ) para diferentes valores de μ...[/yellow]\n")

for mu in mu_range:
    F_mu = objetivo_normalizacion(mu)
    F_values.append(F_mu)

F_values = np.array(F_values)

# Encontrar cambio de signo
sign_changes = np.where(np.diff(np.sign(F_values)))[0]

if len(sign_changes) > 0:
    idx = sign_changes[0]
    mu_lower = mu_range[idx]
    mu_upper = mu_range[idx + 1]

    cons.print(f"[green]:D Cambio de signo detectado en:[/green]")
    cons.print(f"  μ in [{mu_lower:.3f}, {mu_upper:.3f}] eV\n")
else:
    cons.print(
        "[red]D:< No se detectó cambio de signo. Ajustar rango de búsqueda.[/red]\n"
    )
    mu_lower, mu_upper = 0.0, 2.0

# Crear gráfico de exploración
fig_exploracion = plt.figure(figsize=(10, 6))
plt.plot(mu_range, F_values, "b-", linewidth=2, label="F(μ)")
plt.axhline(0, color="red", linestyle="--", linewidth=1.5, label="F = 0")
plt.axvline(
    mu_lower,
    color="green",
    linestyle=":",
    alpha=0.7,
    label=f"μ_lower = {mu_lower:.3f} eV",
)
plt.axvline(
    mu_upper,
    color="orange",
    linestyle=":",
    alpha=0.7,
    label=f"μ_upper = {mu_upper:.3f} eV",
)
plt.xlabel("μ (eV)", fontsize=12)
plt.ylabel("F(μ) = ∫f_FD dE - 1", fontsize=12)
plt.title("Función Objetivo: Búsqueda de Raíz", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

filename_exploracion = os.path.join(output_dir, "exploracion_funcion_objetivo.png")
plt.savefig(filename_exploracion, dpi=300, bbox_inches="tight")
cons.print(f"[bold green]:D Gráfico guardado: {filename_exploracion}[/bold green]\n")
plt.show()

# Buscamos las raices :D

cons.rule("[bold cyan]INCISO (a): CÁLCULO DE μ")

cons.print("[yellow]Aplicando método de Brent para encontrar μ*...[/yellow]\n")

# Método de Brent (robusto y eficiente)
mu_solution = brentq(
    objetivo_normalizacion,
    mu_lower,
    mu_upper,
    args=(),  # Argumentos adicionales si los hubiera
    xtol=1e-10,  # Tolerancia en μ
    rtol=1e-10,  # Tolerancia relativa
    maxiter=100,
    full_output=False,
)

# Verificación
integral_final, error_integral = integral_fermi_dirac(mu_solution)
F_final = objetivo_normalizacion(mu_solution)

# Tabla de resultados
tabla_resultados = Table(title="Resultados de la Normalización", box=box.DOUBLE)
tabla_resultados.add_column("Parámetro", style="cyan", justify="left")
tabla_resultados.add_column("Valor", style="yellow", justify="right")
tabla_resultados.add_column("Unidad", style="green", justify="left")

tabla_resultados.add_row("μ* (energía de Fermi)", f"{mu_solution:.10f}", "eV")
tabla_resultados.add_row("∫₀² f_FD(E) dE", f"{integral_final:.10f}", "")
tabla_resultados.add_row("Error de integración", f"{error_integral:.2e}", "")
tabla_resultados.add_row("F(μ*) = int - 1", f"{F_final:.2e}", "")
tabla_resultados.add_row("kT (temperatura)", f"{0.025:.3f}", "eV")
tabla_resultados.add_row("Intervalo [E_min, E_max]", "[0.00, 2.00]", "eV")

cons.print(tabla_resultados)
cons.print()

# Verificación de normalización
cons.print(
    f"[bold magenta]-> Energía de Fermi obtenida: μ* = {mu_solution:.6f} eV[/bold magenta]"
)
cons.print(
    f"[bold green]-> Verificación: int f_FD dE = {integral_final:.10f} ≈ 1.0 :D[/bold green]\n"
)

# grafica de distribución

cons.rule("[bold cyan]INCISO (b): GRÁFICA DE f_FD(E)")

# Crear array de energías para graficar
E_plot = np.linspace(0, 2, 500)
f_FD_plot = fermi_dirac(E_plot, mu_solution, kT=0.025)

# Crear figura
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel izquierdo: Distribución de Fermi-Dirac
axes[0].plot(E_plot, f_FD_plot, "b-", linewidth=2.5, label="$f_{FD}(E)$")
axes[0].axvline(
    mu_solution,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"μ = {mu_solution:.4f} eV",
)
axes[0].axhline(
    0.5, color="green", linestyle=":", linewidth=1.5, alpha=0.7, label="f = 0.5"
)
axes[0].scatter(
    [mu_solution],
    [0.5],
    color="red",
    s=100,
    zorder=5,
    marker="o",
    edgecolor="black",
    linewidth=2,
)

axes[0].set_xlabel("Energía E (eV)", fontsize=12)
axes[0].set_ylabel("Probabilidad $f_{FD}(E)$", fontsize=12)
axes[0].set_title(
    "Distribución de Fermi-Dirac Normalizada", fontsize=13, fontweight="bold"
)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)
axes[0].set_xlim(0, 2)
axes[0].set_ylim(0, 1.05)

# Panel derecho: Zoom en la región de transición
E_zoom = np.linspace(mu_solution - 0.3, mu_solution + 0.3, 300)
f_zoom = fermi_dirac(E_zoom, mu_solution, kT=0.025)

axes[1].plot(E_zoom, f_zoom, "b-", linewidth=2.5)
axes[1].axvline(
    mu_solution,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"μ = {mu_solution:.4f} eV",
)
axes[1].axhline(0.5, color="green", linestyle=":", linewidth=1.5, alpha=0.7)
axes[1].scatter(
    [mu_solution],
    [0.5],
    color="red",
    s=150,
    zorder=5,
    marker="o",
    edgecolor="black",
    linewidth=2,
)

# Marcar región ±kT
axes[1].axvspan(
    mu_solution - 0.025, mu_solution + 0.025, alpha=0.2, color="orange", label="μ ± kT"
)

axes[1].set_xlabel("Energía E (eV)", fontsize=12)
axes[1].set_ylabel(r"Probabilidad $f_{FD}(E)$", fontsize=12)
axes[1].set_title("Región de Transición (Zoom)", fontsize=13, fontweight="bold")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=9)

plt.tight_layout()
filename_distribucion = os.path.join(output_dir, "distribucion_fermi_dirac.png")
plt.savefig(filename_distribucion, dpi=300, bbox_inches="tight")
cons.print(f"[bold green]:D Gráfico guardado: {filename_distribucion}[/bold green]\n")
plt.show()
# cons.rule("[bold cyan]ANÁLISIS ADICIONAL")

# # Calcular algunas propiedades interesantes
# E_mean = quad(lambda E: E * fermi_dirac(E, mu_solution, 0.025), 0, 2)[0]
# E_squared = quad(lambda E: E**2 * fermi_dirac(E, mu_solution, 0.025), 0, 2)[0]
# E_std = np.sqrt(E_squared - E_mean**2)

# # Probabilidades en regiones específicas
# P_below_mu = quad(lambda E: fermi_dirac(E, mu_solution, 0.025), 0, mu_solution)[0]
# P_above_mu = quad(lambda E: fermi_dirac(E, mu_solution, 0.025), mu_solution, 2)[0]

# # Ancho de la transición (rango donde f va de ~0.1 a ~0.9)
# # Aproximadamente ±2kT alrededor de μ
# ancho_transicion = 4 * 0.025  # ≈ 0.1 eV

# tabla_analisis = Table(title="Análisis de la Distribución", box=box.ROUNDED)
# tabla_analisis.add_column("Propiedad", style="cyan", justify="left")
# tabla_analisis.add_column("Valor", style="yellow", justify="right")

# tabla_analisis.add_row("Energía promedio ⟨E⟩", f"{E_mean:.6f} eV")
# tabla_analisis.add_row("Desviación estándar σ_E", f"{E_std:.6f} eV")
# tabla_analisis.add_row("P(E < μ)", f"{P_below_mu:.6f}")
# tabla_analisis.add_row("P(E > μ)", f"{P_above_mu:.6f}")
# tabla_analisis.add_row("Ancho transición (~4kT)", f"{ancho_transicion:.4f} eV")
# tabla_analisis.add_row("f_FD(E=0)", f"{fermi_dirac(0, mu_solution, 0.025):.6f}")
# tabla_analisis.add_row("f_FD(E=2)", f"{fermi_dirac(2, mu_solution, 0.025):.6f}")

# cons.print(tabla_analisis)
# cons.print()

cons.rule("[bold green]CONCLUSIONES")

conclusiones = f"""
[bold cyan]1. Valor de μ encontrado:[/bold cyan]
   μ* = {mu_solution:.8f} eV
   
   Este valor garantiza que ∫₀² f_FD(E) dE = 1.0000000000

[bold cyan]2. Interpretación física:[/bold cyan]
   • La energía de Fermi μ ≈ {mu_solution:.3f} eV está aproximadamente en el 
     punto medio del intervalo [0, 2] eV.
   
   • A temperatura ambiente (kT = 0.025 eV), la transición de f ≈ 1 a f ≈ 0 
     ocurre en una ventana estrecha de ~0.1 eV alrededor de μ.

[bold cyan]3. Validación numérica:[/bold cyan]
   • El método de Brent convergió exitosamente
   • Error en la normalización: |int - 1| < 10⁻¹⁰
   • Error de integración numérica: {error_integral:.2e}

[bold cyan]4. Observaciones:[/bold cyan]
   • La distribución es suave pero pronunciada debido a kT << (E_max - E_min)
   • En el límite T -> 0, f_FD se convertiría en una función escalón
"""
# • La probabilidad de encontrar la partícula con E < μ es {P_below_mu:.3f}
#    • La probabilidad de encontrar la partícula con E > μ es {P_above_mu:.3f}
# • La energía promedio ⟨E⟩ = {E_mean:.3f} eV está ligeramente por debajo de μ
cons.print(conclusiones)

cons.rule("[bold green]:D PROBLEMA RESUELTO EXITOSAMENTE")
