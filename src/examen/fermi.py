"""
(a) Imaginemos que el sistema cu·ntico se encuentra a temperatura
ambiente (kT  0:025 eV ) donde, por alguna razÛn, la energÌa
E est· restringida entre 0 y 2 eV . øCu·l es el valor de  en este
caso?
(b) Realice una gr·Öca de fF D (E) en el intervalo indicado con el valor
de  estimado."""

# librerias para instalar mas librerias desde el spyder pq ocupo rich por comodidad
import subprocess
import sys
import importlib.metadata

# funcion para instalar dependencias rapido
def instalar_dependencias_rapido(deps_texto):
    """Instala dependencias faltantes de forma rápida"""
    deps = [l.strip() for l in deps_texto.strip().split("\n") if l.strip()]
    faltantes = []
# 
    print(f"Verificando {len(deps)} paquetes...")
# 
    for dep in deps:
        nombre = dep.split("==")[0]
        try:
            importlib.metadata.distribution(nombre)
        except:
            faltantes.append(dep)
# 
    if not faltantes:
        print("✓ Todas las dependencias instaladas")
        return
# 
    print(f"Instalando {len(faltantes)} paquetes faltantes...")
# 
    for i, dep in enumerate(faltantes, 1):
        print(f"  [{i}/{len(faltantes)}] {dep.split('==')[0]}...", end=" ")
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    dep,
                    "--break-system-packages",
                    "-q",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("✓")
        except:
            print("✗")
# 
    print("Instalación completada")
# 
# 
# Uso
DEPS = """
certifi==2025.11.12
charset-normalizer==3.4.4
contourpy==1.3.3
cycler==0.12.1
docutils==0.22.3
fonttools==4.61.1
id==1.5.0
idna==3.11
jaraco.classes==3.4.0
jaraco.context==6.0.1
jaraco.functools==4.3.0
keyring==25.7.0
kiwisolver==1.4.9
markdown-it-py==4.0.0
matplot==0.1.9
matplotlib==3.10.8
mdurl==0.1.2
more-itertools==10.8.0
nh3==0.3.2
numpy==2.3.5
packaging==25.0
pillow==12.0.0
Pygments==2.19.2
pyloco==0.0.139
pyparsing==3.2.5
python-dateutil==2.9.0.post0
pywin32-ctypes==0.2.3
readme_renderer==44.0
requests==2.32.5
requests-toolbelt==1.0.0
rfc3986==2.0.0
rich==14.2.0
scipy==1.16.3
SimpleWebSocketServer==0.1.2
six==1.17.0
twine==6.2.0
typing==3.7.4.3
urllib3==2.6.2
ushlex==0.99.1
websocket-client==1.9.0
"""
# 
instalar_dependencias_rapido(DEPS)
# librerias ahora si para el examen:
import os
# 
import matplotlib.pyplot as plt
import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table
from scipy.integrate import quad
from scipy.optimize import brentq, fsolve

# Creamos una consola para un output elegante
cons = Console()

output_dir = "resultados_examen_final"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    cons.print(
        f"[bold green] El directorio no existia, se van a guardar los graficos en: {output_dir}"
    )


# definimos la distribucion
def fermi_dirac(E, mu, kT=0.025):
    x = (E - mu) / kT
    return 1.0 / (np.exp(np.clip(x, -500, 500)) + 1.0)


# que es lo que buscamos hacer 1?
def integral_fermi_dirac(mu, E_min=0.0, E_max=2.0, kT=0.025):
    # Definir función integrando con mu fijo
    def integrand(E):
        return fermi_dirac(E, mu, kT)

    # Integración numérica adaptativa (cuadratura de Gauss-Legendre)
    integral, error = quad(integrand, E_min, E_max, limit=100)

    return integral, error


# objetivo:
def objetivo_normalizacion(mu, E_min=0.0, E_max=2.0, kT=0.025):
    """
    Lo que buscamos es:
    integral f = 1
    por lo que para usar brentq hacemos:
    integral f - 1 = 0 :D
    """
    integral, _ = integral_fermi_dirac(mu, E_min, E_max, kT)
    return integral - 1.0


cons.rule("[red] Primera parte inciso del inciso a:")
cons.print("[bold cyan]exploramos la funcion F(mu)")

# Explorar F(mu) en un rango amplio
mu_range = np.linspace(-1.5, 3.0, 50)
F_values = []

cons.print("[yellow]Calculando F(mu) para diferentes valores de mu...[/yellow]\n")

# para cada mu del linspace, vamos a explorar como se comporta F(mu)
for mu in mu_range:
    F_mu = objetivo_normalizacion(mu)
    F_values.append(F_mu)

F_values = np.array(F_values)

# Buscamos un cambio de signo
sign_changes = np.where(np.diff(np.sign(F_values)))[0]

# verificamos que si en efecto hay un cambio de signo, hagamos lo siguiente:
if len(sign_changes) > 0:
    idx = sign_changes[0]
    mu_lower = mu_range[idx]
    mu_upper = mu_range[idx + 1]

    cons.print(f"[green]:D Cambio de signo detectado en:[/green]")
    cons.print(f"  mu in [{mu_lower:.3f}, {mu_upper:.3f}] eV\n")
else:
    cons.print(
        "[red]D:< No se detectó cambio de signo. Ajustar rango de búsqueda.[/red]\n"
    )
    mu_lower, mu_upper = 0.0, 2.0

# Crear gráfico de util
fig_exploracion = plt.figure(figsize=(10, 6))
plt.plot(mu_range, F_values, "b-", linewidth=2, label="F(μ)")
plt.axhline(0, color="red", linestyle="--", linewidth=1.5, label="F = 0")
plt.axvline(
    mu_lower,
    color="green",
    linestyle=":",
    alpha=0.7,
    label=f"mu_lower = {mu_lower:.3f} eV",
)
plt.axvline(
    mu_upper,
    color="orange",
    linestyle=":",
    alpha=0.7,
    label=f"mu_upper = {mu_upper:.3f} eV",
)
plt.xlabel("mu (eV)", fontsize=12)
plt.ylabel(r"F($\mu$) = $\int F(mu) -1$ ", fontsize=12)
plt.title("Función Objetivo: Búsqueda de Raíz", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

filename_exploracion = os.path.join(output_dir, "exploracion_funcion_objetivo.png")
plt.savefig(filename_exploracion, dpi=300, bbox_inches="tight")
cons.print(f"[bold green]:D Gráfico guardado: {filename_exploracion}[/bold green]\n")
plt.show()

# Buscamos las raices :D ahora si
cons.rule("[bold cyan]Calculo de mu")

cons.print("[yellow]Aplicando el método de Brent para encontrar mu*...[/yellow]\n")

# Método de Brent (robusto y eficiente)
mu_solution = brentq(
    objetivo_normalizacion,
    mu_lower,
    mu_upper,
    args=(),  # Argumentos adicionales si los hubiera
    xtol=1e-10,  # Tolerancia en mu
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

tabla_resultados.add_row("mu* (energía de Fermi)", f"{mu_solution:.10f}", "eV")
tabla_resultados.add_row("int_0^2 f_FD(E) dE", f"{integral_final:.10f}", "")
tabla_resultados.add_row("Error de integración", f"{error_integral:.2e}", "")
tabla_resultados.add_row("F(mu*) = int - 1", f"{F_final:.2e}", "")
tabla_resultados.add_row("kT (temperatura)", f"{0.025:.3f}", "eV")
tabla_resultados.add_row("Intervalo [E_min, E_max]", "[0.00, 2.00]", "eV")

cons.print(tabla_resultados)
cons.print()

# Verificación de normalización
cons.print(
    f"[bold magenta]-> Energía de Fermi obtenida: mu* = {mu_solution:.6f} eV[/bold magenta]"
)
cons.print(
    f"[bold green]-> Verificación: int f_FD dE = {integral_final:.10f} ~~ (approx) 1.0 :D[/bold green]\n"
)

# grafica de distribución

cons.rule("[bold cyan]Primera parte del inciso b: graficos de f_FD(E)")

# Crear array de energías para graficar
E_plot = np.linspace(0, 2, 500)
f_FD_plot = fermi_dirac(E_plot, mu_solution, kT=0.025)

# Crear figura
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel izquierdo: Distribución de Fermi-Dirac
axes[0].plot(E_plot, f_FD_plot, "b-", linewidth=2.5, label=r"$f_{FD}(E)$")
axes[0].axvline(
    mu_solution,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"mu = {mu_solution:.4f} eV",
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
axes[0].set_ylabel(r"Probabilidad $f_{FD}(E)$", fontsize=12)
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
    label=f"mu = {mu_solution:.4f} eV",
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
    mu_solution - 0.025, mu_solution + 0.025, alpha=0.2, color="orange", label=r"mu $\pm$ kT"
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

cons.rule("[bold green]Resumen")

conclusiones = f"""
[bold cyan]1. Valor de μ encontrado:[/bold cyan]
   μ* = {mu_solution:.8f} eV
   
   Este valor garantiza que int f_FD(E) dE = 1.0000000

[bold cyan]2. Interpretación física:[/bold cyan]
   * La energía de Fermi mu ~~ {mu_solution:.3f} eV está aproximadamente en el 
     punto medio del intervalo [0, 2] eV.
   
   * A temperatura ambiente (kT ~ 0.025 eV), la transición de f ~ 1 a f ~ 0 
     ocurre en una ventana estrecha de ~0.1 eV alrededor de mu.

[bold cyan]3. Validación numérica:[/bold cyan]
   * El método de Brent convergió exitosamente
   * Error en la normalización: |int - 1| < 10^(-10)
   * Error de integración numérica: {error_integral:.2e}

[bold cyan]4. Observaciones:[/bold cyan]
   * La distribución es suave pero pronunciada debido a kT << (E_max - E_min)
"""
cons.print(conclusiones)

cons.rule("[bold green]:D PROBLEMA RESUELTO EXITOSAMENTE")