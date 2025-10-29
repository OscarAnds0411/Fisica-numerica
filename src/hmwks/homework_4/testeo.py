"""
Considere el sistema de resortes que se muestra en la figura 1.

(a) Escriba las ecuaciones de movimiento (acopladas) para los desplazamientos de las dos masas igaules a lo largo de x:

(b) Calcule las frecuencias de los modos normales de vibración del sistema.

(c) Gráfique las posiciones de las masas en función del tiempo cuando estas inician su movimiento en la forma siguiente:
i. Ambas masas parten del reposo habiendo sido desplazadas una igual cantidad hacia la derecha.
ii. Ambas masas parten del reposo habiendo sido desplazadas una igual cantidad en sentidos opuestos.
iii. Una masa parte de su posición de equilibrio y la otra de una posición desplazada hacia la derecha.

(d) Si suponemos que los resortes no son lineales y la fuerza tiene la forma
F = -k(x+0.1x^3)
repita el proceso del iniciso b, y compare las respuestas del caso lineal y el caso no lineal.
"""

import os

import numpy as np
from matplotlib.animation import FuncAnimation
from pylab import *
from rich.console import Console
from rich.table import Table
from scipy.integrate import odeint

# ===================================================================
# PARÁMETROS DEL SISTEMA
# ===================================================================
m = 1.0  # Masa (kg)
k = 10.0  # Constante de resortes externos (N/m)
kp = 5.0  # Constante de resorte central (N/m)

# Crear carpeta para resultados
output_dir = "resultados_harm"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

console = Console()

# ===================================================================
# INCISO (a): ECUACIONES DE MOVIMIENTO
# ===================================================================
console.print("\n" + "=" * 70, style="bold cyan")
console.print(" INCISO (a): ECUACIONES DE MOVIMIENTO ACOPLADAS", style="bold cyan")
console.print("=" * 70, style="bold cyan")

console.print("\n[bold]Sistema:[/bold] |--k--m₁--k'--m₂--k--|")
console.print("\n[bold yellow]Ecuaciones diferenciales acopladas:[/bold yellow]")
console.print("  m·ẍ₁ = -k·x₁ - k'·(x₁ - x₂)")
console.print("  m·ẍ₂ = -k·x₂ - k'·(x₂ - x₁)")
console.print("\n[bold yellow]Forma matricial:[/bold yellow]")
console.print("  ẍ₁ = -(k + k')/m · x₁ + k'/m · x₂")
console.print("  ẍ₂ = k'/m · x₁ - (k + k')/m · x₂")


def equations_linear(state, t):
    """
    Sistema de EDOs para osciladores acoplados lineales.

    Variables de estado:
        state[0] = x1   (posición masa 1)
        state[1] = v1   (velocidad masa 1)
        state[2] = x2   (posición masa 2)
        state[3] = v2   (velocidad masa 2)

    Ecuaciones:
        dx1/dt = v1
        dv1/dt = -(k + k')/m · x1 + k'/m · x2
        dx2/dt = v2
        dv2/dt = k'/m · x1 - (k + k')/m · x2
    """
    x1, v1, x2, v2 = state

    dx1_dt = v1
    dv1_dt = -(k + kp) / m * x1 + kp / m * x2
    dx2_dt = v2
    dv2_dt = kp / m * x1 - (k + kp) / m * x2

    return array([dx1_dt, dv1_dt, dx2_dt, dv2_dt])


# ===================================================================
# INCISO (b): FRECUENCIAS DE MODOS NORMALES
# ===================================================================
console.print("\n" + "=" * 70, style="bold cyan")
console.print(" INCISO (b): FRECUENCIAS DE MODOS NORMALES", style="bold cyan")
console.print("=" * 70, style="bold cyan")

# Matriz de coeficientes del sistema
A = array([[-(k + kp) / m, kp / m], [kp / m, -(k + kp) / m]])

# Eigenvalores y eigenvectores
eigenvalues, eigenvectors = np.linalg.eig(A)

# Frecuencias angulares (ω² = -eigenvalue)
omega_squared = -eigenvalues
omega = np.sqrt(omega_squared)

# Frecuencias en Hz
frequencies = omega / (2 * np.pi)

# Períodos
periods = 1 / frequencies

console.print(f"\n[bold]Parámetros del sistema:[/bold]")
console.print(f"  m = {m} kg")
console.print(f"  k = {k} N/m")
console.print(f"  k' = {kp} N/m")

# Tabla de resultados
table = Table(title="Modos Normales de Vibración", style="cyan")
table.add_column("Modo", style="yellow", justify="center")
table.add_column("ω (rad/s)", justify="center")
table.add_column("f (Hz)", justify="center")
table.add_column("T (s)", justify="center")
table.add_column("Descripción", justify="left")

# Modo 1: Simétrico (ambas masas se mueven igual)
table.add_row(
    "1 (Simétrico)",
    f"{omega[0]:.3f}",
    f"{frequencies[0]:.3f}",
    f"{periods[0]:.3f}",
    "Masas oscilan en fase (→→ o ←←)",
)

# Modo 2: Antisimétrico (masas se mueven opuestas)
table.add_row(
    "2 (Antisimétrico)",
    f"{omega[1]:.3f}",
    f"{frequencies[1]:.3f}",
    f"{periods[1]:.3f}",
    "Masas oscilan fuera de fase (→← o ←→)",
)

console.print(table)

console.print("\n[bold green]✓ Eigenvectores (modos normales):[/bold green]")
console.print(f"  Modo 1: [{eigenvectors[0,0]:.3f}, {eigenvectors[1,0]:.3f}]")
console.print(f"  Modo 2: [{eigenvectors[0,1]:.3f}, {eigenvectors[1,1]:.3f}]")

# ===================================================================
# INCISO (c): GRÁFICAS PARA DIFERENTES CONDICIONES INICIALES
# ===================================================================
console.print("\n" + "=" * 70, style="bold cyan")
console.print(
    " INCISO (c): SIMULACIONES CON DIFERENTES CONDICIONES INICIALES", style="bold cyan"
)
console.print("=" * 70, style="bold cyan")

# Tiempo de simulación
t = linspace(0, 20, 1000)

# Condiciones iniciales
conditions = [
    {
        "name": "Ambas masas desplazadas hacia la derecha",
        "state0": [0.5, 0, 0.5, 0],  # [x1, v1, x2, v2]
        "description": "x₁(0) = x₂(0) = 0.5 m, v₁(0) = v₂(0) = 0",
    },
    {
        "name": "Masas desplazadas en sentidos opuestos",
        "state0": [0.5, 0, -0.5, 0],
        "description": "x₁(0) = 0.5 m, x₂(0) = -0.5 m, v₁(0) = v₂(0) = 0",
    },
    {
        "name": "Una en equilibrio, otra desplazada",
        "state0": [0, 0, 0.5, 0],
        "description": "x₁(0) = 0, x₂(0) = 0.5 m, v₁(0) = v₂(0) = 0",
    },
]

console.print("\n[bold]Simulando movimientos...[/bold]")

for idx, condition in enumerate(conditions, 1):
    console.print(f"\n  {condition['name']}")
    console.print(f"    Condición: {condition['description']}")

    # Resolver EDOs
    solution = odeint(equations_linear, condition["state0"], t)
    x1 = solution[:, 0]
    x2 = solution[:, 2]

    # Crear figura
    fig, (ax1, ax2) = subplots(2, 1, figsize=(12, 8))
    fig.suptitle(
        f"{condition['name']}\n{condition['description']}",
        fontsize=14,
        fontweight="bold",
    )

    # Subplot 1: Posiciones vs tiempo
    ax1.plot(t, x1, "b-", linewidth=2, label="Masa 1 (x₁)")
    ax1.plot(t, x2, "r-", linewidth=2, label="Masa 2 (x₂)")
    ax1.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_xlabel("Tiempo (s)", fontsize=11)
    ax1.set_ylabel("Posición (m)", fontsize=11)
    ax1.set_title("Posiciones de las masas vs tiempo", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Subplot 2: Espacio de fase (x1 vs x2)
    ax2.plot(x1, x2, "g-", linewidth=2, alpha=0.7)
    ax2.plot(x1[0], x2[0], "go", markersize=10, label="Inicio")
    ax2.plot(x1[-1], x2[-1], "rs", markersize=10, label="Final")
    ax2.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.axvline(x=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Posición masa 1, x₁ (m)", fontsize=11)
    ax2.set_ylabel("Posición masa 2, x₂ (m)", fontsize=11)
    ax2.set_title("Espacio de configuración (x₁ vs x₂)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.axis("equal")

    tight_layout()
    filename = f"{output_dir}/caso_{idx}_lineal.png"
    savefig(filename, dpi=300, bbox_inches="tight")
    console.print(f"    [green]✓ Guardado: {filename}[/green]")
    show()
    close(fig)

# ===================================================================
# INCISO (d): SISTEMA NO LINEAL
# ===================================================================
console.print("\n" + "=" * 70, style="bold cyan")
console.print(" INCISO (d): SISTEMA NO LINEAL F = -k(x + 0.1x³)", style="bold cyan")
console.print("=" * 70, style="bold cyan")


def equations_nonlinear(state, t):
    """
    Sistema de EDOs para osciladores acoplados NO LINEALES.

    Fuerza no lineal: F = -k(x + 0.1x³)

    Ecuaciones:
        dx1/dt = v1
        dv1/dt = -k/m·(x1 + 0.1x1³) - k'/m·[(x1-x2) + 0.1(x1-x2)³]
        dx2/dt = v2
        dv2/dt = -k/m·(x2 + 0.1x2³) - k'/m·[(x2-x1) + 0.1(x2-x1)³]
    """
    x1, v1, x2, v2 = state

    # Fuerza no lineal
    F1_ext = -k / m * (x1 + 0.1 * x1**3)
    F1_coupling = -kp / m * ((x1 - x2) + 0.1 * (x1 - x2) ** 3)

    F2_ext = -k / m * (x2 + 0.1 * x2**3)
    F2_coupling = -kp / m * ((x2 - x1) + 0.1 * (x2 - x1) ** 3)

    dx1_dt = v1
    dv1_dt = F1_ext + F1_coupling
    dx2_dt = v2
    dv2_dt = F2_ext + F2_coupling

    return array([dx1_dt, dv1_dt, dx2_dt, dv2_dt])


console.print("\n[bold yellow]Ecuaciones no lineales:[/bold yellow]")
console.print("  F₁ = -k(x₁ + 0.1x₁³) - k'[(x₁-x₂) + 0.1(x₁-x₂)³]")
console.print("  F₂ = -k(x₂ + 0.1x₂³) - k'[(x₂-x₁) + 0.1(x₂-x₁)³]")

console.print("\n[bold]Comparación: Sistema lineal vs no lineal[/bold]")

for idx, condition in enumerate(conditions, 1):
    console.print(f"\n  {condition['name']}")

    # Resolver ambos sistemas
    sol_linear = odeint(equations_linear, condition["state0"], t)
    sol_nonlinear = odeint(equations_nonlinear, condition["state0"], t)

    # Crear figura comparativa
    fig, axes = subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Comparación Lineal vs No Lineal\n{condition['name']}",
        fontsize=15,
        fontweight="bold",
    )

    # Subplot 1: Masa 1 - Lineal vs No lineal
    axes[0, 0].plot(t, sol_linear[:, 0], "b-", linewidth=2, label="Lineal")
    axes[0, 0].plot(t, sol_nonlinear[:, 0], "r--", linewidth=2, label="No lineal")
    axes[0, 0].axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0, 0].set_xlabel("Tiempo (s)", fontsize=10)
    axes[0, 0].set_ylabel("Posición x₁ (m)", fontsize=10)
    axes[0, 0].set_title("Masa 1", fontsize=11, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)

    # Subplot 2: Masa 2 - Lineal vs No lineal
    axes[0, 1].plot(t, sol_linear[:, 2], "b-", linewidth=2, label="Lineal")
    axes[0, 1].plot(t, sol_nonlinear[:, 2], "r--", linewidth=2, label="No lineal")
    axes[0, 1].axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0, 1].set_xlabel("Tiempo (s)", fontsize=10)
    axes[0, 1].set_ylabel("Posición x₂ (m)", fontsize=10)
    axes[0, 1].set_title("Masa 2", fontsize=11, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=9)

    # Subplot 3: Espacio de fase - Lineal
    axes[1, 0].plot(sol_linear[:, 0], sol_linear[:, 2], "b-", linewidth=2)
    axes[1, 0].plot(sol_linear[0, 0], sol_linear[0, 2], "go", markersize=8)
    axes[1, 0].axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[1, 0].axvline(x=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("x₁ (m)", fontsize=10)
    axes[1, 0].set_ylabel("x₂ (m)", fontsize=10)
    axes[1, 0].set_title("Espacio de fase - LINEAL", fontsize=11, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis("equal")

    # Subplot 4: Espacio de fase - No lineal
    axes[1, 1].plot(sol_nonlinear[:, 0], sol_nonlinear[:, 2], "r-", linewidth=2)
    axes[1, 1].plot(sol_nonlinear[0, 0], sol_nonlinear[0, 2], "go", markersize=8)
    axes[1, 1].axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[1, 1].axvline(x=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("x₁ (m)", fontsize=10)
    axes[1, 1].set_ylabel("x₂ (m)", fontsize=10)
    axes[1, 1].set_title("Espacio de fase - NO LINEAL", fontsize=11, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis("equal")

    tight_layout()
    filename = f"{output_dir}/caso_{idx}_comparacion.png"
    savefig(filename, dpi=300, bbox_inches="tight")
    console.print(f"    [green]✓ Guardado: {filename}[/green]")
    show()
    close(fig)

# ===================================================================
# RESUMEN FINAL
# ===================================================================
console.print("\n" + "=" * 70, style="bold green")
console.print(" RESUMEN DE RESULTADOS", style="bold green")
console.print("=" * 70, style="bold green")

console.print(f"\n[bold]Frecuencias de modos normales:[/bold]")
console.print(f"  ω₁ = {omega[0]:.3f} rad/s (modo simétrico)")
console.print(f"  ω₂ = {omega[1]:.3f} rad/s (modo antisimétrico)")

console.print(f"\n[bold]Diferencias entre sistema lineal y no lineal:[/bold]")
console.print("  • Sistema lineal: Oscilaciones armónicas puras (sinusoidales)")
console.print("  • Sistema no lineal: Distorsión de la forma de onda")
console.print("  • No linealidad introduce dependencia de amplitud en frecuencia")
console.print("  • Espacio de fase no lineal muestra trayectorias distorsionadas")

console.print(f"\n[bold cyan]✓ Archivos generados en '{output_dir}/':[/bold cyan]")
console.print("  • caso_1_lineal.png - Ambas masas a la derecha (lineal)")
console.print("  • caso_2_lineal.png - Masas en sentidos opuestos (lineal)")
console.print("  • caso_3_lineal.png - Una en equilibrio (lineal)")
console.print("  • caso_1_comparacion.png - Comparación lineal vs no lineal")
console.print("  • caso_2_comparacion.png - Comparación lineal vs no lineal")
console.print("  • caso_3_comparacion.png - Comparación lineal vs no lineal")

console.print("\n" + "=" * 70, style="bold green")
console.print(" PROCESO COMPLETADO", style="bold green")
console.print("=" * 70 + "\n", style="bold green")
