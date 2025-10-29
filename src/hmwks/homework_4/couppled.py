"""
Codeado: Martes 28/10/2025
Considere el sistema de resortes que se muestra en la figura 1.

(a) Escriba las ecuaciones de movimiento (acopladas) para los desplazamientos de las dos masas igaules a lo largo de x:

(b) Calcule las frecuencias de los modos normales de vibración del sistema.

(c) Gráfique las posiciones de las masas en función del tiempo cuando estas inician su movimiento en la forma siguiente:
i. Ambas masas parten del reposo habiendo sido desplazadas una igual cantidad hacia la derecha.
ii. Ambas masas parten del reposo habiendo sido desplazadas una igual cantidad en sentidos opuestos.
iii. Una masa parte de su posición de equilibrio y la otra de una posición desplazada hacia la derecha.

(d) Si suponemos que los resortes no son lineales y la fuerza tiene la forma
\[
F = -k(x+0.1x^3)
\]
repita el proceso del iniciso b, y compare las respuestas del caso lineal y el caso no lineal.
"""

import os

import numpy as np
from matplotlib.animation import FuncAnimation
from pylab import *
from rich.console import Console
from rich.table import Table
from scipy.integrate import odeint

# Cosas importantes pero no tan importantes que la física
output_dir = "resultados_harm_test"  # Para guardar mis cosas
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
console = Console()  # Para que se vea bonito

# definimos lo imoortante: las constantes físicas del sistema
N = 2000  # numero de pasos
m = [1.0, 2.0, 0.5]  # masa de los bloques la de 2. va con [4] y la de 0.5 con [5]
k = [12.0, 10.0, 8.0, 15.0, 10.0, 20.0]  # constante de los resortes externos
k_c = [4.0, 2.0, 12.0, 5.0, 4.0, 8.0]  # constante del resorte central
tau = 30.0  # tiempo de la simulación
h = tau / float(N - 1)  # el paso del gigante (del tiempo)

ttime = linspace(0, tau, N)  # Generamos tiempo


def calcular_periodo(t, x):
    """
    Calcula el periodo promedio encontrando cruces por cero.
    """
    # Encontrar cruces por cero con pendiente positiva
    cruces = []
    for i in range(len(x) - 1):
        if x[i] <= 0 and x[i + 1] > 0:  # Cruce ascendente
            # Interpolación lineal para mayor precisión
            t_cruce = t[i] - x[i] * (t[i + 1] - t[i]) / (x[i + 1] - x[i])
            cruces.append(t_cruce)

    if len(cruces) < 2:
        return None

    # Calcular diferencias entre cruces consecutivos (medio periodo)
    # Multiplicar por 2 para obtener periodo completo
    periodos = [(cruces[i + 1] - cruces[i]) * 2 for i in range(len(cruces) - 1)]

    return mean(periodos)


def coupled_not_linear(r, t):
    """
    Sistema de EDOs para osciladores acoplados NO LINEALES.

    Fuerza no lineal: F = -k(x + 0.1x³)

    Ecuaciones:
        dx1/dt = v1
        dv1/dt = -k/m·(x1 + 0.1x1³) - k'/m·[(x1-x2) + 0.1(x1-x2)³]
        dx2/dt = v2
        dv2/dt = -k/m·(x2 + 0.1x2³) - k'/m·[(x2-x1) + 0.1(x2-x1)³]
    """
    r_1 = r[0]
    v_1 = r[1]
    r_2 = r[2]
    v_2 = r[3]
    # Fuerzas no lineales:
    F1_nolin = -k[1] / m[0] * (r_1 + 0.1 * r_1**3)
    F1_coupled = -k_c[3] / m[0] * ((r_1 - r_2) + 0.1 * (r_1 - r_2) ** 3)
    F2_nolin = -k[1] / m[0] * (r_2 + 0.1 * r_1**3)
    F2_coupled = -k_c[3] / m[0] * ((r_2 - r_1) + 0.1 * (r_2 - r_2) ** 3)

    # Ecuaciones diferenciales:
    dx_1_dt = v_1
    dv_1_dt = F1_coupled + F1_nolin
    dx_2_dt = v_2
    dv_2_dt = F2_coupled + F2_nolin
    return array([dx_1_dt, dv_1_dt, dx_2_dt, dv_2_dt])


def coupled(r, t):
    """
    Sistema para un oscilador armonico acoplado

    Variables de estado:
        r[0] = x_1 Posición de la masa (de la izquierda)
        r[1] = v_1 Velocidad de la masa (de la izquierda)
        r[2] = x_2 Posición de la masa (de la derecha)
        r[3] = v_2 Velocidad de la masa (de la derecha)

    Ecuaciones:
        dx_1/dt = v_1
        dv_1/dt = -(k+k_c)/m * x_1 + k_c/m*x_2
        dx_2/dt = v_2
        dv_2/dt = k_c/m * x_1 -(k+k_c)/m * x_2
    """
    r_1 = r[0]
    v_1 = r[1]
    r_2 = r[2]
    v_2 = r[3]
    dx_1_dt = v_1
    dv_1_dt = -(k[1] + k_c[3]) / m[0] * r_1 + k_c[3] / m[0] * r_2
    dx_2_dt = v_2
    dv_2_dt = k_c[3] / m[0] * r_1 - (k[1] + k_c[3]) / m[0] * r_2

    return array([dx_1_dt, dv_1_dt, dx_2_dt, dv_2_dt])


# condiciones iniciales:
conditions = [
    {
        "name": "Mismo deslpazamiento hacia la derecha",
        "state0": [0.8, 0, 0.8, 0],
        "description": "x₁(0) = x₂(0) = 0.8 m, v₁(0) = v₂(0) = 0",
    },
    {
        "name": "Deslpazamiento en sentidos opuestos",
        "state0": [0.8, 0, -0.8, 0],
        "description": "x₁(0) = 0.8 m x₂(0) = -0.8 m, v₁(0) = v₂(0) = 0",
    },
    {
        "name": "Una en equilibrio, otra desplazada",
        "state0": [0, 0, 1.0, 0],
        "description": "x₁(0) =0 m x₂(0) = -1 m, v₁(0) = v₂(0) = 0",
    },
]
console.print("Simulando cada cosa ...", style="blink on blue")

for idx, condition in enumerate(conditions, 1):
    console.print(f"\n  {condition['name']}")
    console.print(f"    Condición: {condition['description']}")

    # Resolver EDOs
    solution = odeint(coupled, condition["state0"], ttime)
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
    ax1.plot(ttime, x1, "b-", linewidth=2, label="Masa 1 (x₁)")
    ax1.plot(ttime, x2, "r-", linewidth=2, label="Masa 2 (x₂)")
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
    console.print(f"    [green] Guardado: {filename}[/green]")
    show()
    close(fig)

console.print("\n[bold yellow]Consideramos ahora ecuaciones no lineales:[/bold yellow]")
console.print("  F₁ = -k(x₁ + 0.1x₁³) - k'[(x₁-x₂) + 0.1(x₁-x₂)³]")
console.print("  F₂ = -k(x₂ + 0.1x₂³) - k'[(x₂-x₁) + 0.1(x₂-x₁)³]")

console.print("\n[bold]Comparación: Sistema lineal vs no lineal[/bold]")
console.print("Resimulando cada cosa ...", style="blink on blue")

for idx, condition in enumerate(conditions, 1):
    console.print(f"\n  {condition['name']}")

    # Resolver ambos sistemas
    sol_linear = odeint(coupled, condition["state0"], ttime)
    sol_nonlinear = odeint(coupled_not_linear, condition["state0"], ttime)

    # Crear figura comparativa
    fig, axes = subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Comparación Lineal vs No Lineal\n{condition['name']}",
        fontsize=15,
        fontweight="bold",
    )

    # Subplot 1: Masa 1 - Lineal vs No lineal
    axes[0, 0].plot(ttime, sol_linear[:, 0], "b-", linewidth=2, label="Lineal")
    axes[0, 0].plot(ttime, sol_nonlinear[:, 0], "r--", linewidth=2, label="No lineal")
    axes[0, 0].axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0, 0].set_xlabel("Tiempo (s)", fontsize=10)
    axes[0, 0].set_ylabel("Posición x₁ (m)", fontsize=10)
    axes[0, 0].set_title("Masa 1", fontsize=11, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)

    # Subplot 2: Masa 2 - Lineal vs No lineal
    axes[0, 1].plot(ttime, sol_linear[:, 2], "b-", linewidth=2, label="Lineal")
    axes[0, 1].plot(ttime, sol_nonlinear[:, 2], "r--", linewidth=2, label="No lineal")
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
    console.print(f"    [green] Guardado: {filename}[/green]")
    show()
    close(fig)

# Por ultimo calculamos los modos normales de vibración para ambos casos:
console.print("\n" + "=" * 70, style="bold cyan")
console.print(
    " ANÁLISIS SIMPLE: MODOS NORMALES (LINEAL VS NO LINEAL)", style="bold cyan"
)
console.print("=" * 70, style="bold cyan")

# MODOS NORMALES TEÓRICOS (LINEAL)

console.print("\n" + "=" * 70, style="bold yellow")
console.print(" MODOS NORMALES TEÓRICOS (SISTEMA LINEAL)", style="bold yellow")
console.print("=" * 70, style="bold yellow")

# Matriz del sistema
A = array(
    [[-(k[1] + k_c[3]) / m[0], k_c[3] / m[0]], [k_c[3] / m[0], -(k[1] + k_c[3]) / m[0]]]
)

eigenvalues, eigenvectors = linalg.eig(A)
omega_teorico = sqrt(-eigenvalues)
freq_teorico = omega_teorico / (2 * pi)
periodo_teorico = 1 / freq_teorico

console.print(f"\n[bold]Modo 1 (Simétrico):[/bold]")
console.print(f"  ω₁ = {omega_teorico[0]:.4f} rad/s")
console.print(f"  f₁ = {freq_teorico[0]:.4f} Hz")
console.print(f"  T₁ = {periodo_teorico[0]:.4f} s")

console.print(f"\n[bold]Modo 2 (Antisimétrico):[/bold]")
console.print(f"  ω₂ = {omega_teorico[1]:.4f} rad/s")
console.print(f"  f₂ = {freq_teorico[1]:.4f} Hz")
console.print(f"  T₂ = {periodo_teorico[1]:.4f} s")

# ANÁLISIS PARA DIFERENTES AMPLITUDES
console.print("\n" + "=" * 70, style="bold green")
console.print(" ANÁLISIS NUMÉRICO: DEPENDENCIA CON AMPLITUD", style="bold green")
console.print("=" * 70, style="bold green")

amplitudes = [0.2, 0.5, 0.8, 1.2]

# Almacenar resultados
resultados = {
    "amplitud": [],
    "T_lineal": [],
    "T_nonlinear": [],
    "f_lineal": [],
    "f_nonlinear": [],
    "diff_pct": [],
}

for amp in amplitudes:
    console.print(f"\n[bold cyan]Amplitud: {amp} m[/bold cyan]")

    # Condición inicial: modo simétrico
    state0 = [amp, 0.0, amp, 0.0]

    # Resolver
    sol_lin = odeint(coupled, state0, ttime)
    sol_nonlin = odeint(coupled_not_linear, state0, ttime)

    # Calcular periodos
    T_lin = calcular_periodo(ttime, sol_lin[:, 0])
    T_nonlin = calcular_periodo(ttime, sol_nonlin[:, 0])

    if T_lin and T_nonlin:
        f_lin = 1 / T_lin
        f_nonlin = 1 / T_nonlin
        diff = abs(f_nonlin - f_lin) / f_lin * 100

        resultados["amplitud"].append(amp)
        resultados["T_lineal"].append(T_lin)
        resultados["T_nonlinear"].append(T_nonlin)
        resultados["f_lineal"].append(f_lin)
        resultados["f_nonlinear"].append(f_nonlin)
        resultados["diff_pct"].append(diff)

        console.print(f"  Lineal:    T = {T_lin:.4f} s,  f = {f_lin:.4f} Hz")
        console.print(f"  No lineal: T = {T_nonlin:.4f} s,  f = {f_nonlin:.4f} Hz")
        console.print(f"  Diferencia: {diff:.2f}%")

# ===================================================================
# TABLA COMPARATIVA
# ===================================================================
console.print("\n" + "=" * 70, style="bold magenta")
console.print(" TABLA COMPARATIVA", style="bold magenta")
console.print("=" * 70, style="bold magenta")

table = Table(title="Frecuencias vs Amplitud", style="magenta")
table.add_column("Amplitud (m)", justify="center")
table.add_column("f Lineal (Hz)", justify="center")
table.add_column("f No Lineal (Hz)", justify="center")
table.add_column("Diferencia (%)", justify="center")

for i in range(len(resultados["amplitud"])):
    table.add_row(
        f"{resultados['amplitud'][i]:.1f}",
        f"{resultados['f_lineal'][i]:.4f}",
        f"{resultados['f_nonlinear'][i]:.4f}",
        f"{resultados['diff_pct'][i]:.2f}",
    )

console.print(table)

console.print("\n" + "=" * 70, style="bold green")
console.print(" RESUMEN DE RESULTADOS", style="bold green")
console.print("=" * 70, style="bold green")

# console.print(f"\n[bold]Frecuencias de modos normales:[/bold]")
# console.print(f"  ω₁ = {omega[0]:.3f} rad/s (modo simétrico)")
# console.print(f"  ω₂ = {omega[1]:.3f} rad/s (modo antisimétrico)")

console.print(f"\n[bold]Diferencias entre sistema lineal y no lineal:[/bold]")
console.print("  • Sistema lineal: Oscilaciones armónicas puras (sinusoidales)")
console.print("  • Sistema no lineal: Distorsión de la forma de onda")
console.print("  • No linealidad introduce dependencia de amplitud en frecuencia")
console.print("  • Espacio de fase no lineal muestra trayectorias distorsionadas")

console.print(f"\n[bold cyan] Archivos generados en '{output_dir}/':[/bold cyan]")
console.print("  • caso_1_lineal.png - Ambas masas a la derecha (lineal)")
console.print("  • caso_2_lineal.png - Masas en sentidos opuestos (lineal)")
console.print("  • caso_3_lineal.png - Una en equilibrio (lineal)")
console.print("  • caso_1_comparacion.png - Comparación lineal vs no lineal")
console.print("  • caso_2_comparacion.png - Comparación lineal vs no lineal")
console.print("  • caso_3_comparacion.png - Comparación lineal vs no lineal")

console.print("\n" + "=" * 70, style="bold green")
console.print(" PROCESO COMPLETADO", style="bold green")
console.print("=" * 70 + "\n", style="bold green")
