"""
Problema 3: Simulación de decaimiento de mesones π (piones)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

# Crear consola
cons = Console()

# Crear directorio para resultados
output_dir = "resultados_tarea_7"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    cons.print(f"[bold red]Directorio creado: {output_dir}[/bold red]\n")
else:
    cons.print(f"[bold green]Directorio existente: {output_dir}[/bold green]\n")

# Constantes
m_pion = 139.6  # MeV/c²
tau_reposo = 2.6e-8  # segundos
c = 3e8  # m/s
distancia = 20  # metros
N_inicial = 1_000_000  # 1 millón de piones

cons.rule("[bold red]SIMULACIÓN DE MESONES π (PIONES)")

# Tabla de constantes
tabla_const = Table(title="Constantes del Problema", box=box.ROUNDED)
tabla_const.add_column("Parámetro", style="cyan", justify="left")
tabla_const.add_column("Valor", style="yellow", justify="right")

tabla_const.add_row("Masa del pión", f"{m_pion} MeV/c²")
tabla_const.add_row("Tiempo de vida (reposo)", f"{tau_reposo:.2e} s")
tabla_const.add_row("Velocidad de la luz", f"{c:.2e} m/s")
tabla_const.add_row("Distancia a recorrer", f"{distancia} m")
tabla_const.add_row("Número inicial de piones", f"{N_inicial:,}")

cons.print(tabla_const)
cons.print()

cons.rule("[bold cyan]PIONES MONOENERGÉTICOS")

K = 200  # MeV

# Cálculos
E_total = K + m_pion
gamma = E_total / m_pion
beta = np.sqrt(1 - 1 / gamma**2)
v = beta * c
tau_dilatado = gamma * tau_reposo
tiempo_viaje = distancia / v
P_sobrevivir = np.exp(-tiempo_viaje / tau_dilatado)

# Simular
sobreviven = np.random.random(N_inicial) < P_sobrevivir
N_sobreviven = np.sum(sobreviven)

# Calcular incertidumbre (distribución binomial)
incertidumbre_mono = np.sqrt(N_inicial * P_sobrevivir * (1 - P_sobrevivir))

# Tabla de resultados
tabla_mono = Table(title="Resultados - Monoenergético (K=200 MeV)", box=box.DOUBLE)
tabla_mono.add_column("Cantidad", style="cyan", justify="left")
tabla_mono.add_column("Valor", style="yellow", justify="right")

tabla_mono.add_row("Energía cinética", f"{K} MeV")
tabla_mono.add_row("Energía total", f"{E_total:.2f} MeV")
tabla_mono.add_row("Factor γ (Lorentz)", f"{gamma:.4f}")
tabla_mono.add_row("Velocidad", f"{v:.3e} m/s")
tabla_mono.add_row("Velocidad (en c)", f"{beta:.4f}c")
tabla_mono.add_row("Tiempo de vida dilatado", f"{tau_dilatado:.3e} s")
tabla_mono.add_row("Tiempo de viaje (20 m)", f"{tiempo_viaje:.3e} s")
tabla_mono.add_row("Probabilidad de sobrevivir", f"{P_sobrevivir:.6f}")

cons.print(tabla_mono)

# Resultado final con incertidumbre
cons.print()
cons.print(
    f"[bold green]→ Sobreviven: {N_sobreviven:,} ± {incertidumbre_mono:.0f} piones[/bold green]"
)
cons.print(
    f"[bold magenta]→ Porcentaje: {N_sobreviven/N_inicial*100:.2f}% ± {incertidumbre_mono/N_inicial*100:.2f}%[/bold magenta]"
)
cons.print(
    f"[bold yellow]→ Esperado teórico: {N_inicial*P_sobrevivir:.0f} piones[/bold yellow]"
)
cons.print()

cons.print("[cyan]Generando gráficos para monoenergeticos...[/cyan]")

fig_a, axes_a = plt.subplots(1, 2, figsize=(12, 5))
fig_a.suptitle("PIONES MONOENERGÉTICOS (K = 200 MeV)", fontsize=14, fontweight="bold")

# Gráfico 1a: Factor gamma vs energía
K_range = np.linspace(50, 400, 100)
E_range = K_range + m_pion
gamma_range = E_range / m_pion

axes_a[0].plot(K_range, gamma_range, linewidth=2.5, color="purple")
axes_a[0].axvline(200, color="red", linestyle="--", linewidth=2, label="K = 200 MeV")
axes_a[0].axhline(
    gamma, color="orange", linestyle=":", linewidth=2, label=f"γ = {gamma:.3f}"
)
axes_a[0].scatter([200], [gamma], color="red", s=100, zorder=5)
axes_a[0].set_xlabel("Energía cinética (MeV)", fontsize=11)
axes_a[0].set_ylabel("Factor γ (Lorentz)", fontsize=11)
axes_a[0].set_title("(a.1) Factor de Lorentz vs Energía", fontweight="bold")
axes_a[0].legend(fontsize=9)
axes_a[0].grid(alpha=0.3)

# Gráfico 2a: Resultado visual
categorias = ["Inicial", "Sobreviven\n(20 m)", "Decaen"]
valores = [N_inicial, N_sobreviven, N_inicial - N_sobreviven]
colores = ["steelblue", "green", "red"]

bars = axes_a[1].bar(
    categorias, valores, color=colores, alpha=0.7, edgecolor="black", linewidth=2
)
axes_a[1].set_ylabel("Número de piones", fontsize=11)
axes_a[1].set_title("(a.2) Resultado de la Simulación", fontweight="bold")
axes_a[1].grid(axis="y", alpha=0.3)

# Agregar valores sobre las barras
for i, (v, cat) in enumerate(zip(valores, categorias)):
    if i == 1:  # Para sobrevivientes, mostrar con incertidumbre
        axes_a[1].text(
            i,
            v + 20000,
            f"{v:,}\n± {incertidumbre_mono:.0f}\n({v/N_inicial*100:.2f}%)",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
        # Agregar barra de error
        axes_a[1].errorbar(
            i,
            v,
            yerr=incertidumbre_mono,
            fmt="none",
            color="black",
            capsize=5,
            capthick=2,
        )
    else:
        axes_a[1].text(
            i,
            v + 20000,
            f"{v:,}\n({v/N_inicial*100:.1f}%)",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )

plt.tight_layout()
filename_a = os.path.join(output_dir, "parte_a_monoenergetico.png")
plt.savefig(filename_a, dpi=300, bbox_inches="tight")
cons.print(f"[bold green]:D Gráfico guardado: {filename_a}[/bold green]\n")
plt.show()

cons.rule("[bold cyan]PIONES CON DISTRIBUCIÓN GAUSSIANA")

K_mean = 200  # MeV
K_sigma = 50  # MeV

cons.print(f"[cyan]Generando energías con distribución Gaussiana...[/cyan]")
cons.print(f"[cyan]Media: {K_mean} MeV, Desviación: {K_sigma} MeV[/cyan]\n")

# Generar energías
energias_cineticas = np.random.normal(K_mean, K_sigma, N_inicial)
energias_cineticas = np.abs(energias_cineticas)

# Calcular para cada pión
E_totales = energias_cineticas + m_pion
gammas = E_totales / m_pion
betas = np.sqrt(1 - 1 / gammas**2)
velocidades = betas * c
tau_dilatados = gammas * tau_reposo
tiempos_viaje = distancia / velocidades
P_sobrevivir_array = np.exp(-tiempos_viaje / tau_dilatados)

# Simular
sobreviven_gauss = np.random.random(N_inicial) < P_sobrevivir_array
N_sobreviven_gauss = np.sum(sobreviven_gauss)

# Calcular incertidumbre
P_mean = np.mean(P_sobrevivir_array)
incertidumbre_gauss = np.sqrt(N_inicial * P_mean * (1 - P_mean))

# Incertidumbres adicionales
sigma_energia = np.std(energias_cineticas)
sigma_gamma = np.std(gammas)
sigma_prob = np.std(P_sobrevivir_array)

# Tabla de resultados
tabla_gauss = Table(
    title="Resultados - Gaussiano (μ=200 MeV, σ=50 MeV)", box=box.DOUBLE
)
tabla_gauss.add_column("Estadística", style="cyan", justify="left")
tabla_gauss.add_column("Valor ± Incertidumbre", style="yellow", justify="right")

tabla_gauss.add_row(
    "Energía", f"{np.mean(energias_cineticas):.2f} ± {sigma_energia:.2f} MeV"
)
tabla_gauss.add_row(
    "Energía (rango)",
    f"[{energias_cineticas.min():.1f}, {energias_cineticas.max():.1f}] MeV",
)
tabla_gauss.add_row("Factor γ", f"{np.mean(gammas):.4f} ± {sigma_gamma:.4f}")
tabla_gauss.add_row("Factor γ (rango)", f"[{gammas.min():.3f}, {gammas.max():.3f}]")
tabla_gauss.add_row(
    "Probabilidad", f"{np.mean(P_sobrevivir_array):.6f} ± {sigma_prob:.6f}"
)

cons.print(tabla_gauss)

cons.print()
cons.print(
    f"[bold green]→ Sobreviven: {N_sobreviven_gauss:,} ± {incertidumbre_gauss:.0f} piones[/bold green]"
)
cons.print(
    f"[bold magenta]→ Porcentaje: {N_sobreviven_gauss/N_inicial*100:.2f}% ± {incertidumbre_gauss/N_inicial*100:.2f}%[/bold magenta]"
)
cons.print(
    f"[bold yellow]→ Esperado teórico: {N_inicial*P_mean:.0f} piones[/bold yellow]"
)
cons.print()

cons.print("[cyan]Generando gráficos para distribucion gaussiana...[/cyan]")

fig_b, axes_b = plt.subplots(2, 2, figsize=(12, 9))
fig_b.suptitle(
    "PIONES CON DISTRIBUCIÓN GAUSSIANA (μ=200 MeV, σ=50 MeV)",
    fontsize=14,
    fontweight="bold",
)

# Gráfico 1b: Distribución de energías
axes_b[0, 0].hist(
    energias_cineticas,
    bins=60,
    color="steelblue",
    alpha=0.7,
    edgecolor="black",
    density=True,
)
axes_b[0, 0].axvline(
    K_mean,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Media = {np.mean(energias_cineticas):.1f} MeV",
)
axes_b[0, 0].axvline(
    K_mean - K_sigma, color="orange", linestyle=":", linewidth=1.5, alpha=0.7
)
axes_b[0, 0].axvline(
    K_mean + K_sigma,
    color="orange",
    linestyle=":",
    linewidth=1.5,
    alpha=0.7,
    label=f"±σ = ±{sigma_energia:.1f} MeV",
)
axes_b[0, 0].set_xlabel("Energía cinética (MeV)", fontsize=10)
axes_b[0, 0].set_ylabel("Densidad de probabilidad", fontsize=10)
axes_b[0, 0].set_title("(b.1) Distribución de Energías", fontweight="bold")
axes_b[0, 0].legend(fontsize=8)
axes_b[0, 0].grid(alpha=0.3)

# Gráfico 2b: Distribución de factor gamma
axes_b[0, 1].hist(
    gammas, bins=60, color="purple", alpha=0.7, edgecolor="black", density=True
)
axes_b[0, 1].axvline(
    np.mean(gammas),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Media = {np.mean(gammas):.3f}",
)
axes_b[0, 1].axvline(
    gamma,
    color="green",
    linestyle=":",
    linewidth=2,
    label=f"Monoenergético = {gamma:.3f}",
)
axes_b[0, 1].set_xlabel("Factor γ (Lorentz)", fontsize=10)
axes_b[0, 1].set_ylabel("Densidad de probabilidad", fontsize=10)
axes_b[0, 1].set_title("(b.2) Distribución de Factor γ", fontweight="bold")
axes_b[0, 1].legend(fontsize=8)
axes_b[0, 1].grid(alpha=0.3)

# Gráfico 3b: Distribución de probabilidades
axes_b[1, 0].hist(
    P_sobrevivir_array,
    bins=60,
    color="green",
    alpha=0.7,
    edgecolor="black",
    density=True,
)
axes_b[1, 0].axvline(
    P_sobrevivir,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Monoenergético = {P_sobrevivir:.4f}",
)
axes_b[1, 0].axvline(
    np.mean(P_sobrevivir_array),
    color="orange",
    linestyle=":",
    linewidth=2,
    label=f"Media Gaussiano = {np.mean(P_sobrevivir_array):.4f}",
)
axes_b[1, 0].set_xlabel("Probabilidad de sobrevivir", fontsize=10)
axes_b[1, 0].set_ylabel("Densidad de probabilidad", fontsize=10)
axes_b[1, 0].set_title("(b.3) Distribución de Probabilidades", fontweight="bold")
axes_b[1, 0].legend(fontsize=8)
axes_b[1, 0].grid(alpha=0.3)

# Gráfico 4b: Resultado visual
categorias_b = ["Inicial", "Sobreviven\n(20 m)", "Decaen"]
valores_b = [N_inicial, N_sobreviven_gauss, N_inicial - N_sobreviven_gauss]
colores_b = ["steelblue", "green", "red"]

bars_b = axes_b[1, 1].bar(
    categorias_b, valores_b, color=colores_b, alpha=0.7, edgecolor="black", linewidth=2
)
axes_b[1, 1].set_ylabel("Número de piones", fontsize=10)
axes_b[1, 1].set_title("(b.4) Resultado de la Simulación", fontweight="bold")
axes_b[1, 1].grid(axis="y", alpha=0.3)

# Agregar valores sobre las barras
for i, (v, cat) in enumerate(zip(valores_b, categorias_b)):
    if i == 1:  # Para sobrevivientes, mostrar con incertidumbre
        axes_b[1, 1].text(
            i,
            v + 20000,
            f"{v:,}\n± {incertidumbre_gauss:.0f}\n({v/N_inicial*100:.2f}%)",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
        # Agregar barra de error
        axes_b[1, 1].errorbar(
            i,
            v,
            yerr=incertidumbre_gauss,
            fmt="none",
            color="black",
            capsize=5,
            capthick=2,
        )
    else:
        axes_b[1, 1].text(
            i,
            v + 20000,
            f"{v:,}\n({v/N_inicial*100:.1f}%)",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )

plt.tight_layout()
filename_b = os.path.join(output_dir, "parte_b_gaussiano.png")
plt.savefig(filename_b, dpi=300, bbox_inches="tight")
cons.print(f"[bold green]:D Gráfico guardado: {filename_b}[/bold green]\n")
plt.show()

cons.rule("[bold yellow]COMPARACIÓN DE RESULTADOS")

tabla_comp = Table(title="Comparación Final", box=box.HEAVY)
tabla_comp.add_column("Caso", style="cyan", justify="center")
tabla_comp.add_column("Sobreviven", style="green", justify="center")
tabla_comp.add_column("Incertidumbre", style="yellow", justify="center")
tabla_comp.add_column("Porcentaje", style="magenta", justify="center")

tabla_comp.add_row(
    "Monoenergético (K=200 MeV)",
    f"{N_sobreviven:,}",
    f"± {incertidumbre_mono:.0f}",
    f"{N_sobreviven/N_inicial*100:.2f}%",
)

tabla_comp.add_row(
    "Gaussiano (μ=200, σ=50)",
    f"{N_sobreviven_gauss:,}",
    f"± {incertidumbre_gauss:.0f}",
    f"{N_sobreviven_gauss/N_inicial*100:.2f}%",
)

diferencia = abs(N_sobreviven - N_sobreviven_gauss)
tabla_comp.add_row(
    "Diferencia absoluta",
    f"{diferencia:,}",
    f"± {np.sqrt(incertidumbre_mono**2 + incertidumbre_gauss**2):.0f}",
    f"{diferencia/N_inicial*100:.2f}%",
)

cons.print(tabla_comp)
cons.print()
cons.rule("[bold green]:D SIMULACIÓN COMPLETA")
