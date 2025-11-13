"""
Ajuste de Circuito RL - Decaimiento Exponencial del Voltaje
============================================================

Modelo: V(t) = V₀·e^(-τ·t)

donde:
  - V₀ = voltaje inicial
  - τ = R/L (constante de decaimiento)
  - R = resistencia
  - L = inductancia

Objetivos:
(a) Encontrar mejor estimado de τ y V₀ con incertidumbres
(b) Calcular χ² y evaluar bondad del ajuste
(c) Graficar datos y ajuste (escala semi-log)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import pandas as pd

console = Console()

data = pd.read_csv("datos_circuito.txt", header=0, delim_whitespace=True)

# t = data.iloc[:,0].to_numpy()
# V = data.iloc[:,1]
# sigma_V = data.iloc[:,2]

# ==============================================================================
# DATOS EXPERIMENTALES
# ==============================================================================

# # Tiempo (ms)
t = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 
              2.0, 2.2, 2.4, 2.6, 2.8, 3.0])

# # Voltaje (V)
V = np.array([5.02, 4.08, 3.33, 2.73, 2.24, 1.83, 1.50, 1.23, 1.00, 0.82,
              0.67, 0.55, 0.45, 0.37, 0.30, 0.25])

# Incertidumbre en voltaje (V)
# Si no se proporciona, estimarla como porcentaje del valor
sigma_V = 0.05 * V  # 5% de incertidumbre (típico para instrumentos)

# Mostrar datos
console.rule("[bold red]AJUSTE DE CIRCUITO RL[/bold red]")

table = Table(title="[bold yellow]Datos Experimentales[/bold yellow]", box=box.ROUNDED)
table.add_column("i", justify="center", style="cyan")
table.add_column("Tiempo t (ms)", justify="center", style="blue")
table.add_column("Voltaje V(t) (V)", justify="center", style="green")
table.add_column("σ_V (V)", justify="center", style="red")

for i, (ti, Vi, si) in enumerate(zip(t, V, sigma_V)):
    table.add_row(str(i+1), f"{ti:.1f}", f"{Vi:.2f}", f"{si:.3f}")

console.print(table)

# ==============================================================================
# MODELO EXPONENCIAL
# ==============================================================================

def modelo_exponencial(t, V0, tau):
    """
    Modelo de decaimiento exponencial para circuito RL.
    
    V(t) = V₀·e^(-τ·t)
    
    Parámetros:
    -----------
    t : float o array
        Tiempo
    V0 : float
        Voltaje inicial (en t=0)
    tau : float
        Constante de decaimiento τ = R/L
    
    Retorna:
    --------
    V : float o array
        Voltaje en el tiempo t
    """
    return V0 * np.exp(-tau * t)


# ==============================================================================
# MÉTODO 1: AJUSTE LINEAL (Linearización ln(V) vs t)
# ==============================================================================

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan]MÉTODO 1: AJUSTE LINEAL (Linearización)[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

# Linearizar: ln(V) = ln(V₀) - τ·t
ln_V = np.log(V)
sigma_ln_V = sigma_V / V  # Propagación de error

# Ajuste lineal: y = a + b·x
# donde: y = ln(V), x = t, a = ln(V₀), b = -τ
coef = np.polyfit(t, ln_V, deg=1, w=1/sigma_ln_V)
b_linear = coef[0]  # Pendiente = -τ
a_linear = coef[1]  # Intercepto = ln(V₀)

# Extraer parámetros
tau_linear = -b_linear
V0_linear = np.exp(a_linear)

console.print(f"\n[green]Resultados del ajuste lineal:[/green]")
console.print(f"  V₀ = {V0_linear:.4f} V")
console.print(f"  τ  = {tau_linear:.4f} ms⁻¹")
console.print(f"  ln(V₀) = {a_linear:.4f}")

# Calcular R² para el ajuste lineal
V_ajuste_linear = modelo_exponencial(t, V0_linear, tau_linear)
residuos_linear = V - V_ajuste_linear
SS_res = np.sum(residuos_linear**2)
SS_tot = np.sum((V - np.mean(V))**2)
R2_linear = 1 - (SS_res / SS_tot)

console.print(f"  R² = {R2_linear:.6f}")

# ==============================================================================
# MÉTODO 2: AJUSTE NO LINEAL CON CURVE_FIT (Más preciso)
# ==============================================================================

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan]MÉTODO 2: AJUSTE NO LINEAL (curve_fit)[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

# Valores iniciales (usar resultados del ajuste lineal)
p0 = [V0_linear, tau_linear]

# Ajuste con curve_fit
popt, pcov = curve_fit(
    modelo_exponencial,
    t,
    V,
    p0=p0,
    sigma=sigma_V,
    absolute_sigma=True
)

# Extraer parámetros óptimos
V0_fit = popt[0]
tau_fit = popt[1]

# Extraer incertidumbres (diagonal de la matriz de covarianza)
errores = np.sqrt(np.diag(pcov))
sigma_V0 = errores[0]
sigma_tau = errores[1]

console.print(f"\n[green]✓ Ajuste exitoso[/green]")
console.print(f"\n[bold green]Parámetros ajustados:[/bold green]")
console.print(f"  V₀ = {V0_fit:.4f} ± {sigma_V0:.4f} V")
console.print(f"  τ  = {tau_fit:.4f} ± {sigma_tau:.4f} ms⁻¹")

# Incertidumbre relativa
incert_rel_V0 = 100 * sigma_V0 / V0_fit
incert_rel_tau = 100 * sigma_tau / tau_fit

console.print(f"\n[yellow]Incertidumbres relativas:[/yellow]")
console.print(f"  ΔV₀/V₀ = {incert_rel_V0:.2f}%")
console.print(f"  Δτ/τ   = {incert_rel_tau:.2f}%")

# Matriz de correlación
corr_matrix = pcov / np.sqrt(np.outer(np.diag(pcov), np.diag(pcov)))

console.print(f"\n[yellow]Matriz de correlación:[/yellow]")
console.print(f"  Corr(V₀, τ) = {corr_matrix[0,1]:.4f}")

# ==============================================================================
# (b) CÁLCULO DE CHI-CUADRADO (χ²)
# ==============================================================================

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan](b) BONDAD DEL AJUSTE: χ²[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

# Calcular valores ajustados
V_ajustado = modelo_exponencial(t, V0_fit, tau_fit)

# Residuos
residuos = V - V_ajustado
residuos_normalizados = residuos / sigma_V

# Chi-cuadrado
chi2 = np.sum(residuos_normalizados**2)

# Grados de libertad: n_datos - n_parámetros
n_datos = len(t)
n_parametros = 2  # V₀ y τ
grados_libertad = n_datos - n_parametros

# Chi-cuadrado reducido
chi2_reducido = chi2 / grados_libertad

console.print(f"\n[bold green]Resultados:[/bold green]")
console.print(f"  χ² = {chi2:.4f}")
console.print(f"  Grados de libertad = {grados_libertad}")
console.print(f"  χ²_reducido = {chi2_reducido:.4f}")

# Interpretación
console.print(f"\n[bold yellow]Interpretación:[/bold yellow]")
if chi2_reducido < 0.5:
    interpretacion = "[yellow]⚠ χ²_red < 0.5: Posible sobreestimación de errores[/yellow]"
elif 0.5 <= chi2_reducido <= 2.0:
    interpretacion = "[green]✓ 0.5 ≤ χ²_red ≤ 2.0: Ajuste excelente[/green]"
elif 2.0 < chi2_reducido <= 5.0:
    interpretacion = "[yellow]⚠ 2.0 < χ²_red ≤ 5.0: Ajuste aceptable, revisar errores[/yellow]"
else:
    interpretacion = "[red]✗ χ²_red > 5.0: Ajuste pobre, modelo incorrecto o errores subestimados[/red]"

console.print(f"  {interpretacion}")

# ¿Tiene sentido?
console.print(f"\n[bold cyan]¿Tiene sentido?[/bold cyan]")
if 0.5 <= chi2_reducido <= 2.0:
    console.print("  [green]✓ SÍ. El modelo exponencial describe bien los datos.[/green]")
    console.print("  [green]  Los residuos son consistentes con las incertidumbres.[/green]")
else:
    console.print("  [yellow]⚠ REVISAR. El ajuste puede mejorarse.[/yellow]")

# ==============================================================================
# INFORMACIÓN FÍSICA
# ==============================================================================

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan]INTERPRETACIÓN FÍSICA[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

# Constante de tiempo: T = 1/τ
T_constante = 1.0 / tau_fit
sigma_T = sigma_tau / (tau_fit**2)  # Propagación de error

console.print(f"\n[yellow]Constante de tiempo:[/yellow]")
console.print(f"  T = 1/τ = {T_constante:.4f} ± {sigma_T:.4f} ms")
console.print(f"  (tiempo para que V decaiga a V₀/e = {V0_fit/np.e:.3f} V)")

# Vida media: t_1/2 = ln(2)/τ
t_media = np.log(2) / tau_fit
sigma_t_media = sigma_tau * np.log(2) / (tau_fit**2)

console.print(f"\n[yellow]Vida media:[/yellow]")
console.print(f"  t_1/2 = ln(2)/τ = {t_media:.4f} ± {sigma_t_media:.4f} ms")
console.print(f"  (tiempo para que V decaiga a V₀/2 = {V0_fit/2:.3f} V)")

# Si conocemos R o L, podemos calcular el otro
console.print(f"\n[yellow]Relación con componentes:[/yellow]")
console.print(f"  τ = R/L")
console.print(f"  Si R = 100 Ω → L = R/τ = {100/tau_fit:.2f} mH")
console.print(f"  Si L = 10 mH → R = τ·L = {tau_fit*10:.2f} Ω")

# ==============================================================================
# (c) GRÁFICAS
# ==============================================================================

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan](c) GENERANDO GRÁFICAS[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

# Crear rango suave para la curva ajustada
t_suave = np.linspace(0, t[-1], 1000)
V_suave = modelo_exponencial(t_suave, V0_fit, tau_fit)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ==============================================================================
# Subplot 1: Escala lineal
# ==============================================================================

ax1 = fig.add_subplot(gs[0, 0])

# Datos con barras de error
ax1.errorbar(t, V, yerr=sigma_V, fmt='o', color='red', markersize=8,
            capsize=5, elinewidth=2, capthick=2,
            label='Datos experimentales', zorder=5)

# Curva ajustada
ax1.plot(t_suave, V_suave, '-', color='blue', linewidth=2.5,
        label=f'Ajuste: V(t) = {V0_fit:.3f}·e^(-{tau_fit:.3f}·t)', alpha=0.8)

# Valores especiales
ax1.axhline(V0_fit/np.e, color='green', linestyle='--', linewidth=1.5,
           label=f'V₀/e = {V0_fit/np.e:.3f} V', alpha=0.6)
ax1.axvline(T_constante, color='green', linestyle='--', linewidth=1.5,
           label=f'T = {T_constante:.3f} ms', alpha=0.6)

ax1.set_xlabel('Tiempo t (ms)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Voltaje V(t) (V)', fontsize=13, fontweight='bold')
ax1.set_title('Decaimiento Exponencial - Escala Lineal', fontsize=15, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)

# ==============================================================================
# Subplot 2: Escala semi-log (REQUERIDO)
# ==============================================================================

ax2 = fig.add_subplot(gs[0, 1])

# Datos con barras de error
ax2.errorbar(t, V, yerr=sigma_V, fmt='o', color='red', markersize=8,
            capsize=5, elinewidth=2, capthick=2,
            label='Datos experimentales', zorder=5)

# Curva ajustada
ax2.plot(t_suave, V_suave, '-', color='blue', linewidth=2.5,
        label=f'Ajuste: V₀={V0_fit:.3f} V, τ={tau_fit:.3f} ms⁻¹', alpha=0.8)

# Escala semi-log
ax2.set_yscale('log')

# Línea recta en escala log (para verificar)
ax2.plot(t_suave, V0_fit * np.exp(-tau_fit * t_suave), '--', 
        color='cyan', linewidth=2, label='ln(V) = ln(V₀) - τ·t', alpha=0.5)

ax2.set_xlabel('Tiempo t (ms)', fontsize=13, fontweight='bold')
ax2.set_ylabel('log(Voltaje V(t)) (V)', fontsize=13, fontweight='bold')
ax2.set_title('Gráfica Semi-Log (Requerida)', fontsize=15, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim(left=0)

# ==============================================================================
# Subplot 3: Residuos
# ==============================================================================

ax3 = fig.add_subplot(gs[1, 0])

# Residuos normalizados
ax3.errorbar(t, residuos_normalizados, yerr=1.0, fmt='o', color='purple',
            markersize=8, capsize=5, elinewidth=2, capthick=2)

# Líneas de referencia
ax3.axhline(0, color='blue', linestyle='-', linewidth=2, alpha=0.7)
ax3.axhline(2, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='±2σ')
ax3.axhline(-2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax3.axhline(3, color='red', linestyle=':', linewidth=1, alpha=0.5, label='±3σ')
ax3.axhline(-3, color='red', linestyle=':', linewidth=1, alpha=0.5)

ax3.set_xlabel('Tiempo t (ms)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Residuos Normalizados', fontsize=13, fontweight='bold')
ax3.set_title(f'Análisis de Residuos (χ²_red = {chi2_reducido:.3f})', 
             fontsize=15, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# ==============================================================================
# Subplot 4: Linearización ln(V) vs t
# ==============================================================================

ax4 = fig.add_subplot(gs[1, 1])

# Datos linearizados
ax4.errorbar(t, ln_V, yerr=sigma_ln_V, fmt='o', color='orange', markersize=8,
            capsize=5, elinewidth=2, capthick=2,
            label='ln(V) datos', zorder=5)

# Ajuste lineal
t_lin = np.linspace(0, t[-1], 100)
ln_V_lin = a_linear + b_linear * t_lin
ax4.plot(t_lin, ln_V_lin, '-', color='green', linewidth=2.5,
        label=f'Ajuste: ln(V) = {a_linear:.3f} - {-b_linear:.3f}·t', alpha=0.8)

ax4.set_xlabel('Tiempo t (ms)', fontsize=13, fontweight='bold')
ax4.set_ylabel('ln(Voltaje)', fontsize=13, fontweight='bold')
ax4.set_title('Linearización: ln(V) vs t', fontsize=15, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.suptitle('Análisis Completo del Circuito RL', fontsize=18, fontweight='bold', y=0.995)
plt.show()

console.print("[green]✓ Gráficas generadas[/green]")

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================

console.print("\n")
console.rule("[bold green]RESUMEN DE RESULTADOS[/bold green]")

resumen = f"""[bold cyan](a) Parámetros ajustados con incertidumbres:[/bold cyan]

    V₀ = {V0_fit:.4f} ± {sigma_V0:.4f} V  ({incert_rel_V0:.2f}%)
    τ  = {tau_fit:.4f} ± {sigma_tau:.4f} ms⁻¹  ({incert_rel_tau:.2f}%)

[bold cyan](b) Bondad del ajuste:[/bold cyan]

    χ² = {chi2:.4f}
    χ²_reducido = {chi2_reducido:.4f}
    
    {interpretacion.replace('[green]', '').replace('[/green]', '').replace('[yellow]', '').replace('[/yellow]', '').replace('[red]', '').replace('[/red]', '')}

[bold cyan](c) Gráfica semi-log:[/bold cyan]

    ✓ Generada en el subplot superior derecho
    ✓ En escala semi-log, el decaimiento exponencial es una línea recta
    ✓ Pendiente = -τ = {-tau_fit:.4f} ms⁻¹

[bold cyan]Interpretación física:[/bold cyan]

    • Constante de tiempo: T = {T_constante:.4f} ms
    • Vida media: t_1/2 = {t_media:.4f} ms
    • Relación: τ = R/L
"""

panel = Panel(
    resumen,
    title="[bold]Resultados del Ajuste de Circuito RL[/bold]",
    border_style="green",
    box=box.DOUBLE
)

console.print(panel)

console.print("\n[bold green]✓ Análisis completado exitosamente[/bold green]")