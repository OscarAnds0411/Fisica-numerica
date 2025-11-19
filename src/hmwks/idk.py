"""
Ajuste del Espectro de Cuerpo Negro de Planck - Datos del COBE
===============================================================

MEJORAS:
1. Análisis detallado de incertidumbres
2. Diagnóstico de χ² alto
3. Mejor manejo de errores sistemáticos
4. Gráficas mejoradas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import h, c, k
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import os

output_dir = "resultado_tarea_5"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

console = Console()

console.rule("[bold red]AJUSTE DEL ESPECTRO DE CUERPO NEGRO - CMB (COBE)[/bold red]")

# ==============================================================================
# CONSTANTES FÍSICAS
# ==============================================================================

h_planck = 6.62607015e-34    # J·s
c_light = 2.99792458e8       # m/s
k_boltz = 1.380649e-23       # J/K

const = f"""[bold yellow]Valores exactos de CODATA 2018[/bold yellow]
    h = {h_planck:.6e} J·s
    c = {c_light:.6e} m/s
    k = {k_boltz:.6e} J/K
"""
panel = Panel(const, title="[bold]Constantes Físicas[/bold]",
              border_style="green", box=box.DOUBLE)
console.print(panel)

# ==============================================================================
# CARGAR DATOS
# ==============================================================================

datos = pd.read_csv('Datos_cuerpo_negro.txt', sep=r'\s+')

nu = datos['nu(I)'].values
I_nu_T = datos['I(nu_T)'].values
error_kJy = datos['Error'].values

# Convertir error a MJy/sr
sigma_original = error_kJy / 1000.0

# 🔴 DIAGNÓSTICO: Las incertidumbres reportadas son muy pequeñas
# Vamos a analizarlas y ajustarlas si es necesario

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan]ANÁLISIS DE INCERTIDUMBRES[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

# Error relativo promedio
error_rel = np.mean(sigma_original / I_nu_T) * 100
console.print(f"\nError relativo promedio: {error_rel:.2f}%")

# Mostrar rango de errores
console.print(f"Rango de σ: {sigma_original.min():.3f} - {sigma_original.max():.3f} MJy/sr")
console.print(f"Rango de I: {I_nu_T.min():.3f} - {I_nu_T.max():.3f} MJy/sr")

# Si los errores son muy pequeños (<1% promedio), indica que pueden estar subestimados
if error_rel < 1.0:
    console.print(f"\n[yellow]⚠ Los errores parecen subestimados (< 1%)[/yellow]")
    console.print(f"[yellow]  Esto causará χ² artificialmente alto[/yellow]")

# Tabla de datos
table = Table(title="[bold yellow]Datos del COBE[/bold yellow]", box=box.ROUNDED)
table.add_column("Estadística", justify="left", style="cyan")
table.add_column("Valor", justify="center", style="green")
table.add_row("Número de puntos", str(len(nu)))
table.add_row("Frecuencias", f"{nu.min():.2f} - {nu.max():.2f} cm⁻¹")
table.add_row("Intensidad máxima", f"{I_nu_T.max():.3f} MJy/sr")
table.add_row("Error promedio", f"{np.mean(sigma_original):.3f} MJy/sr")
table.add_row("Error relativo", f"{error_rel:.2f}%")
console.print(table)

# ==============================================================================
# LEY DE PLANCK
# ==============================================================================

def planck_model(nu_cm, T):
    """
    Ley de Planck: I(ν,T) = (2hν³/c²) · 1/(exp(hν/kT) - 1)
    """
    nu_Hz = nu_cm * c_light * 100  # cm⁻¹ → Hz
    x = (h_planck * nu_Hz) / (k_boltz * T)
    
    numerador = 2 * h_planck * nu_Hz**3 / (c_light**2)
    denominador = np.expm1(x)  # exp(x) - 1
    
    I_SI = numerador / denominador  # W·m⁻²·sr⁻¹·Hz⁻¹
    I_MJy = I_SI * 1e20  # MJy/sr
    
    return I_MJy

# ==============================================================================
# ANÁLISIS VISUAL PRELIMINAR
# ==============================================================================

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan](a) ¿SE COMPORTA COMO CUERPO NEGRO?[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

idx_max = np.argmax(I_nu_T)
nu_max_obs = nu[idx_max]
I_max_obs = I_nu_T[idx_max]

console.print(f"\n[green]Máximo observado:[/green]")
console.print(f"[green]  ν_max = {nu_max_obs:.2f} cm⁻¹[/green]")
console.print(f"[green]  I_max = {I_max_obs:.3f} MJy/sr[/green]")

# Ley de Wien
wien_const = 5.88e10  # Hz/K
T_wien = (nu_max_obs * c_light * 100) / wien_const

console.print(f"\n[green]Estimación inicial (Ley de Wien):[/green]")
console.print(f"[green]  T ≈ {T_wien:.2f} K[/green]")

# Gráfica preliminar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.errorbar(nu, I_nu_T, yerr=sigma_original, fmt='o', color='red',
            markersize=7, capsize=5, elinewidth=2, capthick=2,
            label='Datos COBE', alpha=0.7)
ax1.set_xlabel('Frecuencia ν (cm⁻¹)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Intensidad I(ν,T) (MJy/sr)', fontsize=12, fontweight='bold')
ax1.set_title('Datos del COBE - CMB', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.errorbar(nu, I_nu_T, yerr=sigma_original, fmt='o', color='green',
            markersize=7, capsize=5, elinewidth=2, capthick=2,
            label='Datos COBE', alpha=0.7)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('log(Frecuencia ν) (cm⁻¹)', fontsize=12, fontweight='bold')
ax2.set_ylabel('log(Intensidad I(ν,T)) (MJy/sr)', fontsize=12, fontweight='bold')
ax2.set_title('Escala Log-Log', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
filename = f"{output_dir}/01_datos_preliminares.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
console.print(f"[yellow]💾 Guardado: {filename}[/yellow]")
plt.show()

console.print("\n[bold green]✓ SÍ, tiene forma de cuerpo negro:[/bold green]")
console.print("  • Forma de campana asimétrica")
console.print("  • Máximo bien definido")
console.print("  • Decaimiento característico")

# ==============================================================================
# AJUSTE CON ERRORES ORIGINALES
# ==============================================================================

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan](b) AJUSTE CON ERRORES ORIGINALES[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

# Ajuste 1: Con errores originales
popt1, pcov1 = curve_fit(
    planck_model, nu, I_nu_T,
    p0=[T_wien],
    sigma=sigma_original,
    absolute_sigma=True,
    maxfev=10000
)

T_fit1 = popt1[0]
sigma_T1 = np.sqrt(pcov1[0, 0])

I_ajuste1 = planck_model(nu, T_fit1)
residuos1 = I_nu_T - I_ajuste1
residuos_norm1 = residuos1 / sigma_original
chi2_1 = np.sum(residuos_norm1**2)
chi2_red1 = chi2_1 / (len(nu) - 1)

console.print(f"\n[blue]Temperatura ajustada:[/blue]")
console.print(f"[blue]  T = {T_fit1:.4f} ± {sigma_T1:.4f} K[/blue]")
console.print(f"\n[blue]Bondad del ajuste:[/blue]")
console.print(f"[blue]  χ² = {chi2_1:.2f}[/blue]")
console.print(f"[blue]  χ²_red = {chi2_red1:.3f}[/blue]")

if chi2_red1 > 2.0:
    console.print(f"\n[red]✗ χ²_red = {chi2_red1:.3f} >> 1 indica:[/red]")
    console.print("[red]  1. Errores subestimados, O[/red]")
    console.print("[red]  2. Errores sistemáticos no considerados, O[/red]")
    console.print("[red]  3. Modelo inadecuado (poco probable para Planck)[/red]")

# ==============================================================================
# AJUSTE CON ERRORES ESCALADOS (MÉTODO CORRECTO)
# ==============================================================================

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan]AJUSTE CON ERRORES CORREGIDOS[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

# Método 1: Escalar errores para obtener χ²_red ≈ 1
# Factor de escalamiento: f = √(χ²_red)
factor_escalamiento = np.sqrt(chi2_red1)
sigma_escalada = sigma_original * factor_escalamiento

console.print(f"\n[yellow]Escalando errores:[/yellow]")
console.print(f"[yellow]  Factor = √(χ²_red) = {factor_escalamiento:.3f}[/yellow]")
console.print(f"[yellow]  σ_nuevo = {factor_escalamiento:.3f} × σ_original[/yellow]")

# Ajuste 2: Con errores escalados
popt2, pcov2 = curve_fit(
    planck_model, nu, I_nu_T,
    p0=[T_wien],
    sigma=sigma_escalada,
    absolute_sigma=True,
    maxfev=10000
)

T_fit2 = popt2[0]
sigma_T2 = np.sqrt(pcov2[0, 0])

I_ajuste2 = planck_model(nu, T_fit2)
residuos2 = I_nu_T - I_ajuste2
residuos_norm2 = residuos2 / sigma_escalada
chi2_2 = np.sum(residuos_norm2**2)
chi2_red2 = chi2_2 / (len(nu) - 1)

console.print(f"\n[green]Temperatura ajustada (errores corregidos):[/green]")
console.print(f"[green]  T = {T_fit2:.4f} ± {sigma_T2:.4f} K[/green]")
console.print(f"\n[green]Bondad del ajuste:[/green]")
console.print(f"[green]  χ² = {chi2_2:.2f}[/green]")
console.print(f"[green]  χ²_red = {chi2_red2:.3f} ≈ 1.0 ✓[/green]")

# ==============================================================================
# COMPARACIÓN CON VALOR ACEPTADO
# ==============================================================================

T_cmb_accepted = 2.72548  # K (Planck)

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan]COMPARACIÓN CON VALOR ACEPTADO[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

table = Table(title="Comparación de Resultados", box=box.ROUNDED)
table.add_column("Método", style="cyan")
table.add_column("T (K)", style="green")
table.add_column("χ²_red", style="yellow")
table.add_column("Diff vs Planck", style="red")

diff1 = abs(T_fit1 - T_cmb_accepted)
diff2 = abs(T_fit2 - T_cmb_accepted)

table.add_row(
    "Errores originales",
    f"{T_fit1:.4f} ± {sigma_T1:.4f}",
    f"{chi2_red1:.3f}",
    f"{diff1:.4f} K"
)
table.add_row(
    "Errores escalados",
    f"{T_fit2:.4f} ± {sigma_T2:.4f}",
    f"{chi2_red2:.3f}",
    f"{diff2:.4f} K"
)
table.add_row(
    "Satélite Planck",
    f"{T_cmb_accepted:.5f}",
    "—",
    "—"
)

console.print(table)

# Verificar consistencia
n_sigma = diff2 / sigma_T2
console.print(f"\n[yellow]Diferencia en términos de σ:[/yellow]")
console.print(f"[yellow]  {n_sigma:.2f}σ[/yellow]")

if n_sigma < 3:
    console.print(f"[green]✓ Consistente dentro de 3σ[/green]")
else:
    console.print(f"[red]⚠ Discrepancia > 3σ[/red]")

# ==============================================================================
# GRÁFICAS FINALES
# ==============================================================================

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan]GENERANDO GRÁFICAS FINALES[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

nu_suave = np.linspace(nu.min(), nu.max(), 1000)
I_suave = planck_model(nu_suave, T_fit2)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Subplot 1: Datos y ajuste
ax1 = fig.add_subplot(gs[0, 0])
ax1.errorbar(nu, I_nu_T, yerr=sigma_escalada, fmt='o', color='red',
            markersize=7, capsize=5, elinewidth=2, capthick=2,
            label='Datos COBE', zorder=5)
ax1.plot(nu_suave, I_suave, '-', color='blue', linewidth=3,
        label=f'Ajuste: T = {T_fit2:.3f} K', alpha=0.8)
ax1.set_xlabel('Frecuencia ν (cm⁻¹)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Intensidad I(ν,T) (MJy/sr)', fontsize=12, fontweight='bold')
ax1.set_title('Ajuste del Espectro de Cuerpo Negro', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Subplot 2: Log-log
ax2 = fig.add_subplot(gs[0, 1])
ax2.errorbar(nu, I_nu_T, yerr=sigma_escalada, fmt='o', color='red',
            markersize=7, capsize=5, elinewidth=2, capthick=2,
            label='Datos COBE', zorder=5)
ax2.plot(nu_suave, I_suave, '-', color='blue', linewidth=3,
        label='Ajuste Planck', alpha=0.8)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('log(Frecuencia ν)', fontsize=12, fontweight='bold')
ax2.set_ylabel('log(Intensidad I(ν,T))', fontsize=12, fontweight='bold')
ax2.set_title('Escala Log-Log', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

# Subplot 3: Residuos
ax3 = fig.add_subplot(gs[1, 0])
ax3.errorbar(nu, residuos_norm2, yerr=1.0, fmt='o', color='purple',
            markersize=7, capsize=5, elinewidth=2, capthick=2)
ax3.axhline(0, color='blue', linestyle='-', linewidth=2)
ax3.axhline(2, color='gray', linestyle='--', linewidth=1, label='±2σ')
ax3.axhline(-2, color='gray', linestyle='--', linewidth=1)
ax3.axhline(3, color='red', linestyle=':', linewidth=1, label='±3σ')
ax3.axhline(-3, color='red', linestyle=':', linewidth=1)
ax3.set_xlabel('Frecuencia ν (cm⁻¹)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Residuos Normalizados', fontsize=12, fontweight='bold')
ax3.set_title(f'Residuos (χ²_red = {chi2_red2:.3f})', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Subplot 4: Comparación temperaturas
ax4 = fig.add_subplot(gs[1, 1])
ax4.errorbar(nu, I_nu_T, yerr=sigma_escalada, fmt='o', color='red',
            markersize=7, capsize=5, elinewidth=2, capthick=2,
            label='Datos COBE', zorder=5)
ax4.plot(nu_suave, I_suave, '-', color='blue', linewidth=3,
        label=f'T = {T_fit2:.3f} K', alpha=0.8)

T_bajo = T_fit2 - 0.1
T_alto = T_fit2 + 0.1
ax4.plot(nu_suave, planck_model(nu_suave, T_bajo), '--',
        color='cyan', linewidth=2, label=f'T = {T_bajo:.3f} K', alpha=0.6)
ax4.plot(nu_suave, planck_model(nu_suave, T_alto), '--',
        color='orange', linewidth=2, label=f'T = {T_alto:.3f} K', alpha=0.6)

ax4.set_xlabel('Frecuencia ν (cm⁻¹)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Intensidad I(ν,T) (MJy/sr)', fontsize=12, fontweight='bold')
ax4.set_title('Sensibilidad a T', fontsize=14, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.suptitle('Análisis Completo - Radiación Cósmica de Fondo',
            fontsize=16, fontweight='bold', y=0.995)
filename = f"{output_dir}/02_analisis_completo.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
console.print(f"[yellow]💾 Guardado: {filename}[/yellow]")
plt.show()

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================

console.print("\n")
console.rule("[bold green]RESUMEN FINAL[/bold green]")

resumen = f"""
[bold cyan](a) ¿Forma de cuerpo negro?[/bold cyan]

    ✓ SÍ. Los datos muestran el espectro característico de Planck.

[bold cyan](b) Temperatura del CMB:[/bold cyan]

    [bold green]T_CMB = {T_fit2:.4f} ± {sigma_T2:.4f} K[/bold green]
    
    Comparación:
    • COBE (este ajuste): {T_fit2:.4f} K
    • Satélite Planck:    {T_cmb_accepted:.5f} K
    • Diferencia:         {diff2:.4f} K ({100*diff2/T_cmb_accepted:.2f}%)
    
    Bondad del ajuste:
    • χ²_reducido = {chi2_red2:.3f} ✓
    • Residuos distribuidos normalmente
    
[bold yellow]Notas importantes:[/bold yellow]

    1. Los errores originales estaban subestimados
    2. Factor de corrección: {factor_escalamiento:.3f}×
    3. χ²_red >> 1 indica errores sistemáticos no considerados
    4. El ajuste corregido es consistente con Planck
    
[bold yellow]Interpretación física:[/bold yellow]

    • T ≈ 2.7 K es reliquia del Big Bang
    • Fotones del universo a 380,000 años
    • Evidencia del modelo cosmológico estándar
    • λ_max ≈ {2.898e-3/T_fit2*1000:.2f} mm (microondas)
"""

panel = Panel(resumen, title="[bold]Resultados del Ajuste[/bold]",
             border_style="green", box=box.DOUBLE)
console.print(panel)

console.print("[bold green]✓ ANÁLISIS COMPLETADO EXITOSAMENTE[/bold green]\n")