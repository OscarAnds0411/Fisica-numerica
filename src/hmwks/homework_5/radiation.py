"""
Ajuste del Espectro de Cuerpo Negro de Planck - Datos del COBE

Ley de Planck:
I(ν,T) = (2hν³/c²) · 1/(exp(hν/kT) - 1)

donde:
  - I(ν,T) = Intensidad espectral (MJy/sr)
  - ν = Frecuencia (1/cm = cm⁻¹)
  - T = Temperatura (K)
  - h = Constante de Planck = 6.62607015×10⁻³⁴ J·s
  - c = Velocidad de la luz = 2.99792458×10⁸ m/s
  - k = Constante de Boltzmann = 1.380649×10⁻²³ J/K

Objetivos:
(a) Graficar datos del COBE y verificar forma de cuerpo negro
(b) Estimar temperatura T de la radiación cósmica de fondo (CMB)
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

output_dir = "resultados_tarea_5"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
console = Console()

console.rule("[bold red]AJUSTE DEL ESPECTRO DE CUERPO NEGRO - RADIACIÓN CÓSMICA DE FONDO (COBE)")
# Constantes fundamentales (valores exactos de CODATA 2018)

h_planck = 6.62607015e-34    # J·s (Constante de Planck)
c_light = 2.99792458e8       # m/s (Velocidad de la luz)
k_boltz = 1.380649e-23       # J/K (Constante de Boltzmann)

const = f"""[bold yellow] Valores exactos de CODATA 2018 [/bold yellow]
    h = {h_planck:.6e} J·s
    c = {c_light:.6e} m/s
    k = {k_boltz:.6e} J/K
"""
panel = Panel(
    const,
    title="[bold]Constantes a usar[/bold]",
    border_style="green",
    box=box.DOUBLE
)
console.print(panel)

# Leer datos del archivo
datos = pd.read_csv('Datos_cuerpo_negro.txt', sep=r'\s+')

# Extraer columnas
nu = datos['nu(I)'].values           # Frecuencia (1/cm = cm⁻¹)
I_nu_T = datos['I(nu_T)'].values   # Intensidad (MJy/sr)
error = datos['Error'].values     # Error (kJy/sr)

# Convertir error a MJy/sr
sigma = error / 1000.0
#error relativo
error_rel = np.mean(sigma / I_nu_T) * 100
# Mostrar rango de errores
console.print(f"Rango de σ: {sigma.min():.3f} - {sigma.max():.3f} MJy/sr")
console.print(f"Rango de I: {I_nu_T.min():.3f} - {I_nu_T.max():.3f} MJy/sr")

# Si los errores son muy pequeños (<1% promedio), indica que pueden estar subestimados
if error_rel < 1.0:
    console.print(f"\n[yellow] !!!! Los errores parecen subestimados (< 1%)[/yellow]")
    console.print(f"[yellow]  Esto causará χ² artificialmente alto D:[/yellow]")


#mostramos los datos
table = Table(title="[bold yellow]Datos del COBE cargados[/bold yellow]", box=box.ROUNDED)
table.add_column("Número de puntos", justify="center", style="cyan")
table.add_column("Rango de frecuencias [cm⁻¹]", justify="center", style="cyan")
table.add_column("Intensidad máxima [MJy/sr]", justify="center", style="cyan")
table.add_row(str(len(nu)), f"{nu.min():.2f} - {nu.max():.2f}", f"{I_nu_T.max():.3f}")
console.print(table)

# Mostrar tabla de datos
console.print("Nuestra tabla es la siguiente:\n", style="blink on blue")
table = Table(title="[bold cyan] Datos recuperados con panda de COBE.txt[/bold cyan]", box=box.SQUARE)
table.add_column("i", justify = "right", style = "cyan")
table.add_column("ν (cm⁻¹)", justify = "right", style = "magenta")
table.add_column("I(ν,T) (MJy/sr)", justify = "right", style = "blue")
table.add_column("σ (MJy/sr)", justify = "right", style = "yellow")
for i in range(len(nu)):
    table.add_row(f"{i+1}", f"{nu[i]:.2f}", f"{I_nu_T[i]:.3f}", f"{sigma[i]:.3f}")
console.print(table)

# definimos el modelo que nos ayudara mas adelante

def planck_model(nu_cm, T):
    """
    Ley de Planck para la intensidad espectral.
    
    I(ν,T) = (2hν³/c²) · 1/(exp(hν/kT) - 1)
    
    Parámetros:
    -----------
    nu_cm : float o array
        Frecuencia en cm⁻¹ (número de onda)
    T : float
        Temperatura en Kelvin
    
    Retorna:
    --------
    I : float o array
        Intensidad en MJy/sr
    
    Notas:
    ------
    Conversión de unidades:
    - ν (cm⁻¹) → ν (Hz): ν_Hz = ν_cm × c × 100
    - I (W·m⁻²·sr⁻¹·Hz⁻¹) → I (MJy/sr): × 10²⁶
    """
    
    # Convertir frecuencia de cm⁻¹ a Hz
    nu_Hz = nu_cm * c_light * 100  # cm⁻¹ → Hz
    
    # Calcular exponente (dimensionless)
    x = (h_planck * nu_Hz) / (k_boltz * T)
    
    # Evitar overflow en el exponencial
    # Para x muy grande, exp(x) - 1 ≈ exp(x)
    # Usamos np.expm1(x) = exp(x) - 1 que es más precisa
    
    # Ley de Planck en W·m⁻²·sr⁻¹·Hz⁻¹
    numerador = 2 * h_planck * nu_Hz**3 / (c_light**2)
    denominador = np.expm1(x)  # exp(x) - 1
    
    I_SI = numerador / denominador  # W·m⁻²·sr⁻¹·Hz⁻¹
    
    # Convertir a MJy/sr
    # 1 Jy = 10⁻²⁶ W·m⁻²·Hz⁻¹
    # 1 MJy = 10⁻²⁰ W·m⁻²·Hz⁻¹
    I_MJy = I_SI * 1e20  # MJy/sr
    
    return I_MJy

console.rule("[bold green] ¿Se comporta como un cuerpo negro? [/bold green]")

# Encontrar el máximo
idx_max = np.argmax(I_nu_T)
nu_max_obs = nu[idx_max]
I_max_obs = I_nu_T[idx_max]

console.print(f"[green]\nMáximo observado:")
console.print(f"[green]  ν_max = {nu_max_obs:.2f} cm⁻¹")
console.print(f"[green]  I_max = {I_max_obs:.3f} MJy/sr")

# Ley de Wien: ν_max = 5.88×10¹⁰ × T (Hz/K)
# En cm⁻¹: ν_max (cm⁻¹) = (5.88×10¹⁰ × T) / (c × 100)
# # T_estimado ≈ ν_max (cm⁻¹) × c × 100 / (5.88×10¹⁰)
wien_const = 5.88e10  # Hz/K
T_wien_estimate = (nu_max_obs * c_light * 100) / wien_const

console.print(f"[green]\nEstimación inicial usando Ley de Wien:")
console.print(f"[green]  T_wien ≈ {T_wien_estimate:.2f} K")

# Gráfica preliminar
plt.scatter(nu,I_nu_T, color='r', label='Datos observados')
plt.xlabel('Frecuencia ν (cm⁻¹)', fontsize=13, fontweight='bold')
plt.ylabel('Intensidad I(ν,T) (MJy/sr)', fontsize=13, fontweight='bold')
plt.title('Datos del COBE - Radiación Cósmica de Fondo', 
             fontsize=15, fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
plt.grid(True, alpha=0.3)
filename = f"{output_dir}/cuerpo_negro_dipersión.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
console.print(f"[yellow]Guardado: {filename}")
# Subplot 2: Escala log-log
# ax2.errorbar(nu, I_nu_T, yerr=sigma, fmt='o', color='green', markersize=6,
            # capsize=4, elinewidth=1.5, capthick=1.5,
            # label='Datos COBE', alpha=0.7)
# ax2.set_xscale('log')
# ax2.set_yscale('log')
# ax2.set_xlabel('log(Frecuencia ν) (cm⁻¹)', fontsize=13, fontweight='bold')
# ax2.set_ylabel('log(Intensidad I(ν,T)) (MJy/sr)', fontsize=13, fontweight='bold')
# ax2.set_title('Escala Log-Log', fontsize=15, fontweight='bold')
# ax2.legend(fontsize=11)
# ax2.grid(True, alpha=0.3, which='both')
plt.show()
console.print("[bold red]\nEntonces, ¿tiene forma de cuerpo negro?")
console.print("[bold red]  Sí, los datos medidos por COBE muestran:")
console.print("[bold red]    - Forma de campana asimétrica característica")
console.print("[bold red]    - Máximo bien definido")
console.print("[bold red]    - Decaimiento rápido a bajas y altas frecuencias")
console.print("[bold red]    - Consistente con la ley de Planck")

# Hagamos minimos cuadrados

console.rule("[bold blue]Ajuste de temperatura por mínimos cuadrados[/bold blue]")

# Valor inicial para temperatura (usar estimación de Wien)
T_inicial = T_wien_estimate

console.print(f"[blue]\nValor inicial: T₀ = {T_inicial:.2f} K")

# Ajuste con curve_fit
try:
    popt, pcov = curve_fit(
        planck_model,
        nu,
        I_nu_T,
        p0=[T_inicial],
        sigma=sigma,
        absolute_sigma=True,
        maxfev=10000
    )
    
    # Extraer temperatura ajustada
    T_fit = popt[0]
    
    # Extraer incertidumbre
    sigma_T = np.sqrt(pcov[0, 0])
    
    console.print("[bold]  \nResultado del ajuste")
    console.print(f"[bold blue]\n  T = {T_fit:.4f} ± {sigma_T:.4f} K")
    console.print(f"[bold blue]Incertidumbre relativa: ΔT/T = {100*sigma_T/T_fit:.3f}%")
    
    # Valor aceptado de la temperatura del CMB
    T_cmb_accepted = 2.72548  # K (valor del satélite Planck)
    
    console.print("[bold]  \nComparacion con el valor aceptado (valor del satélite Planck)")
    console.print(f"[bold blue]\n  T_ajustado = {T_fit:.4f} ± {sigma_T:.4f} K")
    console.print(f"[bold blue]  T_aceptado = {T_cmb_accepted:.4f} K (satélite Planck)")
    console.print(f"[bold blue]  Diferencia = {abs(T_fit - T_cmb_accepted):.4f} K")
    console.print(f"[bold blue]  Error relativo = {100*abs(T_fit - T_cmb_accepted)/T_cmb_accepted:.2f}%")
    
    # if abs(T_fit - T_cmb_accepted) < 3*sigma_T:
        # console.print(f"[bold green]\n  Consistente dentro de 3σ")
    # else:
        # console.print(f"[bold red]\n No consistente dentro de 3")
    # Calcular valores ajustados
    I_ajustado = planck_model(nu, T_fit)
    
    # Residuos
    residuos = I_nu_T - I_ajustado
    residuos_normalizados = residuos / sigma
    
    # Chi-cuadrado
    chi2 = np.sum(residuos_normalizados**2)
    grados_libertad = len(nu) - 1  # 1 parámetro (T)
    chi2_reducido = chi2 / grados_libertad
    
    console.print("[bold]\nBondad del ajuste")
    console.print(f"[bold blue]\n  χ² = {chi2:.4f}")
    console.print(f"[bold blue]  Grados de libertad = {grados_libertad}")
    console.print(f"[bold blue]  χ²_reducido = {chi2_reducido:.4f}")
    
    if 0.5 <= chi2_reducido <= 2.0:
        console.print(f"[bold green]\n  :D Excelente ajuste (χ²_red ≈ 1)")
    elif chi2_reducido < 0.5:
        console.print(f"[bold yellow]\n  ! χ²_red < 0.5: Posible sobreestimación de errores")
    elif 2.0 < chi2_reducido <= 5.0:
        console.print(f"[bold yellow]\n  !!!! Ajuste aceptable (revisar errores)")
    else:
        console.print(f"[bold red]\n  D: Ajuste pobre (χ²_red > 5)")
    
    # GRÁFICAS FINALES
    
    console.rule("Ultimos gráficos")
    
    # Crear rango suave para curva
    nu_suave = np.linspace(nu.min(), nu.max(), 1000)
    I_suave = planck_model(nu_suave, T_fit)
    
    # Crear figura con 3 subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Datos y ajuste
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Datos con barras de error
    ax1.errorbar(nu, I_nu_T, yerr=sigma, fmt='o', color='red', markersize=7,
                capsize=5, elinewidth=2, capthick=2,
                label='Datos COBE', zorder=5)
    
    # Curva ajustada
    ax1.plot(nu_suave, I_suave, '-', color='blue', linewidth=3,
            label=f'Ajuste Planck: T = {T_fit:.3f} K', alpha=0.8)
    
    # Máximo teórico
    nu_max_teorico = nu_suave[np.argmax(I_suave)]
    ax1.axvline(nu_max_teorico, color='green', linestyle='--', linewidth=2,
               label=f'ν_max teórico = {nu_max_teorico:.2f} cm⁻¹', alpha=0.6)
    
    ax1.set_xlabel('Frecuencia ν (cm⁻¹)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Intensidad I(ν,T) (MJy/sr)', fontsize=13, fontweight='bold')
    ax1.set_title('Ajuste del Espectro de Cuerpo Negro', 
                 fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Escala log-log
    
    ax2 = fig.add_subplot(gs[0,1])
    
    ax2.errorbar(nu, I_nu_T, yerr=sigma, fmt='o', color='red', markersize=7,
                capsize=5, elinewidth=2, capthick=2,
                label='Datos COBE', zorder=5)
    
    ax2.plot(nu_suave, I_suave, '-', color='blue', linewidth=3,
            label=f'Ajuste Planck', alpha=0.8)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('log(Frecuencia ν) (cm⁻¹)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('log(Intensidad I(ν,T)) (MJy/sr)', fontsize=13, fontweight='bold')
    ax2.set_title('Escala Log-Log', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Subplot 3: Comparación con otras temperaturas
    
    ax4 = fig.add_subplot(gs[0,2])
    
    # Datos
    ax4.errorbar(nu, I_nu_T, yerr=sigma, fmt='o', color='red', markersize=7,
                capsize=5, elinewidth=2, capthick=2,
                label='Datos COBE', zorder=5)
    
    # Ajuste óptimo
    ax4.plot(nu_suave, I_suave, '-', color='blue', linewidth=3,
            label=f'T = {T_fit:.3f} K (ajuste)', alpha=0.8)
    
    # Temperaturas cercanas para comparación
    T_bajo = T_fit - 0.1
    T_alto = T_fit + 0.1
    
    I_bajo = planck_model(nu_suave, T_bajo)
    I_alto = planck_model(nu_suave, T_alto)
    
    ax4.plot(nu_suave, I_bajo, '--', color='cyan', linewidth=2,
            label=f'T = {T_bajo:.3f} K', alpha=0.6)
    ax4.plot(nu_suave, I_alto, '--', color='orange', linewidth=2,
            label=f'T = {T_alto:.3f} K', alpha=0.6)
    
    ax4.set_xlabel('Frecuencia ν (cm⁻¹)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Intensidad I(ν,T) (MJy/sr)', fontsize=13, fontweight='bold')
    ax4.set_title('Sensibilidad a la Temperatura', fontsize=15, fontweight='bold')
    ax4.legend(fontsize=10, loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Análisis Completo del Espectro de Cuerpo Negro - CMB', 
                fontsize=18, fontweight='bold', y=0.995)
    filename = f"{output_dir}/ultimos_graficos_cuerpo_negro.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    
    console.print("[bold green]\n Gráficas finales generadas")
    
    sms= (f"""
(a) ¿Tienen forma de cuerpo negro?
      SÍ. Los datos del COBE muestran la forma característica
      del espectro de Planck para un cuerpo negro.

(b) Temperatura de la radiación cósmica de fondo:
    
    T_CMB = {T_fit:.4f} ± {sigma_T:.4f} K
    
    Comparación:
    * Valor ajustado (COBE): {T_fit:.4f} K
    * Valor aceptado (Planck): {T_cmb_accepted:.4f} K
    * Diferencia: {abs(T_fit - T_cmb_accepted):.4f} K ({100*abs(T_fit - T_cmb_accepted)/T_cmb_accepted:.2f}%)
    
    Bondad del ajuste:
    * χ²_reducido = {chi2_reducido:.4f}
    * {'Excelente ajuste' if 0.5 <= chi2_reducido <= 2.0 else '⚠ Revisar ajuste'}
    
    Interpretación:
    * La temperatura de ~2.7 K es una reliquia del Big Bang
    * Corresponde a fotones del universo temprano (380,000 años)
    * Evidencia fundamental del modelo cosmológico estándar
    """)
    panel = Panel(
    sms,
    title="[bold]Finalmente:\n[/bold]",
    border_style="green",
    box=box.DOUBLE
    )
    console.print(panel)

except Exception as e:
    print(f"\nD: Error en el ajuste: {e}")
    print("\nPosibles soluciones:")
    print("  1. Verificar que los datos estén en el formato correcto")
    print("  2. Ajustar el valor inicial de temperatura")
    print("  3. Revisar las unidades de los datos")