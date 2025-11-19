"""
Cuando una fuente de voltaje se conecta a travÈs de una resistencia y
un inductor en serie, el voltaje a travÈs del inductor Vi (t) obedece la
ecuaciÛn

V (t) = V0e
t

donde t es el tiempo y
= R
L
es el cociente de la resistencia R y la
inductacia L del circuito. Los datos obtenidos de un experimento son:
(a) Encuentre el mejo estimado para los valores de

y V0 y las incer-
tidumbres en sus valores (

y V0
).

(b) Encuentre el valor de 

2 para su ajuste. øTiene sentido?
(c) Realice una gr·Öca semi-log para los datos y el ajuste.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from scipy.optimize import curve_fit
import os

output_dir ="resultados_tarea_5"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
data = pd.read_csv("datos_circuito.txt", header=0, delim_whitespace=True)

ttime = data.iloc[:,0]
voltage = data.iloc[:,1]
uncer = data.iloc[:,2]

# print(np.sum(ttime))

# ttime_np = ttime.to_numpy
# voltage_np = voltage.to_numpy
# uncer_np = uncer.to_numpy
console = Console()

console.rule("[bold blue]Ajuste de Circuito[/bold blue]")

table = Table(title="[bold yellow] Datos experimentales[/bold yellow]",box = box.ROUNDED)
table.add_column("i", justify="center", style="cyan")
table.add_column("Tiempo t (ms)", justify="center", style="blue")
table.add_column("Voltaje V(t) (V)", justify="center", style="green")
table.add_column("σ_V (V)", justify="center", style="red")

for i, (ti, Vi, si) in enumerate(zip(ttime, voltage, uncer)):
    table.add_row(str(i+1), f"{ti:.1f}", f"{Vi:.2f}", f"{si:.3f}")

console.print(table)

console.print("[bold green] Generando grafico de dispersion...")

plt.scatter(ttime , voltage, color = 'r', label = 'Datos del circuito')
plt.title("Datos originales de la respuesta de un circuito RLC")
plt.xlabel("Tiempo (ns)")
plt.ylabel("Voltaje (V)")
plt.grid(True)
plt.legend()
filename = f"{output_dir}/grafico_de_dispersion_RLC.png"
plt.savefig(filename, dpi = 300, bbox_inches="tight")
console.print("[bold green] Grafico guardado: ", filename)
plt.show()
# Para las incertidumbres asociadas:
def S(uncertinity):
    return np.sum(1/(uncertinity**2))

def S_x(uncertinity,xi):
    return np.sum(xi/(uncertinity**2))

def S_y(uncertinity,yi):
    return np.sum(yi/(uncertinity**2))

def S_xx(uncertinity,xi):
    return np.sum((xi**2)/(uncertinity**2))

def S_xy(uncertinity,xi,yi):
    return np.sum((xi*yi)/(uncertinity**2))

def delta(s,s_xx,s_x):
    return s*s_xx-(s_x**2)

def sigma_a1(s_xx,delt):
    """
    Sigma para ordenada al origen (V0)
    """
    return s_xx/delt
def sigma_a2(s,delt):
    """"
    Sigma para pendiente (Gamma)
    """
    return s/delt
def dec_exp(t,v_0,tau):
    """
    Modelamos el decaimiento exponencial para los circuitos RL:
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
    return v_0*np.exp(-tau*t)

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan]AJUSTE LINEAL (Linearización)[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

ln_v= np.log(voltage)
sigma_ln_v= uncer/voltage

#por ajustar y=mx+b
#  y= ln_v, m=- tau, x=t, b = ln(vo)
# ln_v = - tau*t+ln(vo)

coeficientes = np.polyfit(ttime, ln_v, deg=1, w=1/sigma_ln_v)
m_linear = coeficientes[0] #pendiente 
b_linear = coeficientes[1] #ordenada

tau_linear = -m_linear #gamma
v_0_linear = np.exp(b_linear) #V0

console.print(f"\n[green]Resultados del ajuste lineal:[/green]")
console.print(f"  V₀ = {v_0_linear:.4f} V")
console.print(f"  τ  = {tau_linear:.4f} ms⁻¹")
console.print(f"  ln(V₀) = {b_linear:.4f}")

# Calcular R² para el ajuste lineal
v_ajuste_linear = dec_exp(ttime, v_0_linear, tau_linear)
residuos_linear = voltage - v_ajuste_linear
SS_res = np.sum(residuos_linear**2)
SS_tot = np.sum((voltage - np.mean(voltage))**2)
R2_linear = 1 - (SS_res / SS_tot)
sigma_gamma= sigma_a2(S(uncer), delta(S(uncer),S_xx(uncer,ttime),S_x(uncer, ttime)))
sigma_vo= sigma_a1(S_xx(uncer, ttime),delta(S(uncer),S_xx(uncer,ttime),S_x(uncer, ttime)))
console.print(f"  σ_τ = {sigma_gamma:.4f}")
console.print(f"  σ_V₀= {sigma_vo:.4f}")
console.print(f"  R² = {R2_linear:.6f}")

# obtenemos chi-cuadrado

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan](b) BONDAD DEL AJUSTE: χ²[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

# Calcular valores ajustados
V_ajustado = dec_exp(ttime, v_0_linear, tau_linear)

# Residuos
residuos = voltage - V_ajustado
residuos_normalizados = residuos / uncer

# Chi-cuadrado
chi2 = np.sum(residuos_normalizados**2)

# Grados de libertad: n_datos - n_parámetros
n_datos = len(ttime)
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
    interpretacion = "[yellow] χ²_red < 0.5: Posible sobreestimación de errores[/yellow]"
elif 0.5 <= chi2_reducido <= 2.0:
    interpretacion = "[green] 0.5 ≤ χ²_red ≤ 2.0: Ajuste excelente[/green]"
elif 2.0 < chi2_reducido <= 5.0:
    interpretacion = "[yellow] 2.0 < χ²_red ≤ 5.0: Ajuste aceptable, revisar errores[/yellow]"
else:
    interpretacion = "[red] χ²_red > 5.0: Ajuste pobre, modelo incorrecto o errores subestimados[/red]"

console.print(f"  {interpretacion}")

# ¿Tiene sentido?
console.print(f"\n[bold cyan]¿Tiene sentido?[/bold cyan]")
if 0.5 <= chi2_reducido <= 2.0:
    console.print("  [green] SÍ. El modelo exponencial describe bien los datos.[/green]")
    console.print("  [green]  Los residuos son consistentes con las incertidumbres.[/green]")
else:
    console.print("  [yellow] REVISAR. El ajuste puede mejorarse.[/yellow]")

#  sisisis, pero y la fisica ?

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan]INTERPRETACIÓN FÍSICA[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

# Constante de tiempo: T = 1/τ
T_constante = 1.0 / tau_linear
sigma_T = sigma_gamma / (tau_linear**2)  # Propagación de error

console.print(f"\n[yellow]Constante de tiempo:[/yellow]")
console.print(f"  T = 1/τ = {T_constante:.4f} ± {sigma_T:.4f} ms")
console.print(f"  (tiempo para que V decaiga a V₀/e = {v_0_linear/np.e:.3f} V)")

# Vida media: t_1/2 = ln(2)/τ
t_media = np.log(2) / tau_linear
sigma_t_media = sigma_gamma * np.log(2) / (tau_linear**2)

console.print(f"\n[yellow]Vida media:[/yellow]")
console.print(f"  t_1/2 = ln(2)/τ = {t_media:.4f} ± {sigma_t_media:.4f} ms")
console.print(f"  (tiempo para que V decaiga a V₀/2 = {v_0_linear/2:.3f} V)")

# Si conocemos R o L, podemos calcular el otro
console.print(f"\n[yellow]Relación con componentes:[/yellow]")
console.print(f"  τ = R/L")
console.print(f"  Si R = 100 Ω → L = R/τ = {100/tau_linear:.2f} mH")
console.print(f"  Si L = 10 mH → R = τ·L = {tau_linear*10:.2f} Ω")

# sisisis pero y como se ve
# Creamos un dominio 
t_dom =  np.linspace(0, 750,1000)
V_ajustado=dec_exp(t_dom,v_0_linear,tau_linear)

plt.plot(t_dom,V_ajustado, '-', color ='b', linewidth =2.5,
         label=f'Ajuste: V₀={v_0_linear:.3f} V, τ={tau_linear:.3f} ms⁻¹', alpha=0.8)
plt.plot(ttime , voltage, 'o', color ='r', linewidth =2.5,
         label=f'Datos experimentales', alpha=0.8)
plt.yscale('log')
plt.xlabel('Tiempo t (ms)', fontsize=13, fontweight='bold')
plt.ylabel('log(Voltaje V(t)) (V)', fontsize=13, fontweight='bold')
plt.title('Gráfica Semi-Log (Requerida)', fontsize=15, fontweight='bold')
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3, which='both')
filename = f"{output_dir}/grafico_semilog_RLC.png"
plt.savefig(filename, dpi = 300, bbox_inches="tight")
plt.show()
console.print("[bold green] Grafico guardado: ", filename)
# popt, pcov = curve_fit(
#     dec_exp, ttime, 
#     voltage,p0=[v_0_linear,tau_linear],
#     sigma = uncer, absolute_sigma=True)
# errores = np.sqrt(np.diag(pcov))
# sigma_V=errores[0]
# ==============================================================================
# RESUMEN FINAL
# ==============================================================================

console.print("\n")
console.rule("[bold green]RESUMEN DE RESULTADOS[/bold green]")

resumen = f"""[bold cyan](a) Parámetros ajustados con incertidumbres:[/bold cyan]

    V₀ = {v_0_linear:.4f} ± {sigma_vo:.4f} V  
    τ  = {tau_linear:.4f} ± {sigma_gamma:.4f} ms⁻¹ 

[bold cyan](b) Bondad del ajuste:[/bold cyan]

    χ² = {chi2:.4f}
    χ²_reducido = {chi2_reducido:.4f}
    
    {interpretacion.replace('[green]', '').replace('[/green]', '').replace('[yellow]', '').replace('[/yellow]', '').replace('[red]', '').replace('[/red]', '')}

[bold cyan](c) Gráfica semi-log:[/bold cyan]

    ✓ Generada
    ✓ En escala semi-log, el decaimiento exponencial es una línea recta
    ✓ Pendiente = -τ = {-tau_linear:.4f} ms⁻¹

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

console.print("\n[bold green] Análisis completado exitosamente[/bold green]")