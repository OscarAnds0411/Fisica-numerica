from pylab import *
import numpy as np
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from rich.console import Console
from rich.table import Table
import os

# resolver esta chingadera: 
# dr^2dt=cte
def ODE(r,t):
    pos = r[0]
    vel = r[1]
    ace = 0.5
    #r[0] = r
    #r[1] = v   ----- a resolver:
    #r[3] = a   ----- dr^2/dt^2=dv/dt=cte
    # a=dv/dt
    #v = dr/dt
    dr_dt= vel
    dv_dt = ace
    return array([dr_dt,dv_dt])
# Parámetros
g = 9.81  # Aceleración gravitacional (m/s²)
a =0.5    # Aceleracion ficticia 

def EDO_caida_libre(estado, tiempo):
    """
    Sistema para caída libre con aceleración constante.
    
    Variables de estado:
        estado[0] = r  (posición vertical)
        estado[1] = v  (velocidad vertical)
    
    Ecuaciones:
        dr/dt = v
        dv/dt = -g  (aceleración constante hacia abajo)
    """
    r = estado[0]
    v = estado[1]
    
    dr_dt = v
    dv_dt = -g  # Aceleración constante (negativa hacia abajo)
    
    return np.array([dr_dt, dv_dt])

# Condiciones iniciales
r0 = 10.0   # Altura inicial (m)
v0 = 0.0     # Velocidad inicial (m/s)
estado_inicial = [r0, v0]

# Tiempo
t = np.linspace(0, 4.5, 1000)

# Resolver
solucion = odeint(EDO_caida_libre, estado_inicial, t)

print(solucion.shape)

# Extraer resultados
r = solucion[:, 0]
v = solucion[:, 1]

# Solución analítica (para comparar)
#r_analitica = r0 + v0*t + 0.5*a*t**2
#v_analitica = v0 + a*t

r_analitica = r0 + v0*t - 0.5*g*t**2
v_analitica = v0 - g*t
# Graficar
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Posición vs tiempo
axes[0].plot(t, r, 'b-', linewidth=2, label='Numérica')
axes[0].plot(t, r_analitica, 'r--', linewidth=2, alpha=0.7, label='Analítica')
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Suelo')
axes[0].set_xlabel('Tiempo (s)', fontsize=12)
axes[0].set_ylabel('Altura r (m)', fontsize=12)
axes[0].set_title('Posición vs Tiempo', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Subplot 2: Velocidad vs tiempo
axes[1].plot(t, v, 'b-', linewidth=2, label='Numérica')
axes[1].plot(t, v_analitica, 'r--', linewidth=2, alpha=0.7, label='Analítica')
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].set_xlabel('Tiempo (s)', fontsize=12)
axes[1].set_ylabel('Velocidad v (m/s)', fontsize=12)
axes[1].set_title('Velocidad vs Tiempo', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Subplot 3: Espacio de fase
axes[2].plot(r, v, 'g-', linewidth=2)
axes[2].plot(r[0], v[0], 'go', markersize=10, label='Inicio')
axes[2].plot(r[-1], v[-1], 'rs', markersize=10, label='Final')
axes[2].set_xlabel('Posición r (m)', fontsize=12)
axes[2].set_ylabel('Velocidad v (m/s)', fontsize=12)
axes[2].set_title('Espacio de Fase', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.show()

print(f"Tiempo hasta tocar el suelo: {np.sqrt(2*r0/g):.2f} s (analítico)")
"""
ANÁLISIS SIMPLE DE MODOS NORMALES: LINEAL VS NO LINEAL
Sin FFT - Solo conteo de periodos y comparación visual
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
import os

# Crear carpeta para resultados
output_dir = 'resultados_harm'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

console = Console()

# ===================================================================
# PARÁMETROS DEL SISTEMA
# ===================================================================
m = 1.0
k = 15.0
k_c = 5.0

console.print("\n" + "="*70, style="bold cyan")
console.print(" ANÁLISIS SIMPLE: MODOS NORMALES (LINEAL VS NO LINEAL)", style="bold cyan")
console.print("="*70, style="bold cyan")

# ===================================================================
# FUNCIONES DEL SISTEMA
# ===================================================================
def coupled_linear(r, t):
    """Sistema LINEAL"""
    x1, v1, x2, v2 = r
    
    dx1_dt = v1
    dv1_dt = -(k/m)*x1 - (k_c/m)*(x1 - x2)
    dx2_dt = v2
    dv2_dt = -(k/m)*x2 - (k_c/m)*(x2 - x1)
    
    return np.array([dx1_dt, dv1_dt, dx2_dt, dv2_dt])

def coupled_nonlinear(r, t):
    """Sistema NO LINEAL"""
    x1, v1, x2, v2 = r
    
    F1_ext = -(k/m) * (x1 + 0.1*x1**3)
    F1_coupling = -(k_c/m) * ((x1 - x2) + 0.1*(x1 - x2)**3)
    
    F2_ext = -(k/m) * (x2 + 0.1*x2**3)
    F2_coupling = -(k_c/m) * ((x2 - x1) + 0.1*(x2 - x1)**3)
    
    dx1_dt = v1
    dv1_dt = F1_ext + F1_coupling
    dx2_dt = v2
    dv2_dt = F2_ext + F2_coupling
    
    return np.array([dx1_dt, dv1_dt, dx2_dt, dv2_dt])

# ===================================================================
# FUNCIÓN PARA CALCULAR PERIODO (MÉTODO SIMPLE)
# ===================================================================
def calcular_periodo(t, x):
    """
    Calcula el periodo promedio encontrando cruces por cero.
    """
    # Encontrar cruces por cero con pendiente positiva
    cruces = []
    for i in range(len(x)-1):
        if x[i] <= 0 and x[i+1] > 0:  # Cruce ascendente
            # Interpolación lineal para mayor precisión
            t_cruce = t[i] - x[i] * (t[i+1] - t[i]) / (x[i+1] - x[i])
            cruces.append(t_cruce)
    
    if len(cruces) < 2:
        return None
    
    # Calcular diferencias entre cruces consecutivos (medio periodo)
    # Multiplicar por 2 para obtener periodo completo
    periodos = [(cruces[i+1] - cruces[i]) * 2 for i in range(len(cruces)-1)]
    
    return np.mean(periodos)

# ===================================================================
# MODOS NORMALES TEÓRICOS (LINEAL)
# ===================================================================
console.print("\n" + "="*70, style="bold yellow")
console.print(" MODOS NORMALES TEÓRICOS (SISTEMA LINEAL)", style="bold yellow")
console.print("="*70, style="bold yellow")

# Matriz del sistema
A = np.array([[-(k + k_c)/m, k_c/m],
              [k_c/m, -(k + k_c)/m]])

eigenvalues, eigenvectors = np.linalg.eig(A)
omega_teorico = np.sqrt(-eigenvalues)
freq_teorico = omega_teorico / (2 * np.pi)
periodo_teorico = 1 / freq_teorico

console.print(f"\n[bold]Modo 1 (Simétrico):[/bold]")
console.print(f"  ω₁ = {omega_teorico[0]:.4f} rad/s")
console.print(f"  f₁ = {freq_teorico[0]:.4f} Hz")
console.print(f"  T₁ = {periodo_teorico[0]:.4f} s")

console.print(f"\n[bold]Modo 2 (Antisimétrico):[/bold]")
console.print(f"  ω₂ = {omega_teorico[1]:.4f} rad/s")
console.print(f"  f₂ = {freq_teorico[1]:.4f} Hz")
console.print(f"  T₂ = {periodo_teorico[1]:.4f} s")

# ===================================================================
# ANÁLISIS PARA DIFERENTES AMPLITUDES
# ===================================================================
console.print("\n" + "="*70, style="bold green")
console.print(" ANÁLISIS NUMÉRICO: DEPENDENCIA CON AMPLITUD", style="bold green")
console.print("="*70, style="bold green")

amplitudes = [0.2, 0.5, 0.8, 1.2]
t = np.linspace(0, 50, 5000)

# Almacenar resultados
resultados = {
    'amplitud': [],
    'T_lineal': [],
    'T_nonlinear': [],
    'f_lineal': [],
    'f_nonlinear': [],
    'diff_pct': []
}

for amp in amplitudes:
    console.print(f"\n[bold cyan]Amplitud: {amp} m[/bold cyan]")
    
    # Condición inicial: modo simétrico
    state0 = [amp, 0.0, amp, 0.0]
    
    # Resolver
    sol_lin = odeint(coupled_linear, state0, t)
    sol_nonlin = odeint(coupled_nonlinear, state0, t)
    
    # Calcular periodos
    T_lin = calcular_periodo(t, sol_lin[:, 0])
    T_nonlin = calcular_periodo(t, sol_nonlin[:, 0])
    
    if T_lin and T_nonlin:
        f_lin = 1 / T_lin
        f_nonlin = 1 / T_nonlin
        diff = abs(f_nonlin - f_lin) / f_lin * 100
        
        resultados['amplitud'].append(amp)
        resultados['T_lineal'].append(T_lin)
        resultados['T_nonlinear'].append(T_nonlin)
        resultados['f_lineal'].append(f_lin)
        resultados['f_nonlinear'].append(f_nonlin)
        resultados['diff_pct'].append(diff)
        
        console.print(f"  Lineal:    T = {T_lin:.4f} s,  f = {f_lin:.4f} Hz")
        console.print(f"  No lineal: T = {T_nonlin:.4f} s,  f = {f_nonlin:.4f} Hz")
        console.print(f"  Diferencia: {diff:.2f}%")

# ===================================================================
# TABLA COMPARATIVA
# ===================================================================
console.print("\n" + "="*70, style="bold magenta")
console.print(" TABLA COMPARATIVA", style="bold magenta")
console.print("="*70, style="bold magenta")

table = Table(title="Frecuencias vs Amplitud", style="magenta")
table.add_column("Amplitud (m)", justify="center")
table.add_column("f Lineal (Hz)", justify="center")
table.add_column("f No Lineal (Hz)", justify="center")
table.add_column("Diferencia (%)", justify="center")

for i in range(len(resultados['amplitud'])):
    table.add_row(
        f"{resultados['amplitud'][i]:.1f}",
        f"{resultados['f_lineal'][i]:.4f}",
        f"{resultados['f_nonlinear'][i]:.4f}",
        f"{resultados['diff_pct'][i]:.2f}"
    )

console.print(table)

# ===================================================================
# GRÁFICAS COMPARATIVAS
# ===================================================================
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Comparación: Sistema Lineal vs No Lineal', 
             fontsize=16, fontweight='bold')

# ===================================================================
# Plot 1: Frecuencia vs Amplitud
# ===================================================================
ax1 = plt.subplot(2, 3, 1)
ax1.plot(resultados['amplitud'], resultados['f_lineal'], 
         'bo-', linewidth=2.5, markersize=10, label='Lineal', alpha=0.8)
ax1.plot(resultados['amplitud'], resultados['f_nonlinear'], 
         'rs-', linewidth=2.5, markersize=10, label='No Lineal', alpha=0.8)
ax1.axhline(y=freq_teorico[0], color='blue', linestyle='--', alpha=0.5, 
            label=f'f₁ teórico = {freq_teorico[0]:.4f} Hz')
ax1.set_xlabel('Amplitud inicial (m)', fontsize=12)
ax1.set_ylabel('Frecuencia (Hz)', fontsize=12)
ax1.set_title('Frecuencia vs Amplitud', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# ===================================================================
# Plot 2: Diferencia porcentual
# ===================================================================
ax2 = plt.subplot(2, 3, 2)
ax2.bar(resultados['amplitud'], resultados['diff_pct'], 
        color='orange', alpha=0.7, edgecolor='black', linewidth=1.5, width=0.15)
ax2.set_xlabel('Amplitud inicial (m)', fontsize=12)
ax2.set_ylabel('Diferencia (%)', fontsize=12)
ax2.set_title('Efecto No Lineal', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# ===================================================================
# Plot 3: Periodo vs Amplitud
# ===================================================================
ax3 = plt.subplot(2, 3, 3)
ax3.plot(resultados['amplitud'], resultados['T_lineal'], 
         'bo-', linewidth=2.5, markersize=10, label='Lineal', alpha=0.8)
ax3.plot(resultados['amplitud'], resultados['T_nonlinear'], 
         'rs-', linewidth=2.5, markersize=10, label='No Lineal', alpha=0.8)
ax3.set_xlabel('Amplitud inicial (m)', fontsize=12)
ax3.set_ylabel('Periodo (s)', fontsize=12)
ax3.set_title('Periodo vs Amplitud', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# ===================================================================
# Plots 4-6: Trayectorias para diferentes amplitudes
# ===================================================================
ejemplos = [0, len(amplitudes)//2, -1]  # Primera, media, última amplitud
titles = ['Amplitud Pequeña', 'Amplitud Media', 'Amplitud Grande']

for idx, (ej, titulo) in enumerate(zip(ejemplos, titles)):
    ax = plt.subplot(2, 3, 4 + idx)
    
    amp = amplitudes[ej]
    state0 = [amp, 0.0, amp, 0.0]
    t_corto = np.linspace(0, 15, 1000)
    
    sol_lin = odeint(coupled_linear, state0, t_corto)
    sol_nonlin = odeint(coupled_nonlinear, state0, t_corto)
    
    ax.plot(t_corto, sol_lin[:, 0], 'b-', linewidth=2, label='Lineal', alpha=0.8)
    ax.plot(t_corto, sol_nonlin[:, 0], 'r--', linewidth=2, label='No Lineal', alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--', alpha=0.3)
    ax.set_xlabel('Tiempo (s)', fontsize=11)
    ax.set_ylabel('Posición x₁ (m)', fontsize=11)
    ax.set_title(f'{titulo} (A = {amp} m)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
filename = f'{output_dir}/comparacion_modos_simple.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
console.print(f"\n[green]✓ Gráfica guardada: {filename}[/green]")
plt.show()

# ===================================================================
# COMPARACIÓN MODO ANTISIMÉTRICO
# ===================================================================
console.print("\n" + "="*70, style="bold cyan")
console.print(" MODO ANTISIMÉTRICO (Desplazamientos Opuestos)", style="bold cyan")
console.print("="*70, style="bold cyan")

amp_anti = 0.8
state0_anti = [amp_anti, 0.0, -amp_anti, 0.0]

sol_lin_anti = odeint(coupled_linear, state0_anti, t)
sol_nonlin_anti = odeint(coupled_nonlinear, state0_anti, t)

T_lin_anti = calcular_periodo(t, sol_lin_anti[:, 0])
T_nonlin_anti = calcular_periodo(t, sol_nonlin_anti[:, 0])

if T_lin_anti and T_nonlin_anti:
    f_lin_anti = 1 / T_lin_anti
    f_nonlin_anti = 1 / T_nonlin_anti
    diff_anti = abs(f_nonlin_anti - f_lin_anti) / f_lin_anti * 100
    
    console.print(f"\n[bold]Resultados (A = {amp_anti} m):[/bold]")
    console.print(f"  Frecuencia teórica:  {freq_teorico[1]:.4f} Hz")
    console.print(f"  Lineal (numérica):   {f_lin_anti:.4f} Hz")
    console.print(f"  No lineal:           {f_nonlin_anti:.4f} Hz")
    console.print(f"  Diferencia:          {diff_anti:.2f}%")

# ===================================================================
# RESUMEN FINAL
# ===================================================================
console.print("\n" + "="*70, style="bold green")
console.print(" CONCLUSIONES", style="bold green")
console.print("="*70, style="bold green")

console.print("\n[bold yellow]Sistema Lineal:[/bold yellow]")
console.print("   ✓ Frecuencia constante (independiente de amplitud)")
console.print(f"   ✓ Modo simétrico: f₁ = {freq_teorico[0]:.4f} Hz")
console.print(f"   ✓ Modo antisimétrico: f₂ = {freq_teorico[1]:.4f} Hz")

console.print("\n[bold yellow]Sistema No Lineal:[/bold yellow]")
console.print("   ✓ Frecuencia aumenta con amplitud")
console.print(f"   ✓ Efecto máximo: {max(resultados['diff_pct']):.2f}% de diferencia")
console.print("   ✓ Término cúbico endurece el resorte")

console.print("\n[bold yellow]Física:[/bold yellow]")
console.print("   • Lineal: T independiente de energía")
console.print("   • No lineal: T decrece con energía (oscila más rápido)")
console.print("   • A mayor amplitud, mayor efecto no lineal")

console.print("\n" + "="*70, style="bold green")
console.print(" ✓ ANÁLISIS COMPLETADO", style="bold green")
console.print("="*70 + "\n", style="bold green")