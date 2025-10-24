#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
heat_1d_explicit_vs_analytical.py

Solución numérica (FTCS) de la ecuación del calor 1D comparada con la solución analítica
serie de Fourier. Visualizaciones optimizadas.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# -----------------------
# Parámetros físicos
# -----------------------
L = 1.0             # longitud (m)
T0 = 100.0          # temperatura inicial (°C)
alpha_Al = 9.7e-5   # difusividad aluminio (m^2/s) aprox
alpha_wood = 1e-6   # difusividad madera (m^2/s) aprox

# -----------------------
# Parámetros numéricos
# -----------------------
N = 200               # número de intervalos espaciales
dx = L / N
x = np.linspace(0, L, N+1)

# elegir material
alpha = alpha_Al

# tiempo de simulación (segundos)
t_final = 20000.0

# condicion de estabilidad: dt <= dx^2 / (2 alpha)
dt_stable = 0.5 * dx*dx / alpha
print(f"Paso de tiempo máximo estable dt <= {dt_stable:.3e} s")

# Usamos dt algo menor que el límite
dt = 0.9 * dt_stable

# número de pasos
nt = int(t_final / dt) + 1
print(f"Usando dt={dt:.3e}, nt={nt}")

# r = alpha dt / dx^2
r = alpha * dt / dx**2
print(f"r = {r:.6f}")

# -----------------------
# Inicialización
# -----------------------
T = np.ones(N+1) * T0
T[0] = 0.0
T[-1] = 0.0

# para superficie: guardamos cada k pasos
sample_stride = max(1, nt // 300)
T_evolution = []
t_evolution = []

# -----------------------
# Esquema explícito (FTCS)
# -----------------------
print("Ejecutando simulación FTCS...")
for n in range(nt+1):
    t = n*dt
    if n % sample_stride == 0:
        T_evolution.append(T.copy())
        t_evolution.append(t)
    # iteración FTCS (excluir puntos frontera)
    Tn = T.copy()
    T[1:-1] = Tn[1:-1] + r*(Tn[2:] - 2*Tn[1:-1] + Tn[:-2])
    T[0] = 0.0
    T[-1] = 0.0

# convertir a arrays
T_evolution = np.array(T_evolution)
t_evolution = np.array(t_evolution)

# -----------------------
# Solución analítica
# -----------------------
def analytical_T(x, t, L=1.0, T0=100.0, alpha=alpha):
    """Serie de Fourier truncada"""
    M = 500  # número de términos
    s = np.zeros_like(x)
    for k in range(M):
        n = 2*k + 1
        coef = 4*T0 / (n*np.pi)
        s += coef * np.sin(n*np.pi*x/L) * np.exp(-alpha*(n*np.pi/L)**2 * t)
    return s

# calcular solución analítica para los mismos tiempos
print("Calculando solución analítica...")
T_analytical_evolution = []
for t in t_evolution:
    T_analytical_evolution.append(analytical_T(x, t, L=L, T0=T0, alpha=alpha))
T_analytical_evolution = np.array(T_analytical_evolution)

# Calcular diferencia
T_diff = T_evolution - T_analytical_evolution

# Calcular error L2 final
error_L2_final = np.sqrt(np.mean(T_diff[-1]**2))
print(f"\nError L2 final: {error_L2_final:.6e} °C")

# -----------------------
# Preparar mallas para gráficos
# -----------------------
X, Tt = np.meshgrid(x, t_evolution)

# -----------------------
# FIGURA 1: Superficies 3D (Numérica, Analítica y Diferencia)
# -----------------------
fig = plt.figure(figsize=(18, 5))

# Subplot 1: Solución numérica
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X, Tt, T_evolution, cmap=cm.viridis, 
                         linewidth=0, antialiased=False, alpha=0.9)
ax1.set_xlabel('x (m)', fontsize=10)
ax1.set_ylabel('t (s)', fontsize=10)
ax1.set_zlabel('T (°C)', fontsize=10)
ax1.set_title('Solución Numérica (FTCS)', fontsize=12, fontweight='bold')
ax1.view_init(elev=25, azim=45)
cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
cbar1.set_label('T (°C)', fontsize=9)

# Subplot 2: Solución analítica
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X, Tt, T_analytical_evolution, cmap=cm.plasma,
                         linewidth=0, antialiased=False, alpha=0.9)
ax2.set_xlabel('x (m)', fontsize=10)
ax2.set_ylabel('t (s)', fontsize=10)
ax2.set_zlabel('T (°C)', fontsize=10)
ax2.set_title('Solución Analítica (Serie Fourier)', fontsize=12, fontweight='bold')
ax2.view_init(elev=25, azim=45)
cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
cbar2.set_label('T (°C)', fontsize=9)

# Subplot 3: Diferencia en 2D (no 3D)
ax3 = fig.add_subplot(133)
im3 = ax3.imshow(T_diff.T, extent=[0, t_final, 0, L], aspect='auto', 
                 origin='lower', cmap='RdBu_r', interpolation='bilinear')
ax3.set_xlabel('t (s)', fontsize=10)
ax3.set_ylabel('x (m)', fontsize=10)
ax3.set_title('Diferencia (Numérica - Analítica)', fontsize=12, fontweight='bold')
cbar3 = fig.colorbar(im3, ax=ax3)
cbar3.set_label('ΔT (°C)', fontsize=9)

# Añadir información del error
error_text = f'Error L2 = {error_L2_final:.2e} °C'
ax3.text(0.02, 0.98, error_text, transform=ax3.transAxes,
         verticalalignment='top', fontsize=10, color='white',
         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

plt.suptitle('Ecuación del Calor 1D: Comparación Numérica vs Analítica',
            fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# -----------------------
# FIGURA 2: Isotermas (Contornos)
# -----------------------
fig = plt.figure(figsize=(18, 5))

# Subplot 1: Isotermas numérica
ax1 = fig.add_subplot(131)
levels = 20
CS1 = ax1.contourf(X, Tt, T_evolution, levels=levels, cmap='viridis')
contours1 = ax1.contour(X, Tt, T_evolution, levels=10, colors='white', 
                        linewidths=0.5, alpha=0.4)
ax1.clabel(contours1, inline=True, fontsize=7, fmt='%1.1f')
ax1.set_xlabel('x (m)', fontsize=10)
ax1.set_ylabel('t (s)', fontsize=10)
ax1.set_title('Isotermas - Solución Numérica', fontsize=12, fontweight='bold')
cbar1 = fig.colorbar(CS1, ax=ax1)
cbar1.set_label('T (°C)', fontsize=9)

# Subplot 2: Isotermas analítica
ax2 = fig.add_subplot(132)
CS2 = ax2.contourf(X, Tt, T_analytical_evolution, levels=levels, cmap='plasma')
contours2 = ax2.contour(X, Tt, T_analytical_evolution, levels=10, colors='white',
                        linewidths=0.5, alpha=0.4)
ax2.clabel(contours2, inline=True, fontsize=7, fmt='%1.1f')
ax2.set_xlabel('x (m)', fontsize=10)
ax2.set_ylabel('t (s)', fontsize=10)
ax2.set_title('Isotermas - Solución Analítica', fontsize=12, fontweight='bold')
cbar2 = fig.colorbar(CS2, ax=ax2)
cbar2.set_label('T (°C)', fontsize=9)

# Subplot 3: Isotermas de la diferencia
ax3 = fig.add_subplot(133)
CS3 = ax3.contourf(X, Tt, T_diff, levels=levels, cmap='RdBu_r')
contours3 = ax3.contour(X, Tt, T_diff, levels=10, colors='black',
                        linewidths=0.5, alpha=0.3)
ax3.clabel(contours3, inline=True, fontsize=7, fmt='%1.2e')
ax3.set_xlabel('x (m)', fontsize=10)
ax3.set_ylabel('t (s)', fontsize=10)
ax3.set_title('Isotermas - Diferencia (Error)', fontsize=12, fontweight='bold')
cbar3 = fig.colorbar(CS3, ax=ax3)
cbar3.set_label('ΔT (°C)', fontsize=9)

plt.suptitle('Isotermas de Temperatura T(x,t)',
            fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# -----------------------
# Resumen de resultados
# -----------------------
print("\n" + "="*60)
print("RESUMEN DE RESULTADOS")
print("="*60)
print(f"Parámetros:")
print(f"  - Resolución espacial: N = {N}")
print(f"  - Pasos temporales: nt = {nt}")
print(f"  - dx = {dx:.6f} m")
print(f"  - dt = {dt:.6e} s")
print(f"  - r = α·dt/dx² = {r:.6f}")
print(f"\nEstabilidad:")
print(f"  - Condición: r ≤ 0.5")
print(f"  - Estado: {'✓ ESTABLE' if r <= 0.5 else '✗ INESTABLE'}")
print(f"\nError final:")
print(f"  - Error L2: {error_L2_final:.6e} °C")
print(f"  - Error máximo: {np.max(np.abs(T_diff[-1])):.6e} °C")
print("="*60)