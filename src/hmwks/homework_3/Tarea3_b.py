"""
poisson_2d_gauss_seidel_periodic.py

Solución numérica de la ecuación de Poisson en [0,2π]x[0,2π] con condiciones periódicas
usando Gauss-Seidel. Visualización de la evolución temporal.
f(x,y) = cos(3x+4y) - cos(5x-2y)
"""

import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Parámetros del problema
# -----------------------
N = 128  # resolución de la malla (prueba con 64, 128, 256)
Lx = 2 * np.pi
Ly = 2 * np.pi
dx = Lx / N
dy = Ly / N

# Crear malla
x = np.linspace(0, Lx, N, endpoint=False)
y = np.linspace(0, Ly, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

# Función fuente f(x,y)
f = np.cos(3 * X + 4 * Y) - np.cos(5 * X - 2 * Y)

# -----------------------
# Inicialización
# -----------------------
phi = np.zeros_like(f)  # condición inicial

# Parámetros de Gauss-Seidel
tol = 1e-8
max_iter = 20000
res = 1.0
it = 0

# Guardar snapshots de la evolución
snapshots = {}
snapshot_iterations = [0, 50, 200, 500, 1000, 5000]

print(f"Iniciando Gauss-Seidel con N={N}, tol={tol:.2e}")
print(f"dx={dx:.4f}, dy={dy:.4f}")
print("-" * 60)

# -----------------------
# Método de Gauss-Seidel con condiciones periódicas
# -----------------------
while res > tol and it < max_iter:
    res = 0.0

    # Guardar snapshot si corresponde
    if it in snapshot_iterations:
        snapshots[it] = phi.copy()
        print(f"  Snapshot guardado en iteración {it}")

    # Recorrer todos los puntos de la malla
    for i in range(N):
        ip = (i + 1) % N  # índice siguiente (periódico)
        im = (i - 1) % N  # índice anterior (periódico)

        for j in range(N):
            jp = (j + 1) % N
            jm = (j - 1) % N

            # Guardar valor anterior
            old_val = phi[i, j]

            # Ecuación de diferencias finitas reorganizada para phi[i,j]
            # ∇²phi = f  →  phi[i,j] = 0.25*(suma_vecinos - dx²*f[i,j])
            phi[i, j] = 0.25 * (
                phi[ip, j] + phi[im, j] + phi[i, jp] + phi[i, jm] - dx * dx * f[i, j]
            )

            # Calcular residuo local
            local_res = abs(phi[i, j] - old_val)
            res = max(res, local_res)

    # Imprimir progreso
    if it % 500 == 0:
        print(f"Iteración {it:5d}: residuo = {res:.6e}")

    it += 1

# Guardar solución final
snapshots[it - 1] = phi.copy()
print(f"  Snapshot guardado en iteración {it-1} (Final)")

print("-" * 60)
print(f"✓ Convergencia alcanzada en {it} iteraciones")
print(f"  Residuo final: {res:.6e}")
print(f"  Tolerancia: {tol:.6e}")

# -----------------------
# VISUALIZACIÓN: Evolución temporal (snapshots)
# -----------------------
samp = max(1, N // 80)  # submuestreo para visualización 3D
sorted_iters = sorted(snapshots.keys())
num_snapshots = len(sorted_iters)

# Calcular configuración de subplots
n_cols = 3
n_rows = (num_snapshots + n_cols - 1) // n_cols  # redondeo hacia arriba

fig = plt.figure(figsize=(18, 6 * n_rows))

for idx, iter_num in enumerate(sorted_iters):
    ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
    snapshot = snapshots[iter_num]

    # Crear superficie 3D
    surf = ax.plot_surface(
        X[::samp, ::samp],
        Y[::samp, ::samp],
        snapshot[::samp, ::samp],
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.9,
        edgecolor="none",
    )

    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_zlabel("φ(x,y)", fontsize=10)

    # Título diferenciado para la solución final
    if iter_num == sorted_iters[-1]:
        ax.set_title(
            f"Iteración {iter_num} (SOLUCIÓN FINAL)\nResiduo: {res:.2e}",
            fontsize=12,
            fontweight="bold",
            color="green",
        )
    else:
        ax.set_title(f"Iteración {iter_num}", fontsize=12, fontweight="bold")

    # Ajustar vista
    ax.view_init(elev=25, azim=45)

    # Barra de color
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label("φ", fontsize=9)

    # Información adicional en la primera gráfica
    if idx == 0:
        info_text = f"N = {N}x{N}\ndx = {dx:.4f}\ntol = {tol:.2e}"
        ax.text2D(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        )

plt.suptitle(
    "Evolución de φ(x,y) - Método de Gauss-Seidel con Condiciones Periódicas",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)
plt.tight_layout()
plt.show()

# -----------------------
# Resumen de resultados
# -----------------------
print("\n" + "=" * 60)
print("RESUMEN DE RESULTADOS")
print("=" * 60)
print(f"Parámetros:")
print(f"  - Resolución: {N}x{N}")
print(f"  - Dominio: [0, {Lx:.4f}] x [0, {Ly:.4f}]")
print(f"  - dx = dy = {dx:.6f}")
print(f"\nConvergencia:")
print(f"  - Iteraciones: {it}")
print(f"  - Residuo final: {res:.6e}")
print(f"  - Tolerancia: {tol:.6e}")
print(f"\nSolución φ(x,y):")
print(f"  - Mínimo: {np.min(phi):.6f}")
print(f"  - Máximo: {np.max(phi):.6f}")
print(f"  - Media: {np.mean(phi):.6f}")
print(f"\nSnapshots guardados en iteraciones: {sorted_iters}")
print("=" * 60)
