"""
Resolución por el método de Newton-Raphson

Ecuaciones:
(a) cos(θ)^2 = θ^2
(b) 4cos(θ)^2 = θ^2
(c) 4sin(θ)^2 = θ^2
"""
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------
# Método de Newton-Raphson
# ------------------------------
def newton_raphson(f, df, x0, tol=1e-8, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-12:
            raise ValueError("Derivada demasiado pequeña; posible división por cero.")
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new, i + 1
        x = x_new
    raise ValueError("No se alcanzó la convergencia tras el máximo de iteraciones.")


# ------------------------------
# Definición de funciones
# ------------------------------

# (a) cos^2(x) = x^2  →  f(x) = cos^2(x) - x^2
f1 = lambda x: np.cos(x) ** 2 - x**2
df1 = lambda x: -2 * np.cos(x) * np.sin(x) - 2 * x

# (b) 4cos^2(x) = x^2  →  f(x) = 4cos^2(x) - x^2
f2 = lambda x: 4 * np.cos(x) ** 2 - x**2
df2 = lambda x: -8 * np.cos(x) * np.sin(x) - 2 * x

# (c) 4sin^2(x) = x^2  →  f(x) = 4sin^2(x) - x^2
f3 = lambda x: 4 * np.sin(x) ** 2 - x**2
df3 = lambda x: 8 * np.sin(x) * np.cos(x) - 2 * x

# ------------------------------
# Aproximaciones iniciales
# ------------------------------
x0_values = [0.5, 1.0, 1.5]

# ------------------------------
# Resolución y resultados
# ------------------------------
try:
    sol1, n1 = newton_raphson(f1, df1, x0_values[0])
    sol2, n2 = newton_raphson(f2, df2, x0_values[1])
    sol3, n3 = newton_raphson(f3, df3, x0_values[2])

    print("=== RESULTADOS ===")
    print(f"(a) cos²(x) = x²   →  x = {sol1:.8f}   (iteraciones: {n1})")
    print(f"(b) 4cos²(x) = x²  →  x = {sol2:.8f}   (iteraciones: {n2})")
    print(f"(c) 4sin²(x) = x²  →  x = {sol3:.8f}   (iteraciones: {n3})")

except Exception as e:
    print("Error:", e)

# ------------------------------
# Comparación gráfica (opcional)
# ------------------------------

x = np.linspace(-2 * np.pi, 2 * np.pi, 400)

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(x, np.cos(x) ** 2, label="cos²(x)")
plt.plot(x, x**2, label="x²")
plt.title("(a) cos²(x) = x²")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x, 4 * np.cos(x) ** 2, label="4cos²(x)")
plt.plot(x, x**2, label="x²")
plt.title("(b) 4cos²(x) = x²")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x, 4 * np.sin(x) ** 2, label="4sin²(x)")
plt.plot(x, x**2, label="x²")
plt.title("(c) 4sin²(x) = x²")
plt.legend()

plt.tight_layout()
plt.show()
