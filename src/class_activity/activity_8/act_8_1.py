""" "
coded: 30/10/25 by: Oscar Valencia

1. Utilizando el mÈtodo de Newton-Raphson resuelva:
(a) cos\2 a = a^2

(b) 4cos^2 a = a^2

(c) 4 sin^2 a = a^2
Compare sus soluciones con los resultados del libro de Zetilli en
la presentaciÛn Raices.
"""

import numpy as np

x0_val = [0.5, 1.0, 1.5]  # aprox iniciales

x0 = 1111.0
dx = 3.0e-4
err = 1e-8
Nmax = 100
# Parámetros

# (a) cos^2(x) = x^2  →  f(x) = cos^2(x) - x^2
f1 = lambda x: np.cos(x) ** 2 - x**2
df1 = lambda x: -2 * np.cos(x) * np.sin(x) - 2 * x

# (b) 4cos^2(x) = x^2  →  f(x) = 4cos^2(x) - x^2
f2 = lambda x: 4 * np.cos(x) ** 2 - x**2
df2 = lambda x: -8 * np.cos(x) * np.sin(x) - 2 * x

# (c) 4sin^2(x) = x^2  →  f(x) = 4sin^2(x) - x^2
f3 = lambda x: 4 * np.sin(x) ** 2 - x**2
df3 = lambda x: 8 * np.sin(x) * np.cos(x) - 2 * x


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


try:
    sol1, n1 = newton_raphson(f1, df1, x0_val[0])
    sol2, n2 = newton_raphson(f2, df2, x0_val[1])
    sol3, n3 = newton_raphson(f3, df3, x0_val[2])

    print("=== RESULTADOS ===")
    print(f"(a) cos²(x) = x²   →  x = {sol1:.8f}   (iteraciones: {n1})")
    print(f"(b) 4cos²(x) = x²  →  x = {sol2:.8f}   (iteraciones: {n2})")
    print(f"(c) 4sin²(x) = x²  →  x = {sol3:.8f}   (iteraciones: {n3})")

except Exception as e:
    print("Error:", e)
