import math


# --- Función para calcular el Overflow ---
# Encuentra el mayor número representable antes del desbordamiento.
def calcular_overflow():
    x = 1.0
    while x * 2 != float("inf"):
        x *= 2.0
    return x


# --- Función para calcular el Underflow ---
# Encuentra el menor número positivo representable antes de 0.
def calcular_underflow():
    y = 1.0
    while y / 2 != 0:
        y /= 2
    return y


# --- Función para calcular el Epsilon de Máquina ---
# Encuentra el menor número que, al sumarse a 1.0, cambia el resultado.
def calcular_epsilon():
    eps = 1.0
    while 1.0 + eps / 2 > 1.0:
        eps /= 2
    return eps


# --- Serie de Taylor para calcular sin(x) ---
def sin_series(x, tol=1e-8, reduce_mod=True, max_terms=1000):
    if reduce_mod:
        x_red = ((x + math.pi) % (2 * math.pi)) - math.pi
    else:
        x_red = x

    term = x_red
    suma = term
    n = 0

    tabla = [
        (
            n + 1,
            suma,
            (
                abs((suma - math.sin(x)) / math.sin(x))
                if math.sin(x) != 0
                else abs(suma - math.sin(x))
            ),
        )
    ]

    while abs(term) >= tol and n < max_terms:
        denom = (2 * (n + 1)) * (2 * (n + 1) + 1)
        term = -term * x_red * x_red / denom
        suma += term
        n += 1
        error_rel = (
            abs((suma - math.sin(x)) / math.sin(x))
            if math.sin(x) != 0
            else abs(suma - math.sin(x))
        )
        tabla.append((n + 1, suma, error_rel))

    return suma, tabla


if __name__ == "__main__":
    of = calcular_overflow()
    uf = calcular_underflow()
    eps = calcular_epsilon()

    print("Overflow estimado:", of)
    print("Underflow estimado:", uf)
    print("Epsilon de máquina:", eps)

    x = math.pi / 4
    suma, tabla = sin_series(x, tol=1e-8)

    print(f"\nCálculo de sin({x}) con serie de Taylor:")
    print("{:<5s} {:<20s} {:<20s}".format("N", "Suma", "Error relativo"))
    for fila in tabla:
        print("{:<5d} {:<20.12e} {:<20.12e}".format(fila[0], fila[1], fila[2]))
