import math

import numpy as np


# --- Función para calcular el Overflow ---
# Esta función encuentra el mayor número representable antes de que ocurra un desbordamiento (overflow).
def calcular_overflow():
    x = 1.0  # Inicializa x en 1.0
    while x * 2 != float(
        "inf"
    ):  # Multiplica x por 2 hasta que el resultado sea infinito
        x *= 2.0
    return x  # Devuelve el último valor antes del infinito


# --- Función para calcular el Underflow ---
# Esta función encuentra el menor número positivo representable antes de que se convierta en 0.
def calcular_underflow():
    y = 1.0  # Inicializa y en 1.0
    while y / 2 != 0:  # Divide y por 2 hasta que el resultado sea 0
        y /= 2
    return y  # Devuelve el último valor antes de 0


# --- Función para calcular el Epsilon de Máquina ---
# Esta función encuentra el menor número que, al sumarse a 1.0, produce un resultado distinto de 1.0.
def calcular_epsilon():
    eps = 1.0  # Inicializa eps en 1.0
    while 1.0 + eps / 2 > 1.0:  # Divide eps por 2 hasta que 1.0 + eps/2 sea igual a 1.0
        eps /= 2
    return eps  # Devuelve el último valor de eps


# --- Serie de Taylor para calcular sin(x) ---
# Esta función calcula el seno de x utilizando la serie de Taylor.
# - x: valor de entrada
# - tol: tolerancia para detener la serie
# - reduce_mod: si es True, reduce x al rango [-π, π]
# - max_terms: número máximo de términos a calcular
def sin_series(x, tol=1e-8, reduce_mod=True, max_terms=1000):
    if reduce_mod:
        # Reduce x al rango [-π, π] para mejorar la convergencia
        x_red = ((x + math.pi) % (2 * math.pi)) - math.pi
    else:
        x_red = x

    term = x_red  # Primer término de la serie
    suma = term  # Inicializa la suma con el primer término
    n = 0  # Contador de términos
    # Tabla para almacenar los resultados de cada término
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

    while (
        abs(term) >= tol and n < max_terms
    ):  # Itera mientras el término sea mayor que la tolerancia
        denom = (2 * (n + 1)) * (2 * (n + 1) + 1)  # Calcula el denominador del término
        term = -term * x_red * x_red / denom  # Calcula el siguiente término
        suma += term  # Suma el término a la suma total
        n += 1  # Incrementa el contador de términos
        # Calcula el error relativo y lo agrega a la tabla
        error_rel = (
            abs((suma - math.sin(x)) / math.sin(x))
            if math.sin(x) != 0
            else abs(suma - math.sin(x))
        )
        tabla.append((n + 1, suma, error_rel))

    return suma, tabla  # Devuelve la suma y la tabla de resultados


# --- Programa principal ---
if __name__ == "__main__":
    # Calcula el overflow, underflow y epsilon de máquina
    of = calcular_overflow()
    uf = calcular_underflow()
    eps = calcular_epsilon()

    # Imprime los resultados
    print("Overflow estimado:", of)
    print("Underflow estimado:", uf)
    print("Epsilon de máquina:", eps)

    # Ejemplo: calcular sin(x) con tabla
    x = math.pi / 4  # Valor de entrada (π/4)
    suma, tabla = sin_series(x, tol=1e-8)  # Calcula sin(x) con la serie de Taylor
    print(f"\nCálculo de sin({x}) con serie de Taylor:")
    # Imprime el encabezado de la tabla
    print("{:<5s} {:<20s} {:<20s}".format("N", "Suma", "Error relativo"))
    for fila in tabla:  # Itera sobre cada fila de la tabla
        # Imprime el número de término, la suma y el error relativo
        print("{:<5d} {:<20.12e} {:<20.12e}".format(fila[0], fila[1], fila[2]))
