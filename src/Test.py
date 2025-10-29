#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tarea2.py
Solución de la Tarea #2 (cancelación sustractiva y funciones de Bessel esféricas).
Contiene:
 - demostraciones numéricas de reescrituras que evitan cancelación
 - cálculo de j_l(x) por recurrencia hacia arriba (up)
 - cálculo de j_l(x) por recurrencia hacia abajo (down) con normalización
 - comparación y ajuste para alcanzar error relativo <= 1e-10 (cuando sea posible)
"""

from __future__ import annotations

import math

import numpy as np

# ----------------------------
# Utilities
# ----------------------------


def isclose_rel(a, b, tol=1e-12):
    # Verifica si dos números son "cercanos" en términos relativos.
    # Calcula la diferencia relativa entre `a` y `b` y la compara con un umbral `tol`.
    a = float(a)
    b = float(b)
    denom = max(
        abs(a), abs(b), 1e-300
    )  # Evita división por cero usando un denominador mínimo.
    return abs(a - b) / denom <= tol


# ----------------------------
# 1) Cancelación sustractiva: reescrituras y comparaciones
# ----------------------------


def sqrt_minus_one_direct(x):
    # Implementación directa de sqrt(x+1) - 1, que puede sufrir cancelación sustractiva para valores pequeños de x.
    return math.sqrt(x + 1.0) - 1.0


def sqrt_minus_one_stable(x):
    # Implementación estable de sqrt(x+1) - 1, reescrita para evitar cancelación sustractiva.
    # Utiliza la identidad: sqrt(x+1) - 1 = x / (sqrt(x+1) + 1).
    return x / (math.sqrt(x + 1.0) + 1.0)


def sin_minus_sin_direct(x, y):
    # Implementación directa de sin(x) - sin(y), que puede sufrir cancelación para x ≈ y.
    return math.sin(x) - math.sin(y)


def sin_minus_sin_stable(x, y):
    # Implementación estable de sin(x) - sin(y), utilizando identidades trigonométricas.
    # Usa la identidad: sin(x) - sin(y) = 2*cos((x+y)/2)*sin((x-y)/2).
    return 2.0 * math.cos(0.5 * (x + y)) * math.sin(0.5 * (x - y))


def sqdiff_direct(x, y):
    # Implementación directa de x^2 - y^2, que puede sufrir cancelación para x ≈ y.
    return x * x - y * y


def sqdiff_stable(x, y):
    # Implementación estable de x^2 - y^2, utilizando factorización algebraica.
    # Usa la identidad: x^2 - y^2 = (x-y)*(x+y).
    return (x - y) * (x + y)


def one_minus_cos_over_sin_direct(x):
    # Implementación directa de (1 - cos(x)) / sin(x), que puede ser inestable para x ≈ 0.
    # Maneja el caso especial x=0 para evitar división por cero.
    if x == 0.0:
        return 0.0
    return (1.0 - math.cos(x)) / math.sin(x)


def one_minus_cos_over_sin_stable(x):
    # Implementación estable de (1 - cos(x)) / sin(x), utilizando identidades trigonométricas.
    # Usa la identidad: (1 - cos(x)) / sin(x) = tan(x/2).
    if x == 0.0:
        return 0.0
    return math.tan(0.5 * x)


def law_of_cos_direct(a, b, theta):
    # Implementación directa de la ley de cosenos para calcular la longitud del tercer lado de un triángulo.
    return math.sqrt(a * a + b * b - 2.0 * a * b * math.cos(theta))


def law_of_cos_stable(a, b, theta):
    # Implementación estable de la ley de cosenos, reescrita para evitar cancelación numérica.
    # Usa la identidad: sqrt(a^2 + b^2 - 2ab*cos(theta)) = sqrt((a-b)^2 + 4ab*sin^2(theta/2)).
    return math.sqrt((a - b) ** 2 + 4.0 * a * b * (math.sin(0.5 * theta) ** 2))


def demo_cancellation_examples():
    # Demostración de ejemplos de cancelación sustractiva y sus soluciones estables.
    print("=== Cancelación sustractiva: comparaciones numéricas ===")

    print("(a) sqrt(x+1)-1 para valores pequeños de x")
    print("x\t\tDirecto\t\t\tEstable\t\t\tRel. Diff")
    for x in [1e-8, 1e-12, 1e-16, 1e-20, 1e-30]:
        d = sqrt_minus_one_direct(x)  # Implementación directa
        s = sqrt_minus_one_stable(x)  # Implementación estable
        rel_diff = abs(d - s) / max(abs(s), 1e-300)
        print(f"{x: .1e}\t{d:.17e}\t{s:.17e}\t{rel_diff:.2e}")
    print()

    print("(b) sin(x) - sin(y) para x ≈ y")
    print("x\ty\t\tDirecto\t\t\tEstable\t\t\tRel. Diff")
    x = 1.23456789
    for eps in [1e-6, 1e-8, 1e-12, 1e-16, 1e-20]:
        y = x + eps
        d = sin_minus_sin_direct(x, y)  # Implementación directa
        s = sin_minus_sin_stable(x, y)  # Implementación estable
        rel_diff = abs(d - s) / max(abs(s), 1e-300)
        print(f"{x:.8f}\t{y:.8f}\t{d:.17e}\t{s:.17e}\t{rel_diff:.2e}")
    print()

    print("(c) x^2 - y^2 para x ≈ y")
    print("x\ty\t\tDirecto\t\t\tEstable\t\t\tRel. Diff")
    for x, y in [
        (1.0000001, 1.0),
        (1.0000000001, 1.0),
        (1.0000000000001, 1.0),
        (1.0000000000000001, 1.0),
    ]:
        d = sqdiff_direct(x, y)  # Implementación directa
        s = sqdiff_stable(x, y)  # Implementación estable
        rel_diff = abs(d - s) / max(abs(s), 1e-300)
        print(f"{x:.16f}\t{y:.16f}\t{d:.17e}\t{s:.17e}\t{rel_diff:.2e}")
    print()

    print("(d) (1-cos(x))/sin(x) para x pequeño")
    print("x\t\tDirecto\t\t\tEstable\t\t\tRel. Diff")
    for x in [1e-6, 1e-8, 1e-12, 1e-16, 1e-20]:
        d = one_minus_cos_over_sin_direct(x)  # Implementación directa
        s = one_minus_cos_over_sin_stable(x)  # Implementación estable
        rel_diff = abs(d - s) / max(abs(s), 1e-300)
        print(f"{x: .1e}\t{d:.17e}\t{s:.17e}\t{rel_diff:.2e}")
    print()

    print("(e) Ley de cosenos para ángulos pequeños")
    print("a\tb\ttheta\t\tDirecto\t\t\tEstable\t\t\tRel. Diff")
    a = 1.0
    b = 0.999999999999
    for theta in [1e-3, 1e-6, 1e-8, 1e-12, 1e-16]:
        d = law_of_cos_direct(a, b, theta)  # Implementación directa
        s = law_of_cos_stable(a, b, theta)  # Implementación estable
        rel_diff = abs(d - s) / max(abs(s), 1e-300)
        print(f"{a:.1f}\t{b:.12f}\t{theta:.1e}\t{d:.17e}\t{s:.17e}\t{rel_diff:.2e}")
    print("=== fin demo cancelación ===\n")


# ----------------------------
# 2) Bessel esférica j_l(x) por up y por down
# ----------------------------
# j0, j1 closed forms
def j0(x):
    # Cálculo cerrado de la función de Bessel esférica de orden 0.
    if x == 0.0:
        return 1.0
    return math.sin(x) / x


def j1(x):
    # Cálculo cerrado de la función de Bessel esférica de orden 1.
    if x == 0.0:
        return 0.0
    return math.sin(x) / (x * x) - math.cos(x) / x


def j_l_up(x, lmax):
    """Calcula j_l para l=0..lmax usando la recurrencia hacia arriba (usando j0,j1 exactas)."""
    # Inicializa un arreglo para almacenar los valores de j_l.
    js = np.zeros(lmax + 1, dtype=float)
    js[0] = j0(x)  # j_0(x) calculado directamente.
    if lmax == 0:
        return js
    js[1] = j1(x)  # j_1(x) calculado directamente.
    # Aplica la recurrencia hacia arriba para calcular j_l(x) para l >= 2.
    for l in range(1, lmax):
        js[l + 1] = ((2 * l + 1) / x) * js[l] - js[l - 1]
    return js


def j_l_down(x, lmax, pad=60):
    """
    Calcula j_l para l=0..lmax usando recurrencia hacia abajo con normalización.
    pad: número adicional de órdenes para arrancar arriba (se usa Lstart = lmax + pad).
    Método:
     - Inicializamos y_{L+1}=0, y_L=1 y aplicamos recurrencia hacia abajo
     - Normalizamos para que y_0 == j0(x)
    """
    # Maneja el caso especial x == 0: sólo j0(0)=1, j_l(0)=0 para l>0.
    if x == 0.0:
        js = np.zeros(lmax + 1, dtype=float)
        js[0] = 1.0
        return js

    # Define el orden inicial para la recurrencia hacia abajo.
    Lstart = lmax + pad
    y = np.zeros(Lstart + 2, dtype=float)  # Arreglo para almacenar valores intermedios.
    y[Lstart + 1] = 0.0  # Condición inicial: y_{L+1} = 0.
    y[Lstart] = 1.0  # Condición inicial: y_L = 1.
    # Aplica la recurrencia hacia abajo para calcular y_{ell-1}.
    for ell in range(Lstart, 0, -1):
        y[ell - 1] = ((2 * ell + 1) / x) * y[ell] - y[ell + 1]
    # Normaliza los valores para que y_0 coincida con j0(x).
    factor = j0(x) / y[0]
    js = factor * y[: lmax + 1]
    return js


def j_l_miller(x, lmax, pad=60):
    """
    Calcula j_l para l=0..lmax usando el algoritmo de Miller con normalización.
    Este método es una variante optimizada de la recurrencia hacia abajo.

    Parámetros:
    - x: valor en el que se evalúan las funciones de Bessel.
    - lmax: orden máximo de las funciones de Bessel a calcular.
    - pad: número adicional de órdenes para arrancar arriba (Lstart = lmax + pad).

    Retorna:
    - js: arreglo con los valores de j_l(x) para l=0..lmax.
    """
    # Maneja el caso especial x == 0: sólo j0(0)=1, j_l(0)=0 para l>0.
    if x == 0.0:
        js = np.zeros(lmax + 1, dtype=float)
        js[0] = 1.0
        return js

    # Define el orden inicial para la recurrencia hacia abajo.
    Lstart = lmax + pad
    y = np.zeros(Lstart + 2, dtype=float)  # Arreglo para almacenar valores intermedios.
    y[Lstart + 1] = 0.0  # Condición inicial: y_{L+1} = 0.
    y[Lstart] = 1.0  # Condición inicial: y_L = 1.

    # Aplica la recurrencia hacia abajo para calcular y_{ell-1}.
    for ell in range(Lstart, 0, -1):
        y[ell - 1] = ((2 * ell + 1) / x) * y[ell] - y[ell + 1]

    # Normaliza los valores para que y_0 coincida con j0(x).
    factor = j0(x) / y[0]
    js = factor * y[: lmax + 1]
    return js


def adjust_pad_for_precision(x, lmax, target_rel=1e-10, pad_start=10, pad_max=2000):
    """
    Ajusta el valor de 'pad' para cumplir con un error relativo objetivo entre
    j_l_down(pad) y j_l_down(pad*2).
    """
    pad = pad_start
    prev = j_l_down(x, lmax, pad=pad)
    while pad <= pad_max:
        pad2 = pad * 2
        curr = j_l_down(x, lmax, pad=pad2)
        # Calcula la diferencia relativa máxima entre prev y curr.
        rels = np.abs(prev - curr) / np.maximum(
            np.maximum(np.abs(prev), np.abs(curr)), 1e-300
        )
        maxrel = np.max(rels)
        if maxrel <= target_rel:
            return pad2, curr
        pad = pad2
        prev = curr
    # Si no se cumple el criterio, retorna el último valor calculado.
    return pad, curr


# ----------------------------
# Routines de comparación y salida
# ----------------------------
def compare_methods_for_x(x, lmax=24, target_rel=1e-10):
    # Compara los métodos de cálculo hacia arriba y hacia abajo para un valor dado de x.
    print(f"---- Comparación para x = {x} ----")
    up = j_l_up(x, lmax)  # Calcula j_l usando recurrencia hacia arriba.
    # Ajusta el padding para la recurrencia hacia abajo.
    pad, down = adjust_pad_for_precision(
        x, lmax, target_rel=target_rel, pad_start=20, pad_max=20480
    )
    print(f"Usando pad = {pad} para down (criterio target_rel={target_rel})")
    # Imprime una tabla con los resultados.
    header = ("l", "j_up", "j_down", "abs_diff", "rel_diff", "metric_enunciado")
    print("{:>3s} {:>17s} {:>17s} {:>12s} {:>12s} {:>12s}".format(*header))
    for l in range(lmax + 1):
        ju = up[l]
        jd = down[l]
        absdiff = abs(ju - jd)
        denom = abs(ju) + abs(jd)
        metric = absdiff / denom if denom != 0 else 0.0
        rel = absdiff / max(abs(jd), 1e-300)
        print(
            f"{l:3d} {ju:17.12e} {jd:17.12e} {absdiff:12.4e} {rel:12.4e} {metric:12.4e}"
        )
    print("---- fin comparacion ----\n")
    return up, down


def compare_methods_with_miller(x, lmax=24, target_rel=1e-10):
    """
    Compara los métodos de cálculo hacia arriba, hacia abajo y Miller para un valor dado de x.

    Parámetros:
    - x: valor en el que se evalúan las funciones de Bessel.
    - lmax: orden máximo de las funciones de Bessel a calcular.
    - target_rel: error relativo objetivo para ajustar el padding en el método hacia abajo.

    Retorna:
    - up: valores calculados con recurrencia hacia arriba.
    - down: valores calculados con recurrencia hacia abajo.
    - miller: valores calculados con el algoritmo de Miller.
    """
    print(f"---- Comparación para x = {x} ----")
    up = j_l_up(x, lmax)  # Calcula j_l usando recurrencia hacia arriba.
    # Ajusta el padding para la recurrencia hacia abajo.
    pad, down = adjust_pad_for_precision(
        x, lmax, target_rel=target_rel, pad_start=20, pad_max=20480
    )
    print(f"Usando pad = {pad} para down (criterio target_rel={target_rel})")
    miller = j_l_miller(x, lmax, pad=pad)  # Calcula j_l usando el algoritmo de Miller.

    # Imprime una tabla con los resultados.
    header = (
        "l",
        "j_up",
        "j_down",
        "j_miller",
        "|up-down|",
        "|up-miller|",
        "|down-miller|",
    )
    print("{:<3} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15}".format(*header))
    print("-" * 105)
    for l in range(lmax + 1):
        ju = up[l]
        jd = down[l]
        jm = miller[l]
        abs_diff_up_down = abs(ju - jd)
        abs_diff_up_miller = abs(ju - jm)
        abs_diff_down_miller = abs(jd - jm)
        print(
            f"{l:<3} {ju:15.8e} {jd:15.8e} {jm:15.8e} {abs_diff_up_down:15.8e} {abs_diff_up_miller:15.8e} {abs_diff_down_miller:15.8e}"
        )
    print("-" * 105)
    print("---- fin comparacion ----\n")
    return up, down, miller


def run_all():
    demo_cancellation_examples()  # Ejecuta las demostraciones de cancelación.
    # Ejecuta todas las demostraciones y comparaciones.
    xs = [0.1, 1.0, 10.0]  # Valores de x para comparar.
    lmax = 24  # Máximo orden de j_l.
    for x in xs:
        compare_methods_with_miller(x, lmax=lmax, target_rel=1e-10)


if __name__ == "__main__":
    run_all()
