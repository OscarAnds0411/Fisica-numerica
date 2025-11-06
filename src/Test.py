#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tarea2.py
Solución de la Tarea #2
(cancelación sustractiva y funciones de Bessel esféricas).
Contiene:
 - demostraciones numéricas de reescrituras que evitan cancelación
 - cálculo de j_l(x) por recurrencia hacia arriba (up)
 - cálculo de j_l(x) por recurrencia hacia abajo (down) con normalización
 - comparación y ajuste para alcanzar error relativo <= 1e-10
"""

from __future__ import annotations
import math
import numpy as np


# ----------------------------
# Utilities
# ----------------------------
def isclose_rel(a, b, tol=1e-12):
    a = float(a)
    b = float(b)
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom <= tol


# ----------------------------
# 1) Cancelación sustractiva
# ----------------------------
def sqrt_minus_one_direct(x):
    return math.sqrt(x + 1.0) - 1.0


def sqrt_minus_one_stable(x):
    return x / (math.sqrt(x + 1.0) + 1.0)


def sin_minus_sin_direct(x, y):
    return math.sin(x) - math.sin(y)


def sin_minus_sin_stable(x, y):
    return 2.0 * math.cos(0.5 * (x + y)) * math.sin(0.5 * (x - y))


def sqdiff_direct(x, y):
    return x * x - y * y


def sqdiff_stable(x, y):
    return (x - y) * (x + y)


def one_minus_cos_over_sin_direct(x):
    if x == 0.0:
        return 0.0
    return (1.0 - math.cos(x)) / math.sin(x)


def one_minus_cos_over_sin_stable(x):
    if x == 0.0:
        return 0.0
    return math.tan(0.5 * x)


def law_of_cos_direct(a, b, theta):
    return math.sqrt(a * a + b * b - 2.0 * a * b * math.cos(theta))


def law_of_cos_stable(a, b, theta):
    return math.sqrt((a - b) ** 2 + 4.0 * a * b * (math.sin(0.5 * theta) ** 2))


def demo_cancellation_examples():
    print("=== Cancelación sustractiva: comparaciones numéricas ===")

    print("(a) sqrt(x+1)-1 para valores pequeños de x")
    print("x\t\tDirecto\t\t\tEstable\t\t\tRel. Diff")
    for x in [1e-8, 1e-12, 1e-16, 1e-20, 1e-30]:
        d = sqrt_minus_one_direct(x)
        s = sqrt_minus_one_stable(x)
        rel_diff = abs(d - s) / max(abs(s), 1e-300)
        print(f"{x: .1e}\t{d:.17e}\t{s:.17e}\t{rel_diff:.2e}")
    print()

    print("(b) sin(x) - sin(y) para x ≈ y")
    print("x\ty\t\tDirecto\t\t\tEstable\t\t\tRel. Diff")
    x = 1.23456789
    for eps in [1e-6, 1e-8, 1e-12, 1e-16, 1e-20]:
        y = x + eps
        d = sin_minus_sin_direct(x, y)
        s = sin_minus_sin_stable(x, y)
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
        d = sqdiff_direct(x, y)
        s = sqdiff_stable(x, y)
        rel_diff = abs(d - s) / max(abs(s), 1e-300)
        print(f"{x:.16f}\t{y:.16f}\t{d:.17e}\t{s:.17e}\t{rel_diff:.2e}")
    print()

    print("(d) (1-cos(x))/sin(x) para x pequeño")
    print("x\t\tDirecto\t\t\tEstable\t\t\tRel. Diff")
    for x in [1e-6, 1e-8, 1e-12, 1e-16, 1e-20]:
        d = one_minus_cos_over_sin_direct(x)
        s = one_minus_cos_over_sin_stable(x)
        rel_diff = abs(d - s) / max(abs(s), 1e-300)
        print(f"{x: .1e}\t{d:.17e}\t{s:.17e}\t{rel_diff:.2e}")
    print()

    print("(e) Ley de cosenos para ángulos pequeños")
    print("a\tb\ttheta\t\tDirecto\t\t\tEstable\t\t\tRel. Diff")
    a = 1.0
    b = 0.999999999999
    for theta in [1e-3, 1e-6, 1e-8, 1e-12, 1e-16]:
        d = law_of_cos_direct(a, b, theta)
        s = law_of_cos_stable(a, b, theta)
        rel_diff = abs(d - s) / max(abs(s), 1e-300)
        print(f"{a:.1f}\t{b:.12f}\t{theta:.1e}\t" f"{d:.17e}\t{s:.17e}\t{rel_diff:.2e}")
    print("=== fin demo cancelación ===\n")


# ----------------------------
# 2) Bessel esférica j_l(x)
# ----------------------------
def j0(x):
    if x == 0.0:
        return 1.0
    return math.sin(x) / x


def j1(x):
    if x == 0.0:
        return 0.0
    return math.sin(x) / (x * x) - math.cos(x) / x


def j_l_up(x, lmax):
    js = np.zeros(lmax + 1, dtype=float)
    js[0] = j0(x)
    if lmax == 0:
        return js
    js[1] = j1(x)
    for ell in range(1, lmax):
        js[ell + 1] = ((2 * ell + 1) / x) * js[ell] - js[ell - 1]
    return js


def j_l_down(x, lmax, pad=60):
    if x == 0.0:
        js = np.zeros(lmax + 1, dtype=float)
        js[0] = 1.0
        return js

    Lstart = lmax + pad
    y = np.zeros(Lstart + 2, dtype=float)
    y[Lstart + 1] = 0.0
    y[Lstart] = 1.0
    for ell in range(Lstart, 0, -1):
        y[ell - 1] = ((2 * ell + 1) / x) * y[ell] - y[ell + 1]
    factor = j0(x) / y[0]
    js = factor * y[: lmax + 1]
    return js


def j_l_miller(x, lmax, pad=60):
    if x == 0.0:
        js = np.zeros(lmax + 1, dtype=float)
        js[0] = 1.0
        return js

    Lstart = lmax + pad
    y = np.zeros(Lstart + 2, dtype=float)
    y[Lstart + 1] = 0.0
    y[Lstart] = 1.0
    for ell in range(Lstart, 0, -1):
        y[ell - 1] = ((2 * ell + 1) / x) * y[ell] - y[ell + 1]
    factor = j0(x) / y[0]
    js = factor * y[: lmax + 1]
    return js


def adjust_pad_for_precision(x, lmax, target_rel=1e-10, pad_start=10, pad_max=2000):
    pad = pad_start
    prev = j_l_down(x, lmax, pad=pad)
    while pad <= pad_max:
        pad2 = pad * 2
        curr = j_l_down(x, lmax, pad=pad2)
        rels = np.abs(prev - curr) / np.maximum(
            np.maximum(np.abs(prev), np.abs(curr)), 1e-300
        )
        maxrel = np.max(rels)
        if maxrel <= target_rel:
            return pad2, curr
        pad = pad2
        prev = curr
    return pad, curr


# ----------------------------
# Comparación y salida
# ----------------------------
def compare_methods_for_x(x, lmax=24, target_rel=1e-10):
    print(f"---- Comparación para x = {x} ----")
    up = j_l_up(x, lmax)
    pad, down = adjust_pad_for_precision(
        x, lmax, target_rel=target_rel, pad_start=20, pad_max=20480
    )
    print(f"Usando pad = {pad} (criterio target_rel={target_rel})")
    header = ("l", "j_up", "j_down", "abs_diff", "rel_diff", "metric")
    print("{:>3s} {:>17s} {:>17s} {:>12s} {:>12s} {:>12s}".format(*header))
    for ell in range(lmax + 1):
        ju = up[ell]
        jd = down[ell]
        absdiff = abs(ju - jd)
        denom = abs(ju) + abs(jd)
        metric = absdiff / denom if denom != 0 else 0.0
        rel = absdiff / max(abs(jd), 1e-300)
        print(
            f"{ell:3d} {ju:17.12e} {jd:17.12e} "
            f"{absdiff:12.4e} {rel:12.4e} {metric:12.4e}"
        )
    print("---- fin comparacion ----\n")
    return up, down


def compare_methods_with_miller(x, lmax=24, target_rel=1e-10):
    print(f"---- Comparación para x = {x} ----")
    up = j_l_up(x, lmax)
    pad, down = adjust_pad_for_precision(
        x, lmax, target_rel=target_rel, pad_start=20, pad_max=20480
    )
    print(f"Usando pad = {pad} (criterio target_rel={target_rel})")
    miller = j_l_miller(x, lmax, pad=pad)

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
    for ell in range(lmax + 1):
        ju, jd, jm = up[ell], down[ell], miller[ell]
        print(
            f"{ell:<3} {ju:15.8e} {jd:15.8e} {jm:15.8e} "
            f"{abs(ju - jd):15.8e} {abs(ju - jm):15.8e} "
            f"{abs(jd - jm):15.8e}"
        )
    print("-" * 105)
    print("---- fin comparacion ----\n")
    return up, down, miller


def run_all():
    demo_cancellation_examples()
    xs = [0.1, 1.0, 10.0]
    lmax = 24
    for x in xs:
        compare_methods_with_miller(x, lmax=lmax, target_rel=1e-10)


if __name__ == "__main__":
    run_all()
