"""
3. Ajuste de la fórmula de resonancia de Breit-Wigner.
(a) Para la teoría indica que la fórmula de Breit-Wigner debe ajustar
a los datos de los dos ejercicios anteriores:
f (E) = fr·Γ / ((E - Er)^2 + Γ^2/4)

Su problema consiste en determinar los valores para los parámetros Er, fr y Γ.
Se sugiere renombrar los parámetros haciendo
a1 = fr·Γ; a2 = ER; a3 = Γ^2/4; x = E;

para escribir
g(x) = a1 / ((x - a2)^2 + a3);

y encontrar los parámetros a partir de minimizar χ^2.

(b) Las ecuaciones que obtuvo en el inciso anterior NO son lineales,
elabore un programa que utilice el método de Newton-Raphson
multidimensional para la búsqueda de las raíces.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve
import pandas as pd

i = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
Ei = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])  # Energía (MeV)
fE = np.array(
    [10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7]
)  # Sección eficaz (mb)
sigma = np.array(
    [9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14]
)  # Incertidumbre (mb)


# -El problema consiste en encontrar el arreglo a = [a_1, a_2, a_3].
# -Definimos nuestra función g.
# -x es el punto de evaluación de g.
def g(a, x):
    return a[0] / ((x - a[1]) ** 2 + a[2])


# -h es un "termino común" en nuestras funciones a minimizar.
def h(a, x, y, sc):
    return (y - g(a, x)) / (((x - a[1]) ** 2 + a[2]) * (sc**2))


# -Definimos f_1
# -Los arreglos 'x' e 'y' corresponden a los dados experimentales.
# -El arreglo sc guarda las incertidumbres.
def f1(a, x, y, sc):
    n = min(len(x), len(y), len(sc))

    if n == 0:
        return 0

    val = 0.0
    for i in range(n):
        val += h(a, x[i], y[i], sc[i])

    return val


# Definimos f_2
# -Los arreglos 'x' e 'y' corresponden a los dados experimentales.
# -El arreglo sc guarda las incertidumbres.
def f2(a, x, y, sc):
    n = min(len(x), len(y), len(sc))

    if n == 0:
        return 0

    val = 0.0
    for i in range(n):
        val += h(a, x[i], y[i], sc[i]) * ((x[i] - a[1]) / ((x[i] - a[1]) ** 2 + a[2]))

    return val


# Definimos f_3
# -Los arreglos 'x' e 'y' corresponden a los dados experimentales.
# -El arreglo sc guarda las incertidumbres.
def f3(a, x, y, sc):
    n = min(len(x), len(y), len(sc))

    if n == 0:
        return 0

    val = 0.0
    for i in range(n):
        val += h(a, x[i], y[i], sc[i]) / ((x[i] - a[1]) ** 2 + a[2])

    return val


# -Newton-Raphson multidimensional.
# -Los arreglos 'x' e 'y' corresponden a los dados experimentales.
# -El arreglo sc guarda las incertidumbres.
# -da es un valor pequeño usado para el cálculo de derivadas parciales
#  mediante central difference.
# -epsilon el la precisión que requerimos.
def NewtonRapshon(a_0, da, x, y, sc, epsilon):
    Nmax = 1000
    a = np.array(a_0)

    Da = list()  # -Da es un arreglo de la forma
    for i in range(3):  #  Da = [[da/2.0, 0.0, 0.0]
        diffa = list()  #        [0.0, da/2.0, 0.0]
        for j in range(3):  #        [0.0, 0.0, da/2.0]]
            if j != i:  #  Este arreglo servira para
                diffa.append(0.0)  #  aproximar las derivadas parciales.
            else:
                diffa.append(da / 2.0)
        Da.append(diffa)
    Da = np.array(Da)

    F = np.array([f1(a, x, y, sc), f2(a, x, y, sc), f3(a, x, y, sc)])

    for k in range(Nmax):  # -Evitamos ciclos infinitos.
        NormaF = max(abs(F[0]), abs(F[1]), abs(F[2]))

        if NormaF <= epsilon:
            print(f"Convergencia alcanzada en {k} iteraciones.")
            return a

        DF = list()
        for i in range(3):

            # -Los siguientes 3 if's son para aproximar las derivadas
            #  parciales correspondientes.
            dFi = list()
            if i == 0:  # -Cálculo del 'gradiente' de f1.
                for j in range(3):
                    dFi.append((f1(a + Da[j], x, y, sc) - f1(a - Da[j], x, y, sc)) / da)
            if i == 1:  # -Cálculo del 'gradiente' de f2.
                for j in range(3):
                    dFi.append((f2(a + Da[j], x, y, sc) - f2(a - Da[j], x, y, sc)) / da)
            if i == 2:  # -Cálculo del 'gradiente' de f3.
                for j in range(3):
                    dFi.append((f3(a + Da[j], x, y, sc) - f3(a - Da[j], x, y, sc)) / da)
            DF.append(dFi)

        Diff_F = np.array(DF)  # -Arreglo de "derivadas parciales".

        # usamos solve() en lugar de inv()
        try:
            delta = solve(Diff_F, -F)  # Resolver J·Δa = -F
            a = a + delta  # Actualizar parámetros
        except np.linalg.LinAlgError:
            print("Matriz singular, terminando...")
            return a

        F[0] = f1(a, x, y, sc)  # -Actualizamos nuestro vector F.
        F[1] = f2(a, x, y, sc)
        F[2] = f3(a, x, y, sc)

    print("No se encontró un valor con suficiente precisión.")
    return a


# -Leemos los datos del archivo 'DatosBW.txt' y los transformamos a un
#  formato conveniente.
datos = pd.read_csv("DatosBW.txt", header=0, sep=r"\s+")
u = datos.iloc[:, 0]
v = datos.iloc[:, 1]
w = datos.iloc[:, 2]

x = u.to_numpy()
y = v.to_numpy()
sc = w.to_numpy()

# semilla inicial (estimada de los datos)
idx_max = np.argmax(y)  # Índice del máximo
E_r_inicial = x[idx_max]  # Energía donde ocurre el máximo (~75 MeV)
f_max = y[idx_max]  # Valor máximo (~83.5 mb)
Gamma_inicial = 50.0  # Estimación razonable del ancho

a_0 = np.array(
    [
        f_max * Gamma_inicial,  # a1 = f_r·Γ ≈ 83.5 * 50 = 4175
        E_r_inicial,  # a2 = E_r ≈ 75 MeV
        (Gamma_inicial**2) / 4.0,  # a3 = Γ²/4 ≈ 625
    ]
)

print("Semilla inicial:")
print("  a_1 =", a_0[0], ", a_2 =", a_0[1], ", a_3 =", a_0[2])

# -Cálculo de nuestras a's.
a = NewtonRapshon(a_0, da=0.1, x=x, y=y, sc=sc, epsilon=0.001)

print("\nResultados del ajuste:")
print("a_1 =", a[0], ", a_2 =", a[1], ", a_3 =", a[2])
print(
    "f_r =", a[0] / np.sqrt(4 * a[2]), ", E_r =", a[1], ", Gamma =", 2.0 * np.sqrt(a[2])
)

# Comparación con teoría
E_r_teorico = 78.0
Gamma_teorico = 55.0
print("\nComparación con teoría:")
print("E_r: experimental =", a[1], "MeV, teórico =", E_r_teorico, "MeV")
print(
    "Gamma: experimental =", 2.0 * np.sqrt(a[2]), "MeV, teórico =", Gamma_teorico, "MeV"
)

# -Imprimimos nuestro ajuste.
X = np.arange(0.0, 200, 0.1)
Y = list()
for i in range(len(X)):
    Y.append(g(a, X[i]))

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(x, y, "o", lw=2, color="r", markersize=8, label="Datos experimentales")
plt.plot(X, Y, "-", lw=2, color="b", label="Ajuste Breit-Wigner")
plt.axvline(
    a[1],
    color="g",
    linestyle="--",
    linewidth=2,
    label=f"E_r = {a[1]:.2f} MeV",
    alpha=0.7,
)
plt.xlabel("Energía E (MeV)", fontsize=12, fontweight="bold")
plt.ylabel("Sección eficaz σ(E) (mb)", fontsize=12, fontweight="bold")
plt.title(
    "Ajuste de Breit-Wigner usando Newton-Raphson", fontsize=14, fontweight="bold"
)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()
