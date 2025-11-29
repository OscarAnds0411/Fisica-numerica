"""
Coded: 06/11/25
Oscar Valencia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def hubble_function(r, a, H):
    return a + H * r


data = pd.read_csv("HUBBLE.txt", header=0, delim_whitespace=True)
df = pd.DataFrame(data)


ob = data.iloc[:, 0]
r = data.iloc[:, 1]
v = data.iloc[:, 2]

plt.scatter(r, v, color="r", label="Datos del Hubble")
plt.title("Datos originales de Hubble (1929)")
plt.xlabel("Distancia (Mpc)")
plt.ylabel("Velocidad radial (km/s)")
plt.grid(True)
plt.legend()
plt.show()

r_aux = r.to_numpy()
v_aux = v.to_numpy()
n = len(r_aux)
a = (sum(r * r) * sum(v) - sum(r) * sum(r * v)) / (n * sum(r * r) - (sum(r)) ** 2)
H = (n * sum(r * v) - sum(r) * sum(v)) / (n * sum(r * r) - (sum(r)) ** 2)
r_aj = np.linspace(0, 2.2, 100)
v_aj = hubble_function(r_aj, a, H)

print(f"Constante de Hubble (H) = {H:.2f} km/s/Mpc")
print(f"Intercepto (a) = {a:.2f}")

plt.scatter(r, v, color="g", label="Datos observados")
plt.plot(r_aj, v_aj, color="r", label=f"Ajuste lineal: v = {a:.1f}+{H:.1f}r")
plt.title("Ajuste de la Ley de Hubble (1929)")
plt.xlabel("Distancia (Mpc)")
plt.ylabel("Velocidad (km/s)")
plt.grid(True)
plt.legend()
plt.show()

v_pred = hubble_function(r, a, H)
residuos = v - v_pred
varianza = np.var(residuos, ddof=2)
desv_est = np.std(residuos, ddof=2)

print(f"Varianza de los residuos = {varianza:.2f}")
print(f"Desviación estándar de los residuos = {desv_est:.2f}")

# Supongamos un error del 10% en la velocidad:
sigma = 0.10 * v.abs()
chi2 = np.sum(((v - v_pred) / sigma) ** 2)
ndof = n - 2  # grados de libertad

print(f"Chi-cuadrado (χ²) = {chi2:.2f}")
print(f"χ² reducido = {chi2 / ndof:.2f}")
