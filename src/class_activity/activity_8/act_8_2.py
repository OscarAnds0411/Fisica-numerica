""" "
coded: 30/10/25 by: Oscar Valencia
(a) Elabore un programa que guarde en un arreglo, por separado, cada
uno de los siguientes par·metros:
i = 1 2 3 4 5 6 7 8 9
Ei (MeV ) 0 25 50 75 100 125 150 175 200
f (Ei) (MeV ) 10:6 16:0 45:0 83:5 52:8 19:9 10:8 8:25 4:7
i (MeV ) 9:34 17:9 41:5 85:5 51:5 21:5 10:8 6:29 4:14
(b) GraÖque la secciÛn eÖcaz f(Ei) contra la energÌa.
(c) Revise cÛmo graÖcar las barras de error para cada punto y elabore
una gr·Öca de la secciÛn eÖcaz f(Ei) contra la energÌa que incluya
la incertidumbre i para cada punto.
(d) La teorÌa indica que la fÛrmula de Breit-Wigner debe ajustar a los
datos:

f (E) = fr
(E
Er)
2 +
2=4
:

En la gr·Öca anterior, dibuje tambiÈn la curva con par·metros
(Er;
) = (78 MeV; 55 MeV ).
"""

import numpy as np
import matplotlib.pyplot as plt

# Datos experimentales en un array
i = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
Ei = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])  # Energía (MeV)
fE = np.array(
    [10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7]
)  # Sección eficaz (MeV)
sigma = np.array(
    [9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14]
)  # Incertidumbre (MeV)

#  Gráfica f(Ei) vs Ei

plt.figure(figsize=(8, 6))
plt.plot(Ei, fE, "o-", color="blue", label="Datos experimentales")
plt.xlabel("Energía $E_i$ (MeV)")
plt.ylabel("Sección eficaz $f(E_i)$ (MeV)")
plt.title("Sección eficaz experimental")
plt.grid(True)
plt.legend()
plt.show()


#  Gráfica con barras de error

plt.figure(figsize=(8, 6))
plt.errorbar(
    Ei, fE, yerr=sigma, fmt="o", capsize=5, color="red", label="Datos con incertidumbre"
)
plt.xlabel("Energía $E_i$ (MeV)")
plt.ylabel("Sección eficaz $f(E_i)$ (MeV)")
plt.title("Sección eficaz con barras de error")
plt.grid(True)
plt.legend()
plt.show()

#  Ajuste teórico de Breit–Wigner


def breit_wigner(E, fr, Er, Gamma):
    """
    Fórmula de Breit-Wigner:
    f(E) = fr / [ (E - Er)^2 + (Gamma^2)/4 ]
    """
    return fr / ((E - Er) ** 2 + (Gamma**2) / 4)


# Parámetros dados
Er = 78  # MeV
Gamma = 55  # MeV
fr = 1e5  # Factor de escala (ajustable para visualizar bien la curva)

# Rango continuo de energía
E_cont = np.linspace(0, 200, 400)
f_teo = breit_wigner(E_cont, fr, Er, Gamma)

# Gráfica final con datos + error + teoría
plt.figure(figsize=(9, 6))
plt.errorbar(
    Ei, fE, yerr=sigma, fmt="o", capsize=5, color="black", label="Datos experimentales"
)
plt.plot(
    E_cont, f_teo, "-", color="blue", label="Ajuste Breit–Wigner (Er=78 MeV, Γ=55 MeV)"
)
plt.xlabel("Energía $E$ (MeV)")
plt.ylabel("Sección eficaz $f(E)$ (MeV)")
plt.title("Ajuste de datos experimentales con la fórmula de Breit–Wigner")
plt.legend()
plt.grid(True)
plt.show()
