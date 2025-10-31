"""
Lea el archivo de datos COBE.txt utilizando las instrucciones:
import pandas as pd
# Importamos los datos de un archivo txt
data = pd.read_csv(íCOBE.txtí,header=0,delim_whitespace = True)
# Recopilamos los datos de las tres columnas
nu= data.iloc[:,0]
I= data.iloc[:,1]
sigma=data.iloc[:,2]
Con los arreglos anteriores, graÖque la intensidad (I) contra la frecuencia
(nu) y la incertidumbre para cada punto.
"""

import pandas as pd
import matplotlib.pyplot as plt

# no se usar pandas
data = pd.read_csv("COBE.txt", header=0, delim_whitespace=True)

# extraemos los datos

nu = data.iloc[:, 0]
I = data.iloc[:, 1]
sigma = data.iloc[:, 2]


# Mostrar una vista previa para verificar
print(data.head())


# Graficar Intensidad vs Frecuencia

plt.figure(figsize=(9, 6))
plt.errorbar(
    nu,
    I,
    yerr=sigma,
    fmt="o",
    markersize=6,
    color="navy",
    ecolor="deepskyblue",
    elinewidth=2,  # grosor de las líneas de error
    capsize=6,  # tamaño de las "tapitas" en los extremos
    capthick=2,  # grosor de las tapitas
    alpha=0.9,
    label="Datos experimentales (COBE)",
)
plt.xlabel("Frecuencia ν")
plt.ylabel("Intensidad I")
plt.title("Mediciones del espectro del CMB (Datos COBE)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
