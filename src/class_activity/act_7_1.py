"""
Considere la funciÛn

f(x; t) = sen(kx + wt)

Elija un intervalo adecuado para x y graÖque la funciÛn para diez val-
ores de t, separados por un paso de 0:1. Guarde cada gr·Öca en for-
mato .png y con nombres consecutivos haciendo uso de la instrucciÛn
plt.savefig(ífigura001.pngí). ModiÖque el programa animacionFIGURAS.py
para animar las diez gr·Öcas generadas. Coloque comentarios en el pro-
grama donde se indique quÈ hace cada instrucciÛn.
"""

from glob import glob

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Parámetros de la onda
k = 2 * np.pi  # número de onda 2 pi
ω = 3 * np.pi  # frecuencia angular

# Definir intervalo para x de 0 a 10 con 1000 puntos
x = np.linspace(0, 4 * np.pi, 10000)

# Generar 300 gráficas con diferentes tiempos
for i in range(50):
    t = i * 0.1  # tiempo actual con paso de 0.1

    # Calcular la función f(x,t) = sen(kx + ωt)
    f = np.sin(k * x + ω * t)

    # Crear figura
    plt.figure(figsize=(8, 4))  # tamaño de la figura en pulgadas
    plt.plot(x, f, "b-", linewidth=2)  # graficar f vs x
    plt.title(
        f"Onda senoidal: f(x,t) = sen(kx + ωt)\n t = {t:.1f} s"
    )  # título con tiempo actual
    plt.xlabel("x")  # etiqueta del eje x
    plt.ylabel("f(x,t)")  # etiqueta del eje y
    plt.grid(True, alpha=0.3)  # agregar cuadrícula con transparencia
    plt.ylim(-1.5, 1.5)  # límites del eje y

    # Guardar figura con nombre consecutivo
    filename = f"figura{i+1:03d}.png"  # Nombres: figura001.png, figura002.png, etc.
    plt.savefig(filename, dpi=100, bbox_inches="tight")  # guardar figura como PNG
    plt.close()  # cerrar la figura para liberar memoria

print("300 figuras generadas: figura001.png a figura300.png")

# Crear figura y ejes para la animación
fig, ax = plt.subplots(figsize=(3.6, 3.5))

# Ajustar los márgenes de la figura para que no haya espacios en blanco
fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

# Desactivar los ejes (no mostrar marcas de ejes ni bordes)
ax.axis("off")

# Lista para almacenar todos los frames de la animación
ims = []

# Obtener la lista de archivos PNG en orden alfabético
# glob('figura0*.png') busca todos los archivos que empiezan con 'figura0' y terminan con '.png'
for fname in sorted(glob("figura0*.png")):
    # Abrir la imagen usando PIL (Python Imaging Library)
    img = Image.open(fname)

    # Mostrar la imagen en los ejes
    # animated=True marca este artista para ser usado en animaciones
    im = ax.imshow(img, animated=True)

    # Agregar la imagen a la lista de frames
    # Cada frame debe ser una lista de artistas (en este caso, solo una imagen)
    ims.append([im])

ani = anim.ArtistAnimation(
    fig, artists=ims, interval=33, repeat=True
)  # Crear la animación con un intervalo de 40 ms entre frames

# Mostrar la figura con la animación
plt.show()

# OPCIONAL: Guardar la animación como archivo GIF
# ani.save('animacion_onda.gif', writer='pillow', fps=30)
