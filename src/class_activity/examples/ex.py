from glob import glob

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# =============================================
# PARTE 1: GENERAR LAS IMÁGENES DE LA ONDA SENO
# =============================================

# Parámetros de la onda
k = 2 * np.pi  # número de onda
ω = 3 * np.pi  # frecuencia angular

# Definir intervalo para x (de 0 a 4π)
x = np.linspace(0, 4 * np.pi, 1000)

# Generar 10 gráficas con diferentes tiempos
for i in range(300):
    t = i * 0.1  # tiempo actual con paso de 0.1

    # Calcular la función f(x,t) = sen(kx + ωt)
    f = np.sin(k * x + ω * t)

    # Crear figura
    plt.figure(figsize=(8, 4))
    plt.plot(x, f, "b-", linewidth=2)
    plt.title(f"Onda senoidal: f(x,t) = sen(kx + ωt)\n t = {t:.1f} s")
    plt.xlabel("x")
    plt.ylabel("f(x,t)")
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.5, 1.5)

    # Guardar figura con nombre consecutivo
    filename = f"figura{i+1:03d}.png"  # Nombres: figura001.png, figura002.png, etc.
    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close()

print("10 figuras generadas: figura001.png a figura010.png")

# =============================================
# PARTE 2: ANIMAR LAS FIGURAS GENERADAS
# =============================================

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

# Crear la animación usando ArtistAnimation
# fig: la figura donde se mostrará la animación
# artists=ims: lista de frames (cada frame es una lista de artistas)
# interval=33: tiempo entre frames en milisegundos (≈30 fps)
# repeat=False: la animación no se repetirá automáticamente
ani = anim.ArtistAnimation(fig, artists=ims, interval=33, repeat=False)

# Mostrar la figura con la animación
plt.show()

# OPCIONAL: Guardar la animación como archivo GIF
# ani.save('animacion_onda.gif', writer='pillow', fps=30)
