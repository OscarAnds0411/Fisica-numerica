import matplotlib.pyplot as plt # para graficas
import matplotlib.animation as anim # para animacion
from PIL import Image # para manejar imagenes
from glob import glob # para buscar archivos



fig, ax = plt.subplots(figsize=(3.6, 3.5)) # crear figura y ejes
fig.subplots_adjust(bottom=0, top=1, left=0, right=1) # ajustar margenes
ax.axis('off') # desactivar ejes

ims = [] # lista para almacenar imagenes

# cargar y agregar imagenes a la lista
for fname in sorted(glob('figura0*.png')): # buscar archivos de imagen 
    im = ax.imshow(Image.open(fname), animated=True) # abrir y mostrar imagen
    ims.append([im]) # agregar imagen a la lista

ani = anim.ArtistAnimation(fig, artists=ims, interval=33,
                              repeat=False) # crear animacion con las imagenes

fig.show() # mostrar figura
