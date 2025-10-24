import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# definimos la funcion de onda 
def ww(x,t,k0,a0,vp,vg):
    """Calcula la funcion de onda en la posicion x y tiempo t."""
    tc=a0*a0+1j*(0.5*vg/k0)*t
    u=np.exp(1.0j*k0*(x-vp*t)-0.25*(x-vg*t)**2/tc)
    return np.real(u/np.sqrt(tc))

# parametros de la onda
wavelength = 1.0   # longitud de onda
a0 = 1.0             # ancho inicial del paquete de onda
k0 = 2*np.pi/wavelength # numero de onda
vp, vg = 5.0, 10.0 # velocidades de fase y de grupo
period = wavelength/vp # periodo de la onda
runtime = 40*period # tiempo total para seguir la onda
rundistance = 0.6*vg*runtime # distancia total para graficar la onda
dt = period/6.0 # tiempo entre fotogramas
tsteps = int(runtime/dt) # numero total de veces que se calcula la forma de la onda

print('Frame time interval = {0:0.3g} ms'.format(1000*dt)) # intervalo de tiempo entre fotogramas
print('Frame rate = {0:0.3g} frames/s'.format(1.0/dt)) # tasa de fotogramas    

fig, ax = plt.subplots(figsize=(12, 3)) # creamos la figura y los ejes
fig.subplots_adjust(bottom=0.2) # permitir espacio para la etiqueta del eje
x = np.arange(-5*a0, rundistance, wavelength/20.0) # definimos el rango de x
line,= ax.plot(x,np.ma.array(x, mask=True), color='r') # inicializamos la linea de la grafica
ax.set_xlabel(r'$x$') # etiqueta del eje x
ax.set_ylabel(r'$y(x,t)$') # etiqueta del eje y
ax.set_xlim(-5*a0, rundistance) # limites del eje x
ax.set_ylim(-1.05, 1.05) # limites del eje y

# funcion de animacion
def animate(i):
    """Actualiza los datos de y para cada fotograma de la animacion."""
    t = float(i)*dt # tiempo actual
    line.set_ydata(ww(x, t, k0, a0, vp, vg)) # actualizar datos de y
    # retornamos la linea actualizada
    return line, 


ani = anim.FuncAnimation(fig, func=animate,
                         frames=range(tsteps),interval=1000*dt,
                         blit=True) # creamos la animacion
# Descomente para guardar como archivo de pelicula mp4. Necesita ffmpeg.
# ani.save('wavepacket.mp4', writer='ffmpeg')
fig.show() # mostramos la figura