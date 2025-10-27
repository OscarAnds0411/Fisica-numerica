"""
Lanzamiento de martillo. El record mundial para hombres en lanzamiento de martillo es de 86.74 m
por Yuri Sedykh y se ha mantenido desde 1986. El martillo pesa 7:26 kg, es esférico,
y tiene un radio de R = 6 cm. 

La fricciÛn en el martillo puede ser considerada proporcional
al cuadrado de la velocidad del martillo relativa al aire:
F_D =\frac{1}{2}\rho AC_Dv^2

donde \rho es la densidad del aire (1.2 kg/m^3 ) y A = \pi R^2 es la sección transver-
sal del martillo. 

El martillo puede experimentar, en principio, un flujo laminar con coeficiente de rozamiento CD = 0.5 
o un flujo inestable oscilante con CD = 0.75.

(a) Resuelva la ecuación de movimiento para el lanzamiento oblicuo
de martillo. Deberá transformar las EDOs para los moviemtos
en x y y en un sistema de cuatro ecuaciones de primer orden.
Considere lanzamientos desde una posición inicial x0 = 0 y y0 = 2
m, para un ángulo ideal \theta = 45 y encuentre la velocidad que
produce la distancia del lanzamiento del record mundial.

(b) Calcule y grafique la dependencia en el tiempo de la altitud del
martillo y su trayectoria y = y (x) en los tres régimenes:
i. Sin fricción
ii. Flujo laminar
iii. Flujo inestable oscilante

(c) En el inciso anterior, estime la cantidad en que es influenciada la
distancia del lanzamiento por la fricción.
"""
from pylab import *
import numpy as np
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
import os

# Parámetros físicos
g = 9.81  # Aceleración debido a la gravedad (m/s^2)
rho = 1.2  # Densidad del aire (kg/m^3) 
R = 0.06  # Radio del martillo (m)
A = np.pi * R**2  # Área de sección transversal del martillo (m^2)
m = 7.26  # Masa del martillo (kg)
record_distance = 86.74  # Distancia del récord mundial (m)
theta = np.radians(45)  # Ángulo de lanzamiento (radianes)
x0, y0 = 0,  2  # Posición inicial (m)
it_max = 500  # Número máximo de iteraciones
dt = 0.01  # Paso de tiempo (s)
N= 500  # Número de pasos de tiempo
tol=1e-3  # Tolerancia para la convergencia
drag_coeffs = [0.0, 0.5, 0.75]  # Coeficientes de arrastre para los tres regímenes

# Crear carpeta para resultados
output_dir = 'resultados_martillo' # Carpeta para guardar resultados
if not os.path.exists(output_dir): # Crear carpeta si no existe
    os.makedirs(output_dir) # Crear carpeta si no existe


#definición de las ecuaciones de movimiento (EDOs)
def equations_of_motion(state, t, k):
    """Devuelve las derivadas de las variables de estado."""
    f0= state[1]
    f1= -k/m*state[1]*np.sqrt(state[1]**2 + state[3]**2)    
    f2= state[3]
    f3 = -g - k/m*state[3]*np.sqrt(state[1]**2 + state[3]**2)
    return array([f0, f1, f2, f3])

#buscamos la distancia alcanzada para una velocidad inicial dada
def distance_reached(initial, v0, k):
    """Calcula la distancia alcanzada para una velocidad inicial dada y coeficiente de arrastre k."""
    #como theta = 45 grados, las componentes x e y de la velocidad inicial son iguales
    v= v0 * np.sin(pi/4)
    finaltime = 10.0 # Tiempo final para la simulación
    r0= array([initial[0],v , initial[1], v])  # Estado inicial: [x0, vx0, y0, vy0]
    r=r0 # variable para almacenar el estado actual
    t = linspace(0, it_max * dt, N)  # Vector de tiempo
    groundtime= 0.0

    #inicia lo dificil, jugar a adivinar con algo numerico
    s=0
    while s < it_max:
        # Integración numérica de las EDOs
        sol = odeint(equations_of_motion, r, t, args=(k,))
        n=len(sol)-1
        # Verificar si el martillo no ha tocado el suelo
        if sol[n, 2] > 0:
            finaltime += 1.0
            t= linspace(0., finaltime, N)
        else:
           for j in range(n):
               # verificamos si la fisica nos falla y fuimos capaces de atravesar el suelo
               if sol[j, 2] <= 0:
                   groundtime += t[j-1]-t[0]
                   #checamos la tolerancia
                   if abs(sol[j,2])<= tol/2.:
                       # buscamos que en caso de ser preciso, reacemos todo pero usando el tiempo para que toque el suelo.
                       t = linspace(0, groundtime+t[j]-t[j-1], N*50*s)
                       sol = odeint(equations_of_motion, r0, t, args=(k,))
                       n=len(sol)-1
                       return (sol[n,0]+sol[n-1,0])/2., groundtime
                   else: #en caso de no ser preciso, resolvemos la ED con los valores inciales, que seran las condiciones que el martillo llevaba antes de tocar el piso.
                        r = array([sol[j-1][0], sol[j-1][1],    
                                    sol[j-1][2], sol[j-1][3]])
                        t=linspace(t[j-1], t[j], 50)
                        break
        s += 1
    print("Distancia recorrida antes de llegar al suelo no encontrada\
          dentro de las iteraciones permitidas. Regresamos 0.")
    return 0.,0.
                       
                 
# nos interesa hallar la velocidad inicial que produce la distancia del récord mundial
def find_initial_velocity(distance, initial, v0, k):
    """Encuentra la velocidad inicial que produce la distancia del récord mundial."""
    f= lambda v: distance_reached(initial, v, k)[0]- distance
    dv= 1.0e-3

    i=0

    #vamos a hacer Newton-Raphson para encontrar la velocidad inicial
    while i < it_max:
        fv = f(v0)
        if abs(fv) <= tol:
            return v0
        # Derivada numérica
        df = (f(v0 + dv/2.) - f(v0 - dv/2.)) / dv
        dv= -fv/df
        v0 += dv

        i += 1
    print("Velocidad inicial no encontrada dentro de las iteraciones permitidas.\
          Regresamos 0.")
    return 0.
#Tambien nos interesa la trayectoria :
def get_full_trajectory(initial, v0, k, tground):
    """Obtiene la trayectoria completa para graficar."""
    v = v0 * np.sin(np.pi/4)
    r0 = array([initial[0], v, initial[1], v])
    
    if tground == 0.:
        tground = 5.0
    
    t = linspace(0., tground, 500)
    sol = odeint(equations_of_motion, r0, t, args=(k,))
    
    return sol, t
#imprimmos las condiciones iniciales
print("\nCondiciones iniciales:")
print(f"  x0 = {x0} m, y0 = {y0} m, θ = 45°")
print("\nEncontrando velocidad inicial para alcanzar récord mundial...")
print("="*70)

v_aux = [] # Lista para almacenar velocidades iniciales
for i, cd in enumerate(drag_coeffs):
    print(f"  * Régimen {i+1} (C_D = {cd}):")
    k = rho * A * cd / 2.0
    v0_record = find_initial_velocity(record_distance, array([x0, y0]), 28.0, k)
    v_aux.append(v0_record) # Agregar velocidad inicial encontrada a la lista --- nos serviran más adelante
    if v0_record == 0.:
        print("ERROR: No se pudo calcular la velocidad inicial.")
        exit(1)
    else:
        print(f"Velocidad inicial necesaria: v0 = {v0_record:.2f} m/s")
        print(f"(Para alcanzar {record_distance} m con C_D = {cd})")
        print("="*70)

print("\n" + "="*70)
print("TRAYECTORIAS EN LOS TRES REGÍMENES")
print("="*70)

solutions = [] # Almacenar soluciones para cada régimen
times_list = [] # Almacenar tiempos para cada régimen
distances_list = [] # Almacenar distancias alcanzadas
labels = ['Sin fricción', 'Flujo laminar', 'Flujo inestable oscilante'] # Etiquetas para los regímenes
cd_labels = ['C_D = 0.0', 'C_D = 0.5', 'C_D = 0.75'] # Etiquetas para los coeficientes de arrastre
colors = ['#2E86AB', '#A23B72', '#F18F01'] # Colores para las gráficas

print("\nCalculando trayectorias...")
for i, cd in enumerate(drag_coeffs): # Iterar sobre los coeficientes de arrastre
    k = rho * A * cd / 2.0
    distance, time = distance_reached(array([x0, y0]), v_aux[i], k) # Calcular distancia alcanzada
    distances_list.append(distance)     # Almacenar distancia alcanzada
    
    print(f"\n{labels[i]} ({cd_labels[i]}):")
    print(f"  Distancia alcanzada: {distance:.2f} m") # Almacenar distancia alcanzada
    print(f"  Tiempo de vuelo: {time:.2f} s") # Almacenar tiempo de vuelo
    print(f"  Velocidad inicial: v0 = {v_aux[i]:.2f} m/s")

    sol, t = get_full_trajectory(array([x0, y0]), v_aux[i], k, time) # Obtener trayectoria completa
    solutions.append(sol) # Almacenar solución
    times_list.append(t) # Almacenar tiempos
print("="*70)
print("Calculo de las trayectorias con la velocidad inicial dada con C_D = 0.0: v0 =",v_aux[0])
print("\nCalculando trayectorias...")
sols=[] # Almacenar soluciones para cada régimen
tims=[] # Almacenar tiempos para cada régimen
dists=[] # Almacenar distancias alcanzadas
for i, cd in enumerate(drag_coeffs): # Iterar sobre los coeficientes de arrastre
    k = rho * A * cd / 2.0
    distance, time = distance_reached(array([x0, y0]), v_aux[0], k) # Calcular distancia alcanzada
    dists.append(distance)     # Almacenar distancia alcanzada
    
    print(f"\n{labels[i]} ({cd_labels[i]}):")
    print(f"  Distancia alcanzada: {distance:.2f} m") # Almacenar distancia alcanzada
    print(f"  Tiempo de vuelo: {time:.2f} s") # Almacenar tiempo de vuelo

    sol, t = get_full_trajectory(array([x0, y0]), v_aux[0], k, time) # Obtener trayectoria completa
    sols.append(sol) # Almacenar solución
    tims.append(t) # Almacenar tiempos
print("="*70)
print(" GENERANDO GRÁFICAS POR RÉGIMEN...")
print("="*70)

for i, (sol, t, cd, label, cd_label, color, distance) in enumerate(
    zip(solutions, times_list, drag_coeffs, labels, cd_labels, colors, distances_list)):
    
    # Crear figura con 2 subplots
    fig, (ax1, ax2) = subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{label}\n{cd_label} | v₀ = {v_aux[i]:.2f} m/s', 
                 fontsize=15, fontweight='bold')
    
    # Subplot 1: Trayectoria y = y(x)
    ax1.plot(sol[:, 0], sol[:, 2], color=color, linewidth=3, label='Trayectoria')
    ax1.axhline(y=0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)
    ax1.set_xlabel('Distancia horizontal x (m)', fontsize=12)
    ax1.set_ylabel('Altura y (m)', fontsize=12)
    ax1.set_title('Trayectoria y = y(x)', fontsize=13, fontweight='bold')
    ax1.text(0.98, 0.95, f'Alcance: {distance:.2f} m', 
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    ax1.grid(True, alpha=0.4)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # Subplot 2: Altura vs tiempo y = y(t)
    ax2.plot(t, sol[:, 2], color=color, linewidth=3, label='Altura')
    ax2.axhline(y=0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Tiempo t (s)', fontsize=12)
    ax2.set_ylabel('Altura y (m)', fontsize=12)
    ax2.set_title('Dependencia temporal y = y(t)', fontsize=13, fontweight='bold')
    ax2.text(0.98, 0.95, f'Tiempo: {t[-1]:.2f} s', 
             transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85))
    ax2.grid(True, alpha=0.4)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    tight_layout()
    filename = f'{output_dir}/regimen_{i+1}_CD_{cd:.2f}.png'
    savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Guardado: {filename}")
    show()
    close(fig)

print("\n" + "="*70)
print("INFLUENCIA DE LA FRICCIÓN")
print("="*70)

print("\n" + "-"*70)
print(f"{'Régimen':<35} {'Distancia (m)':<15} {'Pérdida (m)':<15} {'Pérdida (%)'}")
print("-"*70)

losses_m = []
losses_pct = []

for cd, dist, label in zip(drag_coeffs, dists, labels):
    if dist > 0:
        loss_m = record_distance - dist
        loss_pct = (loss_m / record_distance) * 100
        losses_m.append(loss_m)
        losses_pct.append(loss_pct)
        
        print(f"{label:<35} {dist:<15.2f} {loss_m:<15.2f} {loss_pct:>6.2f}%")
    else:
        losses_m.append(0)
        losses_pct.append(0)
        print(f"{label:<35} {'N/A':<15} {'N/A':<15} {'N/A'}")

print("-"*70)

# Resumen final
print("\n" + "="*70)
print(" RESUMEN DE RESULTADOS")
print("="*70)
print(f"\nVelocidad inicial (sin fricción): v0 = {v0_record:.2f} m/s")
print(f"Distancia récord objetivo: {record_distance:.2f} m\n")

if losses_m[1] > 0:
    print(f"Flujo laminar (C_D = 0.5):")
    print(f"  ->Reduce {losses_m[1]:.2f} m ({losses_pct[1]:.1f}%)")
    print(f"  -> Alcance: {distances_list[1]:.2f} m\n")

if losses_m[2] > 0:
    print(f"Flujo inestable oscilante (C_D = 0.75):")
    print(f"  -> Reduce {losses_m[2]:.2f} m ({losses_pct[2]:.1f}%)")
    print(f"  -> Alcance: {distances_list[2]:.2f} m\n")

print("CONCLUSIÓN:")
print(f"La fricción del aire reduce el alcance hasta en {max(losses_pct):.1f}%,")
print(f"lo que representa una pérdida máxima de {max(losses_m):.2f} metros.")

print("\n" + "="*70)
print(f" PROCESO COMPLETADO")
print("="*70)
print(f"\nArchivos generados en '{output_dir}/':")
print("  * regimen_1_CD_0.00.png - Sin fricción")
print("  * regimen_2_CD_0.50.png - Flujo laminar")
print("  * regimen_3_CD_0.75.png - Flujo inestable oscilante")
print("="*70 + "\n")