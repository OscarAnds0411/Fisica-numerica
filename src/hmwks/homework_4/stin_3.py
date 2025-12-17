"""
VIBRACIÓN DE UNA CUERDA - ECUACIÓN DE ONDA
Solución numérica de la ecuación de onda 1D usando diferencias finitas.
Analiza la estabilidad según la condición de Courant-Friedrichs-Lewy (CFL).

Autor: Oscar Andrés
Proyecto: Física Numérica

Advertencia: Solo Dios y mi yo del pasado saben porqué hice cada cosa de este codigo,
puede que en el momento me haya parecido una graniosa idea
y que ahorita que trate de explicarlo no tenga sentido tanta vuelta, en fin al final del día FUNCIONA
"""

import os  # manejo de archivos
from typing import (  # Callable: indicar que un parametro es una funcion; Tuple: indica que un funcion devuelve una tuple y de que tipo
    Callable,
    Tuple,
)

import matplotlib.animation as animation  # animaciones de graficas
import matplotlib.pyplot as plt  # plots
import numpy as np  # arreglos (num.python)
from matplotlib.animation import PillowWriter  # permite guardar mi animacion en un GIF

# Si bien no hemos visto nada de esto, quiero practicar para ya ponerme a trabajar
# por lo que se me hizo mucho mas facil construir una clase para ya no repetir codigo


# ¿Porque una clase?
# Respuesta corta: Para más facilidad de replicar experimentos sin necesidad de reescribir codigo x cada simulacion
class CuerdaVibrante:
    """
    Simulación de la vibración de una cuerda usando el método de diferencias finitas.

    Resuelve la ecuación de onda:
        ∂²y/∂x² = (1/c²) ∂²y/∂t²

    con condiciones de frontera:
        y(0, t) = y(L, t) = 0

    y condiciones iniciales:
        y(x, 0) = f(x)
        ∂y/∂t(x, 0) = g(x)

    Parametros
    ----------
    L : float
        Longitud de la cuerda (m)
    c : float
        Velocidad de onda (m/s), c = √(T/ρ)
    T_max : float
        Tiempo total de simulación (s)
    Nx : int
        Número de puntos espaciales
    Courant : float
        Número de Courant r = c·Δt/Δx (debe ser ≤ 1 para estabilidad)
    """

    # constructor: ¿Con que se come? funcion reservada que se ejecuta con cada nueva instancia creada de nuestra clase
    # self: objeto para decir ESTO en especifico de ESTA sola clase, es como ''darle un CURP a variables de nuestras clases'' (o acada cuerda)
    # No perder de vista que se buscan hacer varios experimentos :D
    # ejemplo: cuerda1.resolver() === CuerdaVibrante.resolver(cuerda1)
    # Visto como librerias, por asi decirlo ----
    # Por que es necesario? Sin self, Python no sabría: ¿Qué Nt usar? (¿el de cuerda1 o cuerda2?)
    # ¿Qué y actualizar? (¿la matriz de cuerda1 o cuerda2?)
    def __init__(self, L: float, c: float, T_max: float, Nx: int, Courant: float):
        """
        __init__ : palabra reservada para el constructor de mi clase

        ¿Que se busca que haga? hacer los setters (metodos para encapsular datos) de cada parametro de nuestra clase:
        L, c, T_max, Nx, etc.

        Discretizar el espacio:
        Divide la longitud de la cuerda en partes iguales,
        Linspace genera una malla de posiciones de 0 a L con Nx puntos :D

        Discretizacion temporal segun Dios Courant: ¿Pa que? pues la animacion se trata de mandar y juntar varios instantes de un movimiento
        (fotogramas), por lo que esto calcula los tiempos para cada fotograma según Courant: r=(c⋅Δt​)/Δx
        Calcula cuánto tiempo debe pasar entre cada "fotograma" de la simulación, basado en la condición de Courant,
        Calcular número de pasos temporales
        Linspace genera una malla de tiempos de 0 a (dt * Nt) con Nt puntos :D

        Parámetro de estabilidad:
        Calcula el cuadrado del número de Courant, que aparece en la fórmula de diferencias finitas.
        ¿Por qué se guarda r^2? Porque en el algoritmo de diferencias finitas lo usamos, no hay más

        Matrices de solución:
        ¿Qué hace? Crea una matriz de ceros donde guardaremos toda la solución.
            Dimensiones:
                Filas (Nx): Puntos espaciales (posiciones)
                Columnas (Nt): Pasos temporales (fotogramas)

        Información: Como su nombre dice, solo imprime toda la información de nuestra cuerda, segun los parametros dados :D
        """
        self.L = L
        self.c = c
        self.T_max = T_max
        self.Nx = Nx
        self.Courant = Courant

        # Discretización espacial
        self.dx = L / (Nx - 1)
        self.x = np.linspace(0, L, Nx)
        # self.Courant = self.c*(self.T_max/self.x)

        # Discretización temporal (según condición de Courant)
        self.dt = Courant * self.dx / c
        self.Nt = int(T_max / self.dt)
        self.t = np.linspace(0, self.dt * self.Nt, self.Nt)

        # Parámetro de estabilidad
        self.r_squared = (c * self.dt / self.dx) ** 2

        # Matrices de solución
        self.y = np.zeros((Nx, self.Nt))

        # Información
        print(f"{'='*60}")
        print(f"CONFIGURACIÓN DE LA SIMULACIÓN")
        print(f"{'='*60}")
        print(f"Longitud de la cuerda: L = {L} m")
        print(f"Velocidad de onda: c = {c} m/s")
        print(f"Tiempo total: T_max = {T_max} s")
        print(f"Puntos espaciales: Nx = {Nx}")
        print(f"Paso espacial: Δx = {self.dx:.6f} m")
        print(f"Paso temporal: Δt = {self.dt:.6f} s")
        print(f"Número de Courant: r = c·Δt/Δx = {self.Courant:.4f}")
        print(f"Parámetro r² = {self.r_squared:.6f}")
        print(f"Pasos temporales: Nt = {self.Nt}")
        print(f"{'='*60}")

        # Verificar condición de estabilidad
        if self.Courant > 1.0:
            print(f" !!!!! ADVERTENCIA: Número de Courant > 1.0")
            print(f"   La solución puede ser INESTABLE")
        else:
            print(f"Condición de Courant satisfecha (r ≤ 1)")
        print(f"{'='*60}\n")

    # funciones a usar más adelante, como desplazamiento, velocidad, etc.
    def condicion_inicial_desplazamiento(self, f: Callable[[np.ndarray], np.ndarray]):
        """
        Establece la condición inicial de desplazamiento y(x, 0) = f(x).

        Parameters
        ----------
        f : callable
            Función que define el desplazamiento inicial
            f es una función que toma un array de posiciones y devuelve un array de desplazamientos.

            Este es un type hint que dice:
                f es una función (Callable)
                Esa función recibe un np.ndarray como entrada
                Esa función devuelve un np.ndarray como salida
        """
        self.y[:, 0] = f(
            self.x
        )  # manda a llamar a 'y' de nuestra cuerda [:,0]:= todos los puntos espaciales, en el primer tiempo (t=0)
        # y lo asigna a la funcion 'f' que le pasamos en el metodo y la evalua en el x de dicha cuerda generado en el constructor.
        # Condiciones de frontera: extremos fijos
        self.y[0, 0] = 0
        self.y[-1, 0] = 0

    def condicion_inicial_velocidad(self, g: Callable[[np.ndarray], np.ndarray]):
        """
        Establece la condición inicial de velocidad ∂y/∂t(x, 0) = g(x).
        El Problema se basa en que la fórmula principal de actualización es:
        y_{i,j+1} = 2y_{i,j} - y_{i,j-1} + r^2[y_{i+1,j} - 2y_{i,j} + y_{i-1,j}]
        Luego asi, para calcular el paso j+1 necesitamos:

            El paso actual: y_{i,j}
            El paso anterior: y_{i,j-1}

        ¿Pero qué pasa en el primer paso?
        * j = 0 (inicial):  Tenemos self.y[:, 0]  (de condicion_inicial_desplazamiento) funcion anterior
        * j = 1 (primero):  Necesitamos self.y[:, 1]  ?????
        ¿Cómo la calculamos? Para calcular j=1 necesitaríamos j=-1, que no existe D:
        Solución: Usamos la condición inicial de velocidad para calcular directamente el paso j=1
        Calcula y(x, Δt) usando:
            y_{i,1} = y_{i,0} + Δt·g(x_i) + (1/2)r²[y_{i+1,0} - 2y_{i,0} + y_{i-1,0}]

        Parameters
        ----------
        g : callable
            Función que define la velocidad inicial:
            ∂y/∂t(x,0)=g(x)
        """
        g_vals = g(
            self.x
        )  # evaluamos la velocidad en todos los puntos de la malla espacial

        for i in range(
            1, self.Nx - 1
        ):  # recorremos todos los puntos interiores de la cuerda
            # ¿Por que excluimos los extremos? Por las condiciones de frontera: ya los hicimos fijos
            self.y[i, 1] = (
                self.y[i, 0]
                + self.dt * g_vals[i]
                + 0.5
                * self.r_squared
                * (self.y[i + 1, 0] - 2 * self.y[i, 0] + self.y[i - 1, 0])
            )  # evaluamos lo dicho en el DocString

        # Condiciones de frontera
        self.y[0, 1] = 0
        # ¿Qué hace? Fuerza que el extremo izquierdo permanezca fijo en y=0 en el primer paso temporal.
        # ¿Por qué es necesario? Aunque no calculamos este punto en el bucle (empezamos en i=1), explícitamente lo fijamos para garantizar la condición de frontera.
        self.y[-1, 1] = 0
        # Lo mismo pero en el otro extremo XD

    # Dios soy yo de nuevo
    def resolver(self):
        """
        ¿Por que solo recibe self? pues toda la información ya esta en el objeto, solo es usarla

        ¿Qué prentende hacer?
        Resuelve la ecuación de onda usando el esquema de diferencias finitas explícito.

        Algoritmo:
            y_{i,j+1} = 2y_{i,j} - y_{i,j-1} + r²[y_{i+1,j} - 2y_{i,j} + y_{i-1,j}]
        """
        print("Resolviendo ecuación de onda...")

        for j in range(1, self.Nt - 1):
            for i in range(1, self.Nx - 1):
                self.y[i, j + 1] = (
                    2 * self.y[i, j]
                    - self.y[i, j - 1]
                    + self.r_squared
                    * (self.y[i + 1, j] - 2 * self.y[i, j] + self.y[i - 1, j])
                )  # recorre la malla temporal por cada punto espacial, por eso el for anidado

            # Condiciones de frontera, PARA ASEGURAR QUE NETA SE CUMPLA
            # Fuerza que el extremo izquierdo (x=0) permanezca fijo en y=0 en el nuevo paso temporal.
            # Aunque no calculamos este punto en el bucle interno (i empieza en 1), lo fijamos explícitamente para garantizar que los extremos nunca se muevan.
            self.y[0, j + 1] = 0
            self.y[-1, j + 1] = 0

            # Progreso
            if j % 100 == 0:
                print(f"  Paso temporal {j}/{self.Nt} ({100*j/self.Nt:.1f}%)")
                # solo imprime el progreso en formato: Paso temporal 100/990 (10.1%)

        print("Simulación completada\n")

    def analizar_estabilidad(self):
        """
        Analiza la estabilidad de la solución verificando:
        1. Valores máximos a lo largo del tiempo
        2. Si hay crecimiento exponencial (inestabilidad)
        """
        print("Análisis de estabilidad:")

        max_vals = np.max(np.abs(self.y), axis=0)
        max_inicial = max_vals[0]
        max_final = max_vals[-1]
        max_global = np.max(max_vals)

        print(f"  Amplitud inicial: {max_inicial:.6f}")
        print(f"  Amplitud final: {max_final:.6f}")
        print(f"  Amplitud máxima: {max_global:.6f}")

        if max_final > 10 * max_inicial:
            print(f"  !!!! SOLUCIÓN INESTABLE (crecimiento > 10x)")
            return False
        elif max_final > 2 * max_inicial:
            print(f"  !  Solución marginalmente estable")
            return True
        else:
            print(f"  :D Solución estable")
            return True

    # Wuakala, ya no me haga animas más
    def animar(
        self, archivo_salida: str = None, intervalo: int = 50, saltar_frames: int = 1
    ):
        """
        Ahora que si no soy yo?
        Que tenemos? a 'y' una matriz sin sabor, lo cual a los matematicos les dice mucho, pero no a nosotros
        entonces buscamos una forma de transformar esa matriz de frios numeros a una animacion para ver que suecede.
        Es decir es como tener los datos de un sismo, en comparación de ver un video del mismo.
        Los datos pueden ser precisos, pero el video te hace entender qué pasó.
        esta funcion crea una una animación de la vibración de la cuerda.

        Parameters
        ----------
        archivo_salida : str, optional
            Nombre del archivo para guardar la animación (formato .gif)
            Si archivo_salida == None: no se guarda y solo ves la animación
        intervalo : int
            Intervalo entre frames en ms
            Qué hace: Controla la velocidad de reproducción
            50 ms = 20 frames por segundo (velocidad normal)
            Valores más bajos = animación más rápida
            Valores más altos = animación más lenta
        saltar_frames : int
            Número de pasos temporales a saltar entre frames
            Qué hace: Reduce el número de frames para hacer archivos más pequeños
            saltar_frames = 1: Muestra todos los pasos temporales
            saltar_frames = 2: Muestra 1 de cada 2 pasos (mitad de frames)
            saltar_frames = 5: Muestra 1 de cada 5 pasos (⅕ de frames)
        """
        print(f"Creando animación...")  # mucho más texto amable para el usuario

        # Crea 2 paneles una arriba del otro: ax1 es el panel de arriba, ax2 es el panel de abajo
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Configurar subplot 1: Animación de la cuerda
        ax1.set_xlim(0, self.L)  # limites del eje horizontal: 0->L
        y_max = np.max(np.abs(self.y))
        ax1.set_ylim(
            -1.5 * y_max, 1.5 * y_max
        )  # limites de altura, escalando el valor más alto
        ax1.set_xlabel("Posición x (m)", fontsize=12)
        ax1.set_ylabel("Desplazamiento y (m)", fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Crea una sin datos de color azul, grosor 2 y etiqueta Cuerda
        (line,) = ax1.plot([], [], "b-", lw=2, label="Cuerda")
        """
        Nota la coma: (line,)
            ¿Por qué la coma? plot() devuelve una lista con un elemento. La coma desempaca ese elemento.
            Sin coma: line = [Line2D_object] (una lista)
            Con coma: line = Line2D_object (el objeto directo)
            Ahora line es un objeto que podemos actualizar en cada frame de la animación.
        """
        # Cuadro de texto que llenaremos posteriormente
        time_text = ax1.text(
            0.02,
            0.95,
            "",
            transform=ax1.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax1.legend(loc="upper right")

        # Configurar subplot 2: Evolución temporal en puntos específicos
        ax2.set_xlim(0, self.T_max)
        ax2.set_ylim(-1.5 * y_max, 1.5 * y_max)
        ax2.set_xlabel("Tiempo t (s)", fontsize=12)
        ax2.set_ylabel("Desplazamiento y (m)", fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Puntos de observación: arreglo con 3 valores; Nx/4 parte entera, Nx/2 parte entera, 3Nx/4 parte entera
        puntos_obs = [self.Nx // 4, self.Nx // 2, 3 * self.Nx // 4]
        colores = ["r", "g", "b"]  # rgb = colores rojo, verde y azul
        lineas_temp = []  # vamos a crear lineas temporales

        for i, (punto, color) in enumerate(zip(puntos_obs, colores)):
            # Como se come este for? i= contador,
            # zip = Empareja elementos de ambas listas: (0, (25, "r")), (1, (50, "g")), (2, (75, "b"))
            (line_temp,) = ax2.plot(
                [], [], color=color, lw=1.5, label=f"x = {self.x[punto]:.2f} m"
            )
            lineas_temp.append(line_temp)

        ax2.legend(loc="upper right")

        # Título principal
        titulo = f"Vibración de Cuerda: L={self.L}m, c={self.c}m/s, Courant={self.Courant:.3f}"
        if self.Courant > 1.0:
            titulo += " [INESTABLE]"
        fig.suptitle(titulo, fontsize=14, fontweight="bold")

        plt.tight_layout()

        # Función de inicialización: dibuja el primer frame
        def init():
            line.set_data([], [])  # Vaciamos la linea de la cuerda
            for line_temp in lineas_temp:
                # como se come este ciclo: Para cada línea temporal (roja, verde, azul) Vacía esa línea
                line_temp.set_data([], [])
            time_text.set_text("")  # Vaciamos el texto
            return [line] + lineas_temp + [time_text]  # devuelve los cambios

        # ¿Por qué devolver una lista? Para el modo blit=True (se ve más adelante :D). Matplotlib necesita saber qué objetos redibujar.

        # Función de animación: Recibe un número de frame y actualiza TODOS los elementos visuales.
        def animate(frame):
            # Esto nos permite saltar pasos temporales para hacer animaciones más cortas.
            # ¿Por qué el if?
            # Protección contra sobrepaso. Si por alguna razón j excede el número de pasos temporales, lo limita al último paso válido.
            j = frame * saltar_frames
            if j >= self.Nt:
                j = self.Nt - 1

            # Actualizar cuerda
            line.set_data(
                self.x, self.y[:, j]
            )  # Actualiza la línea con los datos del paso temporal j.
            # self.x = Array de posiciones
            # self.y[:, j] = Columna j de la matriz (todos los desplazamientos en el tiempo j)
            time_text.set_text(
                f"t = {self.t[j]:.4f} s\nFrame {frame}/{len(frames)}"
            )  # Actualizamos el recuadro de texto

            # Actualizar evolución temporal
            for i, (punto, line_temp) in enumerate(zip(puntos_obs, lineas_temp)):
                # Actualiza una de las tres líneas (roja, verde, azul).
                line_temp.set_data(self.t[: j + 1], self.y[punto, : j + 1])
                # self.t[: j + 1] = Tiempos desde 0 hasta j (historial completo)
                # self.y[punto, : j + 1] = Desplazamientos del punto específico desde 0 hasta j

            return [line] + lineas_temp + [time_text]  # devuelve los cambios

        # POR FIIIIIIIIIIIIIIIIIIN, VAMOS A ANIMAR :D
        # Frames a animar
        frames = range(
            0, self.Nt, saltar_frames
        )  # Crea una secuencia de números con saltos.
        """
        Dios nos agarre confesados, ahi les va :D

        Creamos un objeto de animation.FuncAnimation.

        Es una clase de Matplotlib que automatiza el proceso de crear animaciones: es decir, decidi ser feliz
        Parámetros:

        fig: La figura (el lienzo completo con ambos paneles)
        animate: La función que actualiza cada frame
        init_func=init: La función que inicializa (frame 0)
        frames=len(frames): Cuántos frames crear
             Si `len(frames) = 495`, crea 495 frames
        interval=intervalo: Milisegundos entre frames
            interval=50 -> 20 frames por segundo
        blit=True: Optimización muy importante
            blit = "Block Transfer"
            Solo redibuja lo que cambió, no toda la figura
            Hace la animación mucho más rápida
            Requiere que animate e init devuelvan listas de objetos
        repeat=True: La animación se repite en loop

        ¿Cómo funciona internamente?
            1. Llama init() -> dibuja frame vacío
            2. Llama animate(0) -> dibuja frame 0
            3. Espera 'interval' ms
            4. Llama animate(1) -> dibuja frame 1
            5. Espera 'interval' ms
            6. Llama animate(2) -> dibuja frame 2
        """
        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(frames),
            interval=intervalo,
            blit=True,
            repeat=True,
        )

        # Guardar animación
        if archivo_salida:
            print(f"  Guardando animación en {archivo_salida}...")
            writer = PillowWriter(
                fps=20
            )  # creamos un writer de Pillow: Un objeto que sabe cómo convertir frames a formato GIF.
            anim.save(archivo_salida, writer=writer)
            print(f"  :D Animación guardada")

        plt.show()
        print(":D Animación completada\n")

        return anim

    def graficar_evolucion_temporal(self, puntos_x: list = None):
        """
        Grafica la evolución temporal del desplazamiento en puntos específicos.

        Parameters
        ----------
        puntos_x : list, optional
            Lista de posiciones x donde observar (por defecto: cuartos de la cuerda)
        """
        if puntos_x is None:
            puntos_x = [self.L / 4, self.L / 2, 3 * self.L / 4]

        fig, axes = plt.subplots(len(puntos_x), 1, figsize=(12, 3 * len(puntos_x)))

        if len(puntos_x) == 1:
            axes = [axes]

        for ax, x_pos in zip(axes, puntos_x):
            idx = np.argmin(np.abs(self.x - x_pos))
            ax.plot(self.t, self.y[idx, :], "b-", lw=1.5)
            ax.set_xlabel("Tiempo t (s)")
            ax.set_ylabel("Desplazamiento y (m)")
            ax.set_title(f"Evolución temporal en x = {self.x[idx]:.3f} m")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def graficar_instantaneas(self, tiempos: list):
        """
        Grafica instantáneas de la cuerda en diferentes tiempos.

        Parameters
        ----------
        tiempos : list
            Lista de tiempos donde graficar
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for t_val in tiempos:
            idx = np.argmin(np.abs(self.t - t_val))
            ax.plot(self.x, self.y[:, idx], label=f"t = {self.t[idx]:.4f} s")

        ax.set_xlabel("Posición x (m)")
        ax.set_ylabel("Desplazamiento y (m)")
        ax.set_title("Instantáneas de la cuerda en diferentes tiempos")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# funcion para condiciones iniciales:


def desplazamiento_gaussiano(x, A=0.1, x0=None, sigma=0.05):
    """
    Pulso gaussiano:
        f(x) = A * exp(-(x - x0)² / (2σ²))

    Parameters
    ----------
    x : array
        Posiciones
    A : float
        Amplitud
    x0 : float
        Centro del pulso (por defecto: centro de la cuerda)
    sigma : float
        Ancho del pulso
    """
    if x0 is None:
        x0 = (x[0] + x[-1]) / 2
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def desplazamiento_triangular(x, A=0.1, x0=None):
    """
    Desplazamiento triangular.

    Parameters
    ----------
    x : array
        Posiciones
    A : float
        Amplitud
    x0 : float
        Posición del pico
    """
    if x0 is None:
        x0 = (x[0] + x[-1]) / 2
    L = x[-1] - x[0]
    y = np.zeros_like(x)
    mask1 = x <= x0
    mask2 = x > x0
    y[mask1] = A * x[mask1] / x0
    y[mask2] = A * (L - x[mask2]) / (L - x0)
    return y


def velocidad_cero(x):
    """Velocidad inicial cero."""
    return np.zeros_like(x)


# para esto creamos una clase:
#  experimentos


def experimento_1_estable():
    """
    Experimento 1: Condición de Courant SATISFECHA (estable)
    """
    print("\n" + "=" * 60)
    print("EXPERIMENTO 1: CONDICIÓN DE COURANT SATISFECHA")
    print("=" * 60 + "\n")

    # Parámetros
    L = 1.0  # Longitud de la cuerda (m)
    c = 10.0  # Velocidad de onda (m/s)
    T_max = 0.5  # Tiempo total (s)
    Nx = 100  # Puntos espaciales
    Courant = 0.5  # r = 0.5 < 1 -> ESTABLE
    # Courant = c*(T_max/L)  # r = 0.5 < 1 -> ESTABLE

    # Crear simulación
    cuerda = CuerdaVibrante(L, c, T_max, Nx, Courant)

    # Condiciones iniciales: pulso gaussiano
    cuerda.condicion_inicial_desplazamiento(
        lambda x: desplazamiento_gaussiano(x, A=0.1, sigma=0.05)
    )
    cuerda.condicion_inicial_velocidad(velocidad_cero)

    # Resolver
    cuerda.resolver()

    # Análisis
    cuerda.analizar_estabilidad()

    # Visualización
    cuerda.animar(archivo_salida="animacion_estable.gif", saltar_frames=2)
    cuerda.graficar_instantaneas([0, T_max / 4, T_max / 2, 3 * T_max / 4, T_max])

    return cuerda


def experimento_2_inestable():
    """
    Experimento 2: Condición de Courant VIOLADA (inestable)
    """
    print("\n" + "=" * 60)
    print("EXPERIMENTO 2: CONDICIÓN DE COURANT VIOLADA")
    print("=" * 60 + "\n")

    # Parámetros
    L = 1.0  # Longitud de la cuerda (m)
    c = 10.0  # Velocidad de onda (m/s)
    T_max = 0.5  # Tiempo total (s)
    Nx = 100  # Puntos espaciales
    Courant = 1.5  # r = 1.5 > 1 -> INESTABLE

    # Crear simulación
    cuerda = CuerdaVibrante(L, c, T_max, Nx, Courant)

    # Condiciones iniciales: pulso gaussiano
    cuerda.condicion_inicial_desplazamiento(
        lambda x: desplazamiento_gaussiano(x, A=0.1, sigma=0.05)
    )
    cuerda.condicion_inicial_velocidad(velocidad_cero)

    # Resolver
    cuerda.resolver()

    # Análisis
    cuerda.analizar_estabilidad()

    # Visualización
    cuerda.animar(archivo_salida="animacion_inestable.gif", saltar_frames=2)
    cuerda.graficar_instantaneas([0, T_max / 4, T_max / 2])

    return cuerda


def experimento_3_critico():
    """
    Experimento 3: Condición de Courant EN EL LÍMITE (marginalmente estable)
    """
    print("\n" + "=" * 60)
    print("EXPERIMENTO 3: CONDICIÓN DE COURANT EN EL LÍMITE")
    print("=" * 60 + "\n")

    # Parámetros
    L = 1.0  # Longitud de la cuerda (m)
    c = 10.0  # Velocidad de onda (m/s)
    T_max = 0.5  # Tiempo total (s)
    Nx = 100  # Puntos espaciales
    Courant = 1.0  # r = 1.0 (crítico)

    # Crear simulación
    cuerda = CuerdaVibrante(L, c, T_max, Nx, Courant)

    # Condiciones iniciales: pulso gaussiano
    cuerda.condicion_inicial_desplazamiento(
        lambda x: desplazamiento_gaussiano(x, A=0.1, sigma=0.05)
    )
    cuerda.condicion_inicial_velocidad(velocidad_cero)

    # Resolver
    cuerda.resolver()

    # Análisis
    cuerda.analizar_estabilidad()

    # Visualización
    cuerda.animar(archivo_salida="animacion_critica.gif", saltar_frames=2)
    cuerda.graficar_instantaneas([0, T_max / 4, T_max / 2, 3 * T_max / 4, T_max])

    return cuerda


def comparacion_courant():
    """
    Comparación de diferentes valores de Courant.
    """
    print("\n" + "=" * 60)
    print("COMPARACIÓN: DIFERENTES VALORES DE COURANT")
    print("=" * 60 + "\n")

    L = 1.0
    c = 10.0
    T_max = 0.2
    Nx = 100

    courant_vals = [0.3, 0.7, 1.0, 1.2, 1.5]

    fig, axes = plt.subplots(len(courant_vals), 1, figsize=(12, 3 * len(courant_vals)))

    for ax, r in zip(axes, courant_vals):
        cuerda = CuerdaVibrante(L, c, T_max, Nx, r)
        cuerda.condicion_inicial_desplazamiento(
            lambda x: desplazamiento_gaussiano(x, A=0.1, sigma=0.05)
        )
        cuerda.condicion_inicial_velocidad(velocidad_cero)
        cuerda.resolver()

        # Graficar solución final
        ax.plot(cuerda.x, cuerda.y[:, -1], "b-", lw=2)
        ax.set_ylabel("y (m)")
        ax.set_title(f'Courant = {r:.2f} ({"ESTABLE" if r <= 1 else "INESTABLE"})')
        ax.grid(True, alpha=0.3)

        max_val = np.max(np.abs(cuerda.y[:, -1]))
        (
            ax.set_ylim(-1.5 * max_val, 1.5 * max_val)
            if max_val > 0
            else ax.set_ylim(-0.1, 0.1)
        )

    axes[-1].set_xlabel("Posición x (m)")
    plt.tight_layout()
    plt.savefig("comparacion_courant.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(":D Gráfica de comparación guardada en 'comparacion_courant.png'\n")


# llamamos al MAIN


if __name__ == "__main__":
    # Crear carpeta para resultados
    os.makedirs("resultados_cuerda", exist_ok=True)
    os.chdir("resultados_cuerda")

    print("\n" + "-" * 30)
    print("SIMULACIÓN DE VIBRACIÓN DE CUERDA")
    print("Ecuación de Onda - Análisis de Estabilidad")
    print("-" * 30 + "\n")

    # Ejecutar experimentos
    print("\nEjecutando experimentos...\n")

    # Experimento 1: Estable
    cuerda1 = experimento_1_estable()

    # Experimento 2: Inestable
    cuerda2 = experimento_2_inestable()

    # Experimento 3: Crítico
    cuerda3 = experimento_3_critico()

    # Comparación
    # comparacion_courant()

    #     print("\n" + "=" * 60)
    #     print("RESUMEN DE RESULTADOS")
    #     print("=" * 60)
    #     print(
    #         """
    # CONCLUSIONES:

    # 1. Condición de Courant r ≤ 1:
    #    - r = c·Δt/Δx ≤ 1
    #    - Equivalente a: Δt ≤ Δx/c

    # 2. Interpretación física:
    #    - La onda no puede viajar más de Δx en un tiempo Δt
    #    - Respeta la causalidad del sistema

    # 3. Resultados experimentales:
    #    - r < 1: Solución ESTABLE y físicamente correcta
    #    - r = 1: Marginalmente estable (caso límite)
    #    - r > 1: Solución INESTABLE (oscilaciones espurias)

    # 4. Recomendación práctica:
    #    - Usar r ≈ 0.5 - 0.8 para máxima estabilidad
    #    - Nunca usar r > 1.0
    #     """
    #     )
    #     print("=" * 60 + "\n")

    print(":D Todos los experimentos completados")
    print(f"dir: Resultados guardados en: {os.getcwd()}")
