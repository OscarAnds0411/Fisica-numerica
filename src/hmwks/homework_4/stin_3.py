"""
VIBRACIÓN DE UNA CUERDA - ECUACIÓN DE ONDA
==========================================

Solución numérica de la ecuación de onda 1D usando diferencias finitas.
Analiza la estabilidad según la condición de Courant-Friedrichs-Lewy (CFL).

Autor: Oscar Andrés
Proyecto: Física Numérica
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from typing import Callable, Tuple
import os

# Si bien no hemos visto nada de esto, quiero practicar para ya ponerme a trabajar
# por lo que se me hizo mucho mas facil construir una clase para ya no repetir codigo
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

    # constructor:
    def __init__(self, L: float, c: float, T_max: float, Nx: int, Courant: float):
        self.L = L
        self.c = c
        self.T_max = T_max
        self.Nx = Nx
        self.Courant = Courant

        # Discretización espacial
        self.dx = L / (Nx - 1)
        self.x = np.linspace(0, L, Nx)

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
        """
        self.y[:, 0] = f(self.x)
        # Condiciones de frontera
        self.y[0, 0] = 0
        self.y[-1, 0] = 0

    def condicion_inicial_velocidad(self, g: Callable[[np.ndarray], np.ndarray]):
        """
        Establece la condición inicial de velocidad ∂y/∂t(x, 0) = g(x).

        Calcula y(x, Δt) usando:
            y_{i,1} = y_{i,0} + Δt·g(x_i) + (1/2)r²[y_{i+1,0} - 2y_{i,0} + y_{i-1,0}]

        Parameters
        ----------
        g : callable
            Función que define la velocidad inicial
        """
        g_vals = g(self.x)

        for i in range(1, self.Nx - 1):
            self.y[i, 1] = (
                self.y[i, 0]
                + self.dt * g_vals[i]
                + 0.5
                * self.r_squared
                * (self.y[i + 1, 0] - 2 * self.y[i, 0] + self.y[i - 1, 0])
            )

        # Condiciones de frontera
        self.y[0, 1] = 0
        self.y[-1, 1] = 0

    def resolver(self):
        """
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
                )

            # Condiciones de frontera
            self.y[0, j + 1] = 0
            self.y[-1, j + 1] = 0

            # Progreso
            if j % 100 == 0:
                print(f"  Paso temporal {j}/{self.Nt} ({100*j/self.Nt:.1f}%)")

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
        Crea una animación de la vibración de la cuerda.

        Parameters
        ----------
        archivo_salida : str, optional
            Nombre del archivo para guardar la animación (formato .gif)
        intervalo : int
            Intervalo entre frames en ms
        saltar_frames : int
            Número de pasos temporales a saltar entre frames
        """
        print(f"Creando animación...")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Configurar subplot 1: Animación de la cuerda
        ax1.set_xlim(0, self.L)
        y_max = np.max(np.abs(self.y))
        ax1.set_ylim(-1.5 * y_max, 1.5 * y_max)
        ax1.set_xlabel("Posición x (m)", fontsize=12)
        ax1.set_ylabel("Desplazamiento y (m)", fontsize=12)
        ax1.grid(True, alpha=0.3)

        (line,) = ax1.plot([], [], "b-", lw=2, label="Cuerda")
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

        # Puntos de observación
        puntos_obs = [self.Nx // 4, self.Nx // 2, 3 * self.Nx // 4]
        colores = ["r", "g", "b"]
        lineas_temp = []

        for i, (punto, color) in enumerate(zip(puntos_obs, colores)):
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

        # Función de inicialización
        def init():
            line.set_data([], [])
            for line_temp in lineas_temp:
                line_temp.set_data([], [])
            time_text.set_text("")
            return [line] + lineas_temp + [time_text]

        # Función de animación
        def animate(frame):
            j = frame * saltar_frames
            if j >= self.Nt:
                j = self.Nt - 1

            # Actualizar cuerda
            line.set_data(self.x, self.y[:, j])
            time_text.set_text(f"t = {self.t[j]:.4f} s\nFrame {frame}/{len(frames)}")

            # Actualizar evolución temporal
            for i, (punto, line_temp) in enumerate(zip(puntos_obs, lineas_temp)):
                line_temp.set_data(self.t[: j + 1], self.y[punto, : j + 1])

            return [line] + lineas_temp + [time_text]

        # Frames a animar
        frames = range(0, self.Nt, saltar_frames)

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
            writer = PillowWriter(fps=20)
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
    Courant = 0.5  # r = 0.5 < 1 → ESTABLE

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
    Courant = 1.5  # r = 1.5 > 1 → INESTABLE

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
    comparacion_courant()

    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    print(
        """
CONCLUSIONES:

1. Condición de Courant r ≤ 1:
   - r = c·Δt/Δx ≤ 1
   - Equivalente a: Δt ≤ Δx/c
   
2. Interpretación física:
   - La onda no puede viajar más de Δx en un tiempo Δt
   - Respeta la causalidad del sistema
   
3. Resultados experimentales:
   - r < 1: Solución ESTABLE y físicamente correcta
   - r = 1: Marginalmente estable (caso límite)
   - r > 1: Solución INESTABLE (oscilaciones espurias)
   
4. Recomendación práctica:
   - Usar r ≈ 0.5 - 0.8 para máxima estabilidad
   - Nunca usar r > 1.0
    """
    )
    print("=" * 60 + "\n")

    print(":D Todos los experimentos completados")
    print(f"dir: Resultados guardados en: {os.getcwd()}")
