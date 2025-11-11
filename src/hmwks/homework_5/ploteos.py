"""
COMBINAR GRÁFICAS GENERADAS POR CÓDIGO EN SUBPLOT
=================================================

Cómo mostrar múltiples gráficas en una sola figura sin guardar imágenes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from numpy.polynomial import Polynomial

# Datos de ejemplo (tus datos de sección eficaz)
Ei = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])
fE = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])
sigma = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])

E_range = np.linspace(0, 200, 1000)


# ==============================================================================
# MÉTODO 1: CREAR SUBPLOTS DESDE EL INICIO (MEJOR PRÁCTICA)
# ==============================================================================


def metodo_1_subplots_desde_inicio():
    """
    La mejor forma: crear la figura con subplots desde el inicio.
    """
    print("\n" + "=" * 60)
    print("MÉTODO 1: Crear subplots desde el inicio")
    print("=" * 60)

    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ========================================
    # SUBPLOT 1: Gráfica de Lagrange
    # ========================================

    # Calcular Lagrange
    def lagrange_interpolation(x_data, y_data):
        n = len(x_data)
        poly_total = Polynomial([0.0])
        for i in range(n):
            indices = [j for j in range(n) if j != i]
            numerador = Polynomial.fromroots(x_data[indices])
            denominador = np.prod(x_data[i] - x_data[indices])
            L_i = numerador / denominador
            poly_total = poly_total + y_data[i] * L_i
        return poly_total

    polinomio = lagrange_interpolation(Ei, fE)
    f_lagrange = polinomio(E_range)

    # Graficar en ax1
    ax1.errorbar(
        Ei, fE, yerr=sigma, fmt="o", color="red", markersize=8, capsize=5, label="Datos"
    )
    ax1.plot(E_range, f_lagrange, "-", color="blue", linewidth=2.5, label="Lagrange")
    ax1.set_xlabel("Energía E (MeV)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Sección eficaz σ(E) (mb)", fontsize=12, fontweight="bold")
    ax1.set_title("Interpolación de Lagrange", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ========================================
    # SUBPLOT 2: Gráfica de Splines
    # ========================================

    # Calcular Splines
    spline = CubicSpline(Ei, fE, bc_type="natural")
    f_spline = spline(E_range)

    # Graficar en ax2
    ax2.errorbar(
        Ei, fE, yerr=sigma, fmt="o", color="red", markersize=8, capsize=5, label="Datos"
    )
    ax2.plot(
        E_range, f_spline, "-", color="green", linewidth=2.5, label="Splines Cúbicos"
    )
    ax2.set_xlabel("Energía E (MeV)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Sección eficaz σ(E) (mb)", fontsize=12, fontweight="bold")
    ax2.set_title("Splines Cúbicos", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Título general
    fig.suptitle("Comparación de Métodos", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ==============================================================================
# MÉTODO 2: USAR FUNCIONES QUE GRAFICAN EN UN AX DADO
# ==============================================================================


def graficar_lagrange(ax, Ei, fE, sigma, E_range):
    """
    Función que grafica Lagrange en el ax dado.

    Parámetros:
    -----------
    ax : matplotlib.axes.Axes
        El subplot donde graficar
    """

    # Calcular interpolación
    def lagrange_interpolation(x_data, y_data):
        n = len(x_data)
        poly_total = Polynomial([0.0])
        for i in range(n):
            indices = [j for j in range(n) if j != i]
            numerador = Polynomial.fromroots(x_data[indices])
            denominador = np.prod(x_data[i] - x_data[indices])
            L_i = numerador / denominador
            poly_total = poly_total + y_data[i] * L_i
        return poly_total

    polinomio = lagrange_interpolation(Ei, fE)
    f_lagrange = polinomio(E_range)

    # Graficar
    ax.errorbar(
        Ei, fE, yerr=sigma, fmt="o", color="red", markersize=8, capsize=5, label="Datos"
    )
    ax.plot(E_range, f_lagrange, "-", color="blue", linewidth=2.5, label="Lagrange")
    ax.set_xlabel("Energía E (MeV)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sección eficaz σ(E) (mb)", fontsize=12, fontweight="bold")
    ax.set_title("Interpolación de Lagrange", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)


def graficar_splines(ax, Ei, fE, sigma, E_range):
    """
    Función que grafica Splines en el ax dado.
    """
    # Calcular splines
    spline = CubicSpline(Ei, fE, bc_type="natural")
    f_spline = spline(E_range)

    # Graficar
    ax.errorbar(
        Ei, fE, yerr=sigma, fmt="o", color="red", markersize=8, capsize=5, label="Datos"
    )
    ax.plot(
        E_range, f_spline, "-", color="green", linewidth=2.5, label="Splines Cúbicos"
    )
    ax.set_xlabel("Energía E (MeV)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sección eficaz σ(E) (mb)", fontsize=12, fontweight="bold")
    ax.set_title("Splines Cúbicos", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)


def metodo_2_funciones_con_ax():
    """
    Usar funciones que reciben el ax como parámetro.
    """
    print("\n" + "=" * 60)
    print("MÉTODO 2: Funciones que reciben ax")
    print("=" * 60)

    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Llamar funciones pasando el ax correspondiente
    graficar_lagrange(ax1, Ei, fE, sigma, E_range)
    graficar_splines(ax2, Ei, fE, sigma, E_range)

    fig.suptitle("Comparación de Métodos", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ==============================================================================
# MÉTODO 3: GUARDAR OBJETOS FIGURE Y COMBINARLOS
# ==============================================================================


def crear_grafica_lagrange():
    """Crea y retorna la figura de Lagrange."""
    fig, ax = plt.subplots(figsize=(8, 6))

    def lagrange_interpolation(x_data, y_data):
        n = len(x_data)
        poly_total = Polynomial([0.0])
        for i in range(n):
            indices = [j for j in range(n) if j != i]
            numerador = Polynomial.fromroots(x_data[indices])
            denominador = np.prod(x_data[i] - x_data[indices])
            L_i = numerador / denominador
            poly_total = poly_total + y_data[i] * L_i
        return poly_total

    polinomio = lagrange_interpolation(Ei, fE)
    f_lagrange = polinomio(E_range)

    ax.errorbar(Ei, fE, yerr=sigma, fmt="o", color="red", markersize=8, capsize=5)
    ax.plot(E_range, f_lagrange, "-", color="blue", linewidth=2.5)
    ax.set_xlabel("Energía E (MeV)")
    ax.set_ylabel("Sección eficaz σ(E) (mb)")
    ax.set_title("Interpolación de Lagrange")
    ax.grid(True, alpha=0.3)

    return fig, ax


def crear_grafica_splines():
    """Crea y retorna la figura de Splines."""
    fig, ax = plt.subplots(figsize=(8, 6))

    spline = CubicSpline(Ei, fE, bc_type="natural")
    f_spline = spline(E_range)

    ax.errorbar(Ei, fE, yerr=sigma, fmt="o", color="red", markersize=8, capsize=5)
    ax.plot(E_range, f_spline, "-", color="green", linewidth=2.5)
    ax.set_xlabel("Energía E (MeV)")
    ax.set_ylabel("Sección eficaz σ(E) (mb)")
    ax.set_title("Splines Cúbicos")
    ax.grid(True, alpha=0.3)

    return fig, ax


def metodo_3_combinar_figuras_existentes(fig1, ax1, fig2, ax2):
    """
    Combina dos figuras existentes en una nueva figura con subplots.

    NOTA: Este método copia las líneas y propiedades de los axes originales.
    """
    print("\n" + "=" * 60)
    print("MÉTODO 3: Combinar figuras existentes")
    print("=" * 60)

    # Crear nueva figura con subplots
    fig_nueva, (ax_nuevo1, ax_nuevo2) = plt.subplots(1, 2, figsize=(16, 6))

    # Copiar contenido de ax1 a ax_nuevo1
    for line in ax1.get_lines():
        ax_nuevo1.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linewidth=line.get_linewidth(),
            linestyle=line.get_linestyle(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            label=line.get_label(),
        )

    # Copiar contenedores de ErrorBar de ax1
    for container in ax1.containers:
        if hasattr(container, "get_segments"):  # Es ErrorbarContainer
            # Extraer datos del errorbar
            lines = container.get_children()
            if lines:
                x = lines[0].get_xdata()
                y = lines[0].get_ydata()
                # Redibujar errorbar (simplificado)
                ax_nuevo1.errorbar(
                    Ei, fE, yerr=sigma, fmt="o", color="red", markersize=8, capsize=5
                )

    ax_nuevo1.set_xlabel(ax1.get_xlabel())
    ax_nuevo1.set_ylabel(ax1.get_ylabel())
    ax_nuevo1.set_title(ax1.get_title())
    ax_nuevo1.grid(True, alpha=0.3)
    ax_nuevo1.legend()

    # Copiar contenido de ax2 a ax_nuevo2
    for line in ax2.get_lines():
        ax_nuevo2.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linewidth=line.get_linewidth(),
            linestyle=line.get_linestyle(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            label=line.get_label(),
        )

    for container in ax2.containers:
        if hasattr(container, "get_segments"):
            ax_nuevo2.errorbar(
                Ei, fE, yerr=sigma, fmt="o", color="red", markersize=8, capsize=5
            )

    ax_nuevo2.set_xlabel(ax2.get_xlabel())
    ax_nuevo2.set_ylabel(ax2.get_ylabel())
    ax_nuevo2.set_title(ax2.get_title())
    ax_nuevo2.grid(True, alpha=0.3)
    ax_nuevo2.legend()

    fig_nueva.suptitle("Gráficas Combinadas", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ==============================================================================
# MÉTODO 4: GUARDAR DATOS Y REDIBUJAR (MÁS LIMPIO)
# ==============================================================================


class DatosGrafica:
    """Clase para almacenar datos de una gráfica."""

    def __init__(self):
        self.lineas = []
        self.errorbar = None
        self.xlabel = ""
        self.ylabel = ""
        self.title = ""
        self.grid = True


def crear_y_guardar_lagrange():
    """Crea gráfica de Lagrange y guarda los datos."""
    datos = DatosGrafica()

    # Calcular
    def lagrange_interpolation(x_data, y_data):
        n = len(x_data)
        poly_total = Polynomial([0.0])
        for i in range(n):
            indices = [j for j in range(n) if j != i]
            numerador = Polynomial.fromroots(x_data[indices])
            denominador = np.prod(x_data[i] - x_data[indices])
            L_i = numerador / denominador
            poly_total = poly_total + y_data[i] * L_i
        return poly_total

    polinomio = lagrange_interpolation(Ei, fE)
    f_lagrange = polinomio(E_range)

    # Guardar datos
    datos.lineas.append(
        {
            "x": E_range,
            "y": f_lagrange,
            "color": "blue",
            "linewidth": 2.5,
            "label": "Lagrange",
        }
    )

    datos.errorbar = {"x": Ei, "y": fE, "yerr": sigma, "color": "red", "label": "Datos"}

    datos.xlabel = "Energía E (MeV)"
    datos.ylabel = "Sección eficaz σ(E) (mb)"
    datos.title = "Interpolación de Lagrange"

    return datos


def crear_y_guardar_splines():
    """Crea gráfica de Splines y guarda los datos."""
    datos = DatosGrafica()

    # Calcular
    spline = CubicSpline(Ei, fE, bc_type="natural")
    f_spline = spline(E_range)

    # Guardar datos
    datos.lineas.append(
        {
            "x": E_range,
            "y": f_spline,
            "color": "green",
            "linewidth": 2.5,
            "label": "Splines",
        }
    )

    datos.errorbar = {"x": Ei, "y": fE, "yerr": sigma, "color": "red", "label": "Datos"}

    datos.xlabel = "Energía E (MeV)"
    datos.ylabel = "Sección eficaz σ(E) (mb)"
    datos.title = "Splines Cúbicos"

    return datos


def redibujar_desde_datos(ax, datos):
    """Redibuja una gráfica a partir de datos guardados."""
    # Errorbar primero
    if datos.errorbar:
        eb = datos.errorbar
        ax.errorbar(
            eb["x"],
            eb["y"],
            yerr=eb.get("yerr"),
            fmt="o",
            color=eb["color"],
            markersize=8,
            capsize=5,
            label=eb["label"],
        )

    # Líneas
    for linea in datos.lineas:
        ax.plot(
            linea["x"],
            linea["y"],
            color=linea["color"],
            linewidth=linea["linewidth"],
            label=linea["label"],
        )

    # Etiquetas
    ax.set_xlabel(datos.xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(datos.ylabel, fontsize=12, fontweight="bold")
    ax.set_title(datos.title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)


def metodo_4_guardar_datos_y_redibujar():
    """
    Guarda los datos de las gráficas y las redibuja juntas.
    """
    print("\n" + "=" * 60)
    print("MÉTODO 4: Guardar datos y redibujar")
    print("=" * 60)

    # Crear y guardar datos de ambas gráficas
    datos_lagrange = crear_y_guardar_lagrange()
    datos_splines = crear_y_guardar_splines()

    # Crear nueva figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Redibujar desde los datos guardados
    redibujar_desde_datos(ax1, datos_lagrange)
    redibujar_desde_datos(ax2, datos_splines)

    fig.suptitle("Gráficas Combinadas", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ==============================================================================
# EJEMPLO PRÁCTICO: CÓMO ADAPTAR TU CÓDIGO ACTUAL
# ==============================================================================


def ejemplo_adaptacion_codigo_actual():
    """
    Ejemplo de cómo adaptar código que ya tienes.

    SI TIENES:
    ---------
    # Código 1: Genera gráfica de Lagrange
    plt.figure()
    plt.plot(...)
    plt.show()

    # Código 2: Genera gráfica de Splines
    plt.figure()
    plt.plot(...)
    plt.show()

    CÁMBIALO A:
    -----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Código 1: En lugar de plt.plot(...), usa ax1.plot(...)
    ax1.plot(...)

    # Código 2: En lugar de plt.plot(...), usa ax2.plot(...)
    ax2.plot(...)

    plt.show()  # Un solo show al final
    """
    print("\n" + "=" * 60)
    print("EJEMPLO: Adaptación de código existente")
    print("=" * 60)

    # ANTES: Dos gráficas separadas
    # ===============================
    print("\nANTES (2 gráficas separadas):")
    print("-" * 40)
    print(
        """
    # Gráfica 1
    plt.figure()
    plt.plot(x, y1, 'r-')
    plt.title('Gráfica 1')
    plt.show()
    
    # Gráfica 2
    plt.figure()
    plt.plot(x, y2, 'b-')
    plt.title('Gráfica 2')
    plt.show()
    """
    )

    # DESPUÉS: Una figura con subplots
    # =================================
    print("\nDESPUÉS (subplot combinado):")
    print("-" * 40)
    print(
        """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfica 1 → Cambiar plt. por ax1.
    ax1.plot(x, y1, 'r-')
    ax1.set_title('Gráfica 1')
    
    # Gráfica 2 → Cambiar plt. por ax2.
    ax2.plot(x, y2, 'b-')
    ax2.set_title('Gráfica 2')
    
    plt.tight_layout()
    plt.show()  # Un solo show
    """
    )


# ==============================================================================
# MAIN: EJECUTAR EJEMPLOS
# ==============================================================================

if __name__ == "__main__":

    print("\n" + "🎨" * 30)
    print("COMBINAR GRÁFICAS EN SUBPLOT - SIN GUARDAR IMÁGENES")
    print("🎨" * 30)

    # Ejecutar ejemplos
    metodo_1_subplots_desde_inicio()

    input("\nPresiona Enter para continuar con el Método 2...")
    metodo_2_funciones_con_ax()

    input("\nPresiona Enter para continuar con el Método 3...")
    # Crear figuras separadas primero
    fig_lag, ax_lag = crear_grafica_lagrange()
    plt.close(fig_lag)  # Cerrar sin mostrar
    fig_spl, ax_spl = crear_grafica_splines()
    plt.close(fig_spl)  # Cerrar sin mostrar
    # Combinarlas
    metodo_3_combinar_figuras_existentes(fig_lag, ax_lag, fig_spl, ax_spl)

    input("\nPresiona Enter para continuar con el Método 4...")
    metodo_4_guardar_datos_y_redibujar()

    # Mostrar ejemplo de adaptación
    ejemplo_adaptacion_codigo_actual()

    print("\n" + "=" * 60)
    print("✓ Todos los ejemplos completados")
    print("=" * 60)

    print("\n📝 RESUMEN:")
    print("  • MÉTODO 1 (recomendado): Crear subplots desde el inicio")
    print("  • MÉTODO 2: Usar funciones que reciben ax")
    print("  • MÉTODO 3: Copiar de figuras existentes")
    print("  • MÉTODO 4: Guardar datos y redibujar")
