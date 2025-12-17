"""
Generador de números aleatorios.

(a) Escriba un programa que genere números pseudo-aleatorios uti-
lizando el mÈtodo de congruencias lineales.

(b) Con un objetivo pedagÛgico, pruebe su programa con (a; c; M; x0) =
(57; 1; 256; 10). Determine el periodo, es decir, cu·ntos n ̇meros
deben generarse para que la sucesiÛn se repita.

(c) Tome la sucesiÛn del inciso anterior, graÖcando los pares (x2i
1; x2i),

i = 1; 2; :::
(d) Ahora graÖque xi vs i.
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np

# importacion de librerias
import pylab as pyl
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.stats import chi2

# creamos una consola
cons = Console()

output_dir = "resultados_tarea_7"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    cons.print(
        f"[bold red] directorio no existente\n directorio generado con exito :D {output_dir}\n"
    )
else:
    cons.print(f"[bold green] directorio ya existente {output_dir}\n")


# creamos la rutina a explotar:
def linear_congruential(a, c, M, x0, max_iterations=None):
    """
    buscamos hacer una función que haga:
    x_{n+1}=(ax_n+c) mod m
        Parámetros:
    -----------
    a : int
        Multiplicador (0 < a < M)
    c : int
        Incremento (0 ≤ c < M)
    M : int
        Módulo (M > 0)
    x0 : int
        Semilla inicial (0 ≤ x0 < M)
    max_iterations : int, optional
        Número máximo de iteraciones (previene ciclos infinitos)
    Retorna:
    --------
    num_ale : list
        Lista de números generados (SIN incluir x0 repetido)
    periodo : int
        Longitud del período detectado
    """
    # realizamos una validación rapida:
    if M <= 0:
        raise ValueError(f"M debe ser positivo, recibido: {M}")
    if not (0 <= x0 < M):
        raise ValueError(f"x0 debe estar en [0, M), recibido: x0={x0}, M={M}")
    if not (0 < a < M):
        raise ValueError(f"a debe estar en (0, M), recibido: a={a}, M={M}")
    if not (0 <= c < M):
        raise ValueError(f"c debe estar en [0, M), recibido: c={c}, M={M}")
    # evadimos romper la memoria
    if max_iterations == None:
        max_iterations = M + 1  # Esto no debe de superar las M iteraciones

    num_ale = []
    x = x0
    for i in range(max_iterations):
        x_s = (a * x + c) % M
        if x_s == x0:
            break
        num_ale.append(x_s)
        x = x_s
    else:
        # Si llegamos aquí, no encontramos ciclo (caso extremo)
        import warnings

        warnings.warn(
            f"No se detectó período completo en {max_iterations} iteraciones. "
            f"Parámetros: a={a}, c={c}, M={M}, x0={x0}"
        )
    long_g = len(num_ale)
    return num_ale, long_g


def graficos(lista_aleatorios, titulo, output_dir, flag=True):
    """
    funciones para graficar
    """
    if len(lista_aleatorios) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 números, recibidos: {len(lista_aleatorios)}"
        )

    if flag:
        # Ajustar longitud a número par para emparejamiento correcto
        lon_g_par = len(lista_aleatorios) - len(lista_aleatorios) % 2

        # Separar en índices pares e impares (slicing)
        xpar = lista_aleatorios[:lon_g_par:2]  # Índices 0, 2, 4, ... -> x_{2i}
        ximpar = lista_aleatorios[1:lon_g_par:2]  # Índices 1, 3, 5, ... -> x_{2i+1}

        # Graficar Test de Correlación Serial
        # plt.figure(figsize=(14, 12))

        # plt.subplot(1, 2, 1)
        plt.plot(ximpar, xpar, "o", markersize=3, alpha=0.6, color="dodgerblue")
        plt.xlabel(r"$x_{2i+1}$ (índices impares)", fontsize=11)
        plt.ylabel(r"$x_{2i}$ (índices pares)", fontsize=11)
        plt.title(f"{titulo}", fontsize=12, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = os.path.join(output_dir, f"grafico_par_impar.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        cons.print(f"[bold green] Imagen guardada en:[/bold green] {filename}\n")
        plt.show()

        # Análisis visual
        correlacion_info = (
            f"Puntos: {len(xpar)}\n"
            f"Un buen LCG debe mostrar distribución uniforme\n"
            f"sin patrones visibles (líneas, clusters, etc.)"
        )
        cons.print(f"{correlacion_info}", style="bold on magenta")
    else:
        I = np.arange(len(lista_aleatorios))

        # plt.subplot(1, 2, 2)
        plt.plot(I, lista_aleatorios, "o", markersize=2, alpha=0.5, color="forestgreen")
        plt.xlabel("Índice $i$", fontsize=11)
        plt.ylabel(r"$x_i$", fontsize=11)
        plt.title(f"{titulo}: Distribución Temporal", fontsize=12, fontweight="bold")
        plt.grid(True, alpha=0.3)

        # Líneas de referencia (min/max)
        plt.axhline(
            min(lista_aleatorios), color="red", linestyle="--", alpha=0.3, linewidth=1
        )
        plt.axhline(
            max(lista_aleatorios), color="red", linestyle="--", alpha=0.3, linewidth=1
        )

        plt.tight_layout()
        filename = os.path.join(output_dir, f"{titulo}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        cons.print(f"[bold green] Imagen guardada en:[/bold green] {filename}\n")
        plt.show()
    return filename


cons.rule("[bold red] Primera parte, generación de numeros ''aleatorios''")

cons.print(
    "[bold cyan] Creamos una funcion llamada [/bold cyan][bold yellow] linear_congruential\n"
)
a, c, M, x0 = 57, 1, 256, 10
# a, c, M, x0 =7,0,10,7
cons.print(
    f"[bold cyan] Usamos la variables a: {a}, c: {c}, M: {M} y una semilla x0: {x0}\n"
)

lista_ale, periodo = linear_congruential(a, c, M, x0)
tab = Table(title="[yellow]Primeros 10 números pseudo-aleatorios", box=box.ROUNDED)
columns = ["i", "x_aleatorio"]
for r in columns:
    tab.add_column(r, justify="center", style="magenta")

if len(lista_ale) < 10:
    for i in range(len(lista_ale)):
        tab.add_row(f"{str(i+1)}", f"{lista_ale[i]}")
else:
    for i in range(10):
        tab.add_row(f"{str(i+1)}", f"{lista_ale[i]}")
cons.print(tab)
cons.print(f"El periodo del algoritmo es: {periodo}")

cons.rule("[bold red]Graficos del generador de pseudo numeros aleatorios")
cons.print(
    "[bold cyan] De nuestra sucesión generada, graficaremos los pares (x_{2i-1}, x_{2i})"
)
graficos(lista_ale, r"Grafico de los puntos $(x_{2i-1}, x_{2i})$", output_dir)
cons.print("[bold cyan] De nuestra sucesión generada, graficaremos los pares (x_i, i)")
graficos(lista_ale, r"Grafico de los puntos $(x_i, i)$", output_dir, False)


def chi_cuadrada(numeros, M):
    """
    Prueba chi-cuadrada simple para ver si los números son uniformes.
    Parámetros:
    -----------
    numeros : list
        Lista de números a evaluar
    M : int
        Módulo del generador (rango de valores)
    Retorna:
    --------
    dict con resultados de la prueba
    """
    N = len(numeros)
    k = int(np.sqrt(N))  # Número de intervalos

    # Contar cuántos números caen en cada intervalo
    freq_obs, _ = np.histogram(numeros, bins=k, range=(0, M))

    # Lo que esperaríamos si fueran uniformes
    freq_esp = N / k

    # Calcular chi-cuadrada
    chi2_calc = np.sum((freq_obs - freq_esp) ** 2 / freq_esp)

    # Valor crítico (95% de confianza)
    gl = k - 1
    chi2_crit = chi2.ppf(0.95, gl)

    # ¿Pasa la prueba?
    pasa = chi2_calc < chi2_crit

    return {
        "chi2": chi2_calc,
        "critico": chi2_crit,
        "pasa": pasa,
        "freq_obs": freq_obs,
        "freq_esp": freq_esp,
        "k": k,
    }


def graficar_chi2_simple(resultado, numeros, titulo, output_dir):
    """
    Gráfico simple de la prueba
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Histograma
    ax[0].hist(
        numeros, bins=resultado["k"], alpha=0.7, color="steelblue", edgecolor="black"
    )
    ax[0].axhline(
        resultado["freq_esp"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Esperado",
    )
    ax[0].set_xlabel("Valor")
    ax[0].set_ylabel("Frecuencia")
    ax[0].set_title(f"{titulo}: Histograma")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # Frecuencias
    x = np.arange(resultado["k"])
    ax[1].bar(x, resultado["freq_obs"], alpha=0.7, color="steelblue", label="Observado")
    ax[1].axhline(
        resultado["freq_esp"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Esperado",
    )
    ax[1].set_xlabel("Intervalo")
    ax[1].set_ylabel("Frecuencia")
    ax[1].set_title(f"{titulo}: Obs vs Esp")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    filename = os.path.join(output_dir, f"chi2_{titulo}.png")
    plt.savefig(filename, dpi=300)
    cons.print(f"[green]Guardado: {filename}")
    plt.show()


cons.rule("[bold red]Prueba Chi-Cuadrada")

print("\n¿Cómo saber si los números son aleatorios?")
print("Usamos la prueba chi-cuadrada (χ²) para verificar si son uniformes")
print("Si χ² calculado < χ² crítico -> SON uniformes :D")
print("Si χ² calculado ≥ χ² crítico -> NO SON uniformes D:<\n")

# --- Prueba 1: Nuestro LCG ---
cons.print("[cyan]Probando nuestro generador LCG...[/cyan]")
resultado_lcg = chi_cuadrada(lista_ale, M)

print(f"\nResultados LCG:")
print(f"  χ² calculado: {resultado_lcg['chi2']:.2f}")
print(f"  χ² crítico:   {resultado_lcg['critico']:.2f}")
print(
    f"  Resultado:    {':D Pasa la prueba' if resultado_lcg['pasa'] else 'D:< NO pasa'}\n"
)

graficar_chi2_simple(resultado_lcg, lista_ale, "LCG", output_dir)

# --- Prueba 2: random de Python ---
cons.print("[cyan]Probando random() de Python...[/cyan]")
N = len(lista_ale)
numeros_python = [random.randint(0, M - 1) for _ in range(N)]
resultado_python = chi_cuadrada(numeros_python, M)

print(f"\nResultados Python random:")
print(f"  χ² calculado: {resultado_python['chi2']:.2f}")
print(f"  χ² crítico:   {resultado_python['critico']:.2f}")
print(
    f"  Resultado:    {':D Pasa la prueba' if resultado_python['pasa'] else 'D:< NO pasa'}\n"
)

graficar_chi2_simple(resultado_python, numeros_python, "Python_random", output_dir)

# --- Comparación ---
cons.rule("[green]COMPARACIÓN")
print(
    f"\nLCG:           {':D Uniforme' if resultado_lcg['pasa'] else 'D:< NO uniforme'}"
)
print(
    f"Python random: {':D Uniforme' if resultado_python['pasa'] else 'D:< NO uniforme'}"
)

cons.print("\n[green]:D Tarea completa[/green]\n")
