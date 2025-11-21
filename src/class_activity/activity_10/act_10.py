"""
Oscar A. Valencia Magaña
20/11/2025
Elabore un programa que simule n experimentos para realizar una es-
timaciÛn del siguiente problema. El mÈtodo del ritmo para control de

la natalidad se sabe que es 70% efectivo. Esto es, la probabilidad de
que alguien que utilice este mÈtodo quede embarazada en un aÒo es
de 30%. øCu·l es el n ̇mero esperado de aÒos antes de que alguien
que lo usa quede embarazada? Construya una tabla donde reporte los
valores de la media (valor esperado estimado) y la desviaciÛn estandar
con n = 1000; 10000 y 100000: Puede utilizar las funciones numpy.mean
y numpy.std para calcular la media y la desviaciÛn estandar (ver la
documentaciÛn pertinente).
"""

import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

cons = Console()

def simular_anios_hasta_embarazo(n, p=0.3):
    """
    Simula n experimentos donde cada experimento consiste en 
    contar cuántos años pasan antes de quedar embarazada.
    """
    resultados = []
    for _ in range(n):
        anios = 0
        while True:
            anios += 1
            if np.random.rand() < p:  # sucede embarazo
                resultados.append(anios)
                break
    return np.array(resultados)

# Valores de n solicitados
ns = [1000, 10000, 100000]
s=0
while s <= 9: 
    cons.rule("[bold green] Simulación de años hasta un embarazo número "+ str(s+1))
    table = Table(title="[bold cyan] Simulaciones [/bold cyan]", box=box.ROUNDED)
    table.add_column("n", justify="right", style="cyan", no_wrap=True)
    table.add_column("Media (años)", justify="right", style="magenta")
    table.add_column("Desviación estandar", justify="right", style="green")

    for n in ns:
        datos = simular_anios_hasta_embarazo(n)
        media = np.mean(datos)
        std = np.std(datos)
        table.add_row(str(n), f"{media:.4f}", f"{std:.4f}")

    cons.print(table)
    s +=1