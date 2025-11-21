import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

cons = Console()

def simular_anios_hasta_embarazo(n, p=0.3):
    """
    Simula n experimentos y devuelve un arreglo con los años necesarios
    para que ocurra un embarazo en cada experimento.
    """
    resultados = []
    for _ in range(n):
        anios = 0
        while np.random.rand() >= p:
            anios += 1
        resultados.append(anios)
    return np.array(resultados)

# Valores de n solicitados
ns = [1000, 10000, 100000]

for s in range(10):
    cons.rule(f"[bold green] Simulación de años hasta embarazo número {s+1}")
    
    table = Table(title="[bold cyan]Resultados de la Simulación[/bold cyan]", box=box.ROUNDED)
    table.add_column("n", justify="right", style="cyan")
    table.add_column("Media (años)", justify="right", style="magenta")
    table.add_column("Desviación estándar", justify="right", style="green")
    
    for n in ns:
        datos = simular_anios_hasta_embarazo(n)
        media = np.mean(datos)
        std = np.std(datos)
        table.add_row(str(n), f"{media:.4f}", f"{std:.4f}")
    
    cons.print(table)