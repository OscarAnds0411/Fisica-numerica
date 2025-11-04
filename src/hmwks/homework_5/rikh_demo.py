#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demostración de la librería Rich (rich.console)
Muestra texto con estilo, tablas, paneles, logs y barra de progreso.

Autor: Oscar Valencia
Fecha: Noviembre 2025
"""

from time import sleep
from io import StringIO
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

# Crear una consola principal
console = Console()

# --- 1. Texto con estilo ---
console.rule("[bold blue]DEMOSTRACIÓN DE RICH[/bold blue]")
console.print("Hola [bold green]mundo[/bold green]! 🌎", style="bold white on black")
console.print("Texto con color hexadecimal", style="#ff8800")
console.print("Texto con fondo y múltiples estilos", style="bold underline magenta on yellow")

# --- 2. Panel ---
panel = Panel(
    "[bold cyan]Rich[/bold cyan] te permite crear salidas de texto con formato, colores y diseños enriquecidos.",
    title="[white]Panel de Ejemplo[/white]",
    subtitle="[italic green]Fácil y elegante[/italic green]",
)
console.print(panel)

# --- 3. Tabla ---
table = Table(title="[bold yellow]Tabla de Resultados[/bold yellow]")
table.add_column("Nombre", justify="left", style="cyan")
table.add_column("Puntaje", justify="center", style="magenta")
table.add_column("Aprobado", justify="center", style="green")

table.add_row("Alice", "89", "✅")
table.add_row("Bob", "72", "✅")
table.add_row("Carlos", "58", "❌")

console.print(table)

# --- 4. Logs y estado ---
console.rule("[bold red]LOGS Y ESTADO[/bold red]")

with console.status("[bold green]Procesando datos...[/bold green]"):
    for step in range(3):
        sleep(1)
        console.log(f"Paso {step + 1} completado")

console.log("[bold green]Proceso finalizado exitosamente ✔️[/bold green]")

# --- 5. Barra de progreso ---
console.rule("[bold blue]BARRA DE PROGRESO[/bold blue]")
for i in track(range(10), description="[yellow]Cargando...[/yellow]"):
    sleep(0.2)

# --- 6. Inspección de objeto ---
#console.rule("[bold cyan]INSPECCIÓN DE OBJETO[/bold cyan]")
#data = {"usuario": "Oscar", "rol": "admin", "activo": True}
#console.inspect(data, methods=True)

# --- 7. Captura de salida ---
buf = StringIO()
capture_console = Console(file=buf)
capture_console.print("[bold magenta]Salida capturada correctamente[/bold magenta]")
output = buf.getvalue()

console.rule("[bold white]CAPTURA DE CONSOLA[/bold white]")
console.print(output)

# --- 8. Cierre ---
console.rule("[bold green]FIN DE LA DEMOSTRACIÓN[/bold green]")
console.print("[bold white on blue]Ejemplo completado con éxito![/bold white on blue]")
