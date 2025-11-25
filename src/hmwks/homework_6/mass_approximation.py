"""
¿Qué partículas son? El objetivo de este ejercicio es estimar la masa  
de una partícula que decae en dos muones. Los datos son reales tomados del  

CMS (Compact Muon Solenoid) que han sido adquiridos, analizados, filtra-  
dos e identificados como colisiones en el LHC (Large Hadron Collider) y que  

presentan un par muón–antimuón, conocidos usualmente como dimuones, se-  
leccionados para obtener eventos que son candidatos para observar partículas  

J/ψ, Υ, W y Z. En el archivo adjunto **Jpsimumu_Run2011A.csv** se pre-  
sentan los datos de poco más de 31 000 colisiones. Las columnas en la tabla  

corresponden a
"""
import pandas as ts
import numpy as st
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import matplotlib.pyplot as gp

cons = Console()

output_dir = "resultados_Tarea_6"
cons.print(f"[bold] Verficando si existe el directorio {output_dir}...")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    cons.print(f"\n[bold red] El directorio {output_dir} no existe D:< ...")
    cons.print(f"\n[bold green] Directorio creado correctamente :DD")
else: cons.print(f"[bold green] {output_dir} si existe :D")
cons.rule("[bold cyan] Cargando los datos ...")
df = ts.read_csv("Jpsimumu_Run2011A.csv")

num_eventos = len(df) 

E_1 = df['E1'].to_numpy()
px_1 = df['px1'].to_numpy()
py_1 = df['py1'].to_numpy()
pz_1 = df['pz1'].to_numpy()

E_2 = df['E2'].to_numpy() 
px_2 = df['px2'].to_numpy() 
py_2 = df['py2'].to_numpy() 
pz_2 = df['pz2'].to_numpy() 

table = Table(title="[bold yellow]Primeros 15 datos del archivo Jpsimumu_run2011A.csv[/bold yellow]", box=box.ROUNDED)
columns = ["i","E_1","px_1","py_1","pz_1","E_2","px_2","py_2","pz_2"]

for c in columns:
    table.add_column(c, justify="center", style="magenta")

for i in range(15):
    table.add_row(f"{i+1}",f"{E_1[i]:.4f}",f"{px_1[i]:.4f}",f"{py_1[i]:.4f}",f"{pz_1[i]:.4f}",f"{E_2[i]:.4f}",f"{px_2[i]:.4f}",f"{py_2[i]:.4f}",f"{pz_2[i]:.4f}")
cons.print(table)

cons.rule("[bold blue] Calculando masas invariantes...")

E = E_1 + E_2
px = px_1 + px_2
py = py_1 + py_2
pz = pz_1 + pz_2

m2 = E**2 - (px**2 + py**2 + pz**2)
m2 = st.clip(m2, 0, None)   # evita negativos por redondeo
mass = st.sqrt(m2)

df["mass"] = mass

cons.print("[bold green]Masas calculadas correctamente.[/bold green]")

cons.print("\n[bold yellow]Estadísticas de la masa invariante:[/bold yellow]")
# cons.print(df["mass"].describe())
cons.print(f"[green]✓ Masas calculadas: {len(mass):,} eventos[/green]")
cons.print(f"\n[yellow]Estadísticas de masa:[/yellow]")
cons.print(f"  Mínima: {mass.min():.3f} GeV/c²")
cons.print(f"  Máxima: {mass.max():.3f} GeV/c²")
cons.print(f"  Media: {mass.mean():.3f} GeV/c²")
cons.print(f"  Mediana: {st.median(mass):.3f} GeV/c²")
cons.rule("[bold cyan]Generando histograma...[/bold cyan]")

bins = 120  # >100 bins como pide la tarea

gp.figure(figsize=(14,12))
gp.hist(df["mass"], bins=bins,
        color='coral',
        edgecolor='black',
        alpha=0.7,
        label=f'{num_eventos} eventos')
gp.xlabel("Masa invariante (GeV/c²)", fontsize=14, fontweight='bold')
gp.ylabel("Frecuencia", fontsize=14, fontweight='bold')
gp.title("Histograma de masas invariantes (120 bins)")
gp.legend(fontsize=12)
gp.grid(True, alpha=0.3)
gp.xlim(mass.min(), mass.max())
gp.tight_layout()

filename = os.path.join(output_dir, "histograma_masas_Jpsimumu_Run2011A.png")
gp.savefig(filename, dpi=300, bbox_inches="tight")
gp.show()

cons.print(f"[bold green]Histograma guardado en:[/bold green] {filename}\n")

cons.rule("[bold cyan]Detectando picos en el histograma...[/bold cyan]")

counts, bin_edges = st.histogram(df["mass"], bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

peaks_idx = []
for i in range(1, len(counts) - 1):
    if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
        peaks_idx.append(i)

peak_masses = bin_centers[peaks_idx]
peak_counts = counts[peaks_idx]

# Ordena los picos por altura
ordenados = sorted(zip(peak_masses, peak_counts), key=lambda x: x[1], reverse=True)

# Mostrar los 10 picos más grandes
table_picos = Table(title="[bold magenta]Picos detectados[/bold magenta]", box=box.ROUNDED)
table_picos.add_column("Masa (GeV)", justify="right", style="green")
table_picos.add_column("Cuentas", justify="right", style="yellow")

for masa, cnt in ordenados[:10]:
    table_picos.add_row(f"{masa:.3f}", f"{cnt}")

cons.print(table_picos)

cons.print("\n[bold green]Análisis completado con éxito :D[/bold green]")