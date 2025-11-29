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
from scipy.signal import find_peaks
from rich.panel import Panel
from rich import box
import matplotlib.pyplot as gp

cons = Console()
# Identificar partículas conocidas
particulas_conocidas = {
    'J/ψ': 3.097,  
    'ψ(2S)': 3.686,  
    'Υ(1S)': 9.460,  
    'Υ(2S)': 10.023,
    'Υ(3S)': 10.355,
    'Z⁰': 91.188  
}
# particulas_conocidas_lista = {
    # 'J/ψ': [3.097, 'Mesón de charmonio (c͞c)'],
    # 'ψ(2S)': [3.686, 'Excitación del J/ψ'],
    # 'Υ(1S)': [9.460, 'Mesón de bottomonio (b͞b)'],
    # 'Υ(2S)': [10.023, 'Primera excitación del Υ'],
    # 'Υ(3S)': [10.355, 'Segunda excitación del Υ'],
    # 'Z⁰': [91.188, 'Bosón Z (mediador débil)']
# }

#podría ser un diccionario, pero decidí ser feliz :D
descripciones = {
    'J/ψ': 'Mesón de charmonio (c͞c)',
    'ψ(2S)': 'Excitación del J/ψ',
    'Υ(1S)': 'Mesón de bottomonio (b͞b)',
    'Υ(2S)': 'Primera excitación del Υ',
    'Υ(3S)': 'Segunda excitación del Υ',
    'Z⁰': 'Bosón Z (mediador débil)'
}
#funciones auxiliares
def calculo_masa(E_1,px_1,py_1,pz_1,E_2,px_2,py_2,pz_2):
    """
    Me dio flojera documentarlo, pero calcula masas como se pidió
    fuentes: Vealo por usted mismo
    """
    # Energia total
    E_total = E_1 + E_2
    
    # momentos en componentes
    px_t= px_1+px_2
    py_t= py_1+py_2
    pz_t= pz_1+pz_2

    # Magnitud del momento total al cuadrado
    p2_total = px_t**2 + py_t**2 + pz_t**2
    
    # Masa invariante al cuadrado
    M2 = E_total**2 - p2_total
    
    # Masa invariante (tomar raíz cuadrada, evitar negativos por errores numéricos)
    M = st.sqrt(st.maximum(M2, 0))
    
    return M
# μ⁺μ⁻ - Datos del CMS Run 2011A
def histograma(masa, titulo,  events, is_log= False, colors = 'coral', edge_color = 'black', bins = 120):
    gp.figure(figsize=(14,12))
    counts, bin_edges, patches = gp.hist(masa, bins=bins,
            color= colors,
            edgecolor=edge_color,
            alpha=0.7,
            label=f'{events} eventos')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    if is_log:
        gp.xlabel("Masa Invariante (GeV/c²)", fontsize=14, fontweight='bold')
        gp.ylabel("log(Frecuencia)", fontsize=14, fontweight='bold')
        gp.title(f"Histograma de masas invariantes {titulo} (Escala Logarítmica - {bins} bins)", fontsize=16, fontweight='bold')
        gp.yscale('log')  # ¡Escala logarítmica en el eje Y!
        gp.legend(fontsize=12)
        gp.grid(True, alpha=0.3, which='both')
        gp.tight_layout()

        filename_log = os.path.join(output_dir, f"histograma_{titulo}_logaritmico.png")
        gp.savefig(filename_log, dpi=300, bbox_inches="tight")
        cons.print(f"[bold green] Histograma logarítmico guardado en:[/bold green] {filename_log}\n")
        gp.show()
        return counts, bin_centers, filename_log
    else:
        gp.ylabel("Frecuencia", fontsize=14, fontweight='bold')
        gp.xlabel("Masa invariante (GeV/c²)", fontsize=14, fontweight='bold')
        gp.title(f"Histograma de masas invariantes {titulo} ({bins} bins)")
        gp.legend(fontsize=12)
        gp.grid(True, alpha=0.3)
        # gp.xlim(mass.min(), mass.max())
        gp.tight_layout()

        filename = os.path.join(output_dir, f"histograma_masas_{titulo}.png")
        gp.savefig(filename, dpi=300, bbox_inches="tight")
        gp.show()
        return counts, bin_centers, filename
def counting_peaks(counts, bin_centers):
    """ Encontrar picos en el histograma
    Usar find_peaks para detectar resonancias automáticamente
    """
    peaks_indices, properties = find_peaks(counts, 
                                          height=st.max(counts)*0.05,  # Al menos 5% del máximo
                                          distance=5,  # Separación mínima entre picos
                                          prominence=100)
                                        #   ,width=(2,20))  # Prominencia mínima

    masas_picos = bin_centers[peaks_indices]
    alturas_picos = counts[peaks_indices]

    cons.print(f"\n[yellow]Resonancias detectadas: {len(masas_picos)}[/yellow]\n")

    # Tabla de resonancias detectadas
    table = Table(title="Resonancias Detectadas", box=box.DOUBLE)
    table.add_column("Pico", justify="center", style="cyan")
    table.add_column("Masa (GeV/c²)", justify="center", style="green")
    table.add_column("Eventos", justify="center", style="yellow")
    table.add_column("Candidato", justify="center", style="red")
    def identificar_particula(masa, tolerancia=0.5):
        """Identifica la partícula más cercana."""
        for nombre, masa_teorica in particulas_conocidas.items():
            if abs(masa - masa_teorica) < tolerancia:
                return f"{nombre} ({masa_teorica:.3f} GeV/c²)"
        return "Desconocida"

    for i, (masa_pico, altura_pico) in enumerate(zip(masas_picos, alturas_picos)):
        candidato = identificar_particula(masa_pico)
        table.add_row(
            f"#{i+1}",
            f"{masa_pico:.3f}",
            f"{int(altura_pico)}",
            candidato
        )

    cons.print(table)
    return masas_picos
def buscar_log(counts, bin_centers):
    """ Encontrar picos en el histograma
    Usar find_peaks para detectar resonancias automáticamente
    """
    # --- LIMITAR LA BÚSQUEDA A 60–110 GeV ---
    mask = (bin_centers >= 60) & (bin_centers <= 110)

    counts_roi = counts[mask]
    bins_roi   = bin_centers[mask]

    # --- DETECCIÓN DE PICO ---
    peaks_idx, props = find_peaks(
        counts_roi,
        prominence=st.max(counts_roi)*0.1,     # Evita picos falsos
        width=3,            # Ignora ruido de un solo bin
        distance=5          # No sub-picos
    )

    masas_picos = bins_roi[peaks_idx]
    alturas_picos = counts_roi[peaks_idx]

    cons.print(f"\n[yellow]Resonancias detectadas: {len(masas_picos)}[/yellow]\n")

    # Tabla de resonancias detectadas
    table = Table(title="Resonancias Detectadas", box=box.DOUBLE)
    table.add_column("Pico", justify="center", style="cyan")
    table.add_column("Masa (GeV/c²)", justify="center", style="green")
    table.add_column("Eventos", justify="center", style="yellow")
    table.add_column("Candidato", justify="center", style="red")
    def identificar_particula(masa, tolerancia=0.5):
        """Identifica la partícula más cercana."""
        for nombre, masa_teorica in particulas_conocidas.items():
            if abs(masa - masa_teorica) < tolerancia:
                return f"{nombre} ({masa_teorica:.3f} GeV/c²)"
        return "Desconocida"

    for i, (masa_pico, altura_pico) in enumerate(zip(masas_picos, alturas_picos)):
        candidato = identificar_particula(masa_pico,2.0)
        table.add_row(
            f"#{i+1}",
            f"{masa_pico:.3f}",
            f"{int(altura_pico)}",
            candidato
        )

    cons.print(table)
    return masas_picos
def stats(masa):
    cons.print("\n[bold yellow]Estadísticas de la masa invariante:[/bold yellow]")
    cons.print(f"[green] Masas calculadas: {len(masa):,} eventos[/green]")
    cons.print(f"\n[yellow]Estadísticas de masa:[/yellow]")
    cons.print(f"  Mínima: {masa.min():.3f} GeV/c²")
    cons.print(f"  Máxima: {masa.max():.3f} GeV/c²")
    cons.print(f"  Media: {masa.mean():.3f} GeV/c²")
    cons.print(f"  Mediana: {st.median(masa):.3f} GeV/c²")
def cargar_datos(path: str):
    df= ts.read_csv(path)

    numero_de_eventos = len(df)
    E_1 = df['E1'].to_numpy()
    px_1 = df['px1'].to_numpy()
    py_1 = df['py1'].to_numpy()
    pz_1 = df['pz1'].to_numpy()

    E_2 = df['E2'].to_numpy() 
    px_2 = df['px2'].to_numpy() 
    py_2 = df['py2'].to_numpy() 
    pz_2 = df['pz2'].to_numpy() 
    return df, numero_de_eventos, E_1,px_1,py_1,pz_1,E_2,px_2,py_2,pz_2

output_dir = "resultados_tarea_6"
cons.print(f"[bold] Verficando si existe el directorio {output_dir}...")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    cons.print(f"\n[bold red] El directorio {output_dir} no existe D:< ...")
    cons.print(f"\n[bold green] Directorio creado correctamente :DD")
else: cons.print(f"[bold green] {output_dir} si existe :D")

cons.rule("[bold cyan] Cargando los datos ...")
df , num_eventos, E_1,px_1,py_1, pz_1, E_2, px_2, py_2,pz_2 = cargar_datos("Jpsimumu_Run2011A.csv")

table = Table(title="[bold yellow]Primeros 15 datos del archivo Jpsimumu_run2011A.csv[/bold yellow]", box=box.ROUNDED)
columns = ["i","E_1","px_1","py_1","pz_1","E_2","px_2","py_2","pz_2"]

for c in columns:
    table.add_column(c, justify="center", style="magenta")

for i in range(15):
    table.add_row(f"{i+1}",f"{E_1[i]:.4f}",f"{px_1[i]:.4f}",f"{py_1[i]:.4f}",f"{pz_1[i]:.4f}",f"{E_2[i]:.4f}",f"{px_2[i]:.4f}",f"{py_2[i]:.4f}",f"{pz_2[i]:.4f}")
cons.print(table)

cons.rule("[bold blue] Calculando masas invariantes...")

mass = calculo_masa(E_1,px_1,py_1,pz_1,E_2,px_2,py_2,pz_2)

df["mass"] = mass

cons.print("[bold green]Masas calculadas correctamente.[/bold green]")

stats(mass)

cons.rule("[bold cyan]Generando histograma...[/bold cyan]")

counts , bin_centers, filename = histograma(mass, "μ⁺μ⁻ - Datos del CMS Run 2011A", len(mass))

cons.print(f"[bold green]Histograma guardado en:[/bold green] {filename}\n")

cons.rule("[bold cyan]Detectando picos en el histograma...[/bold cyan]")
# Encontrar picos en el histograma
masas_picos = counting_peaks(counts, bin_centers)

cons.print("\n[bold green]Análisis completado con éxito :D[/bold green]")
cons.rule("[bold cyan] COMPARACIÓN CON PARTICLE DATA GROUP[/bold cyan]")

# Tabla detallada de partículas
table_pdg = Table(title="Comparación con PDG (Particle Data Group)", box=box.DOUBLE_EDGE)
table_pdg.add_column("Partícula", justify="center", style="cyan")
table_pdg.add_column("Masa PDG (GeV/c²)", justify="center", style="green")
table_pdg.add_column("Masa Observada", justify="center", style="yellow")
table_pdg.add_column("Diferencia", justify="center", style="red")
table_pdg.add_column("Descripción", justify="left", style="blue")

for nombre, masa_pdg in particulas_conocidas.items():
    # Buscar si hay pico cerca
    diferencias = st.abs(masas_picos - masa_pdg)
    if len(diferencias) > 0 and st.min(diferencias) < 0.5:
        idx_cercano = st.argmin(diferencias)
        masa_obs = masas_picos[idx_cercano]
        diff = masa_obs - masa_pdg
        table_pdg.add_row(
            nombre,
            f"{masa_pdg:.3f}",
            f"{masa_obs:.3f} ± 0.010",
            f"{diff:+.3f}",
            descripciones[nombre]
        )
    else:
        table_pdg.add_row(
            nombre,
            f"{masa_pdg:.3f}",
            "No detectada",
            "—",
            descripciones[nombre]
        )

cons.print(table_pdg)

# parte 2 :D
cons.rule("[bold cyan] Cargando los datos para la parte 2 ...")
df , num_eventos, E_1,px_1,py_1, pz_1, E_2, px_2, py_2,pz_2 = cargar_datos("MuRun2010B.csv")

table = Table(title="[bold yellow]Primeros 15 datos del archivo MuRun2010B.csv[/bold yellow]", box=box.ROUNDED)
columns = ["i","E_1","px_1","py_1","pz_1","E_2","px_2","py_2","pz_2"]

for c in columns:
    table.add_column(c, justify="center", style="magenta")

for i in range(15):
    table.add_row(f"{i+1}",f"{E_1[i]:.4f}",f"{px_1[i]:.4f}",f"{py_1[i]:.4f}",f"{pz_1[i]:.4f}",f"{E_2[i]:.4f}",f"{px_2[i]:.4f}",f"{py_2[i]:.4f}",f"{pz_2[i]:.4f}")
cons.print(table)

cons.rule("[bold blue] Calculando masas invariantes...")

mass = calculo_masa(E_1,px_1,py_1,pz_1,E_2,px_2,py_2,pz_2)

df["mass"] = mass

cons.print("[bold green]Masas calculadas correctamente.[/bold green]")

stats(mass)

cons.rule("[bold cyan]Generando histograma...[/bold cyan]")


counts , bin_centers, filename = histograma(mass, "Bosón_Z_Run2018B_Lineal", len(mass), colors= 'blue', edge_color='coral')


cons.print(f"[bold green]Histograma guardado en:[/bold green] {filename}\n")
cons.rule("[bold cyan]Generando histograma logaritmico...[/bold cyan]")
log_counts , log_bin_centers, filename = histograma(mass, "Bosón_Z_Run2018B_Log", len(mass),True , 'forestgreen', 'coral')
cons.print(f"[bold green]Histograma guardado en:[/bold green] {filename}\n")
cons.rule("[bold cyan]Detectando picos en el histograma...[/bold cyan]")
# Encontrar picos en el histograma
masas_picos = buscar_log(log_counts, log_bin_centers)

cons.print("\n[bold green]Análisis completado con éxito :D[/bold green]")
cons.rule("[bold cyan] COMPARACIÓN CON PARTICLE DATA GROUP[/bold cyan]")

# Tabla detallada de partículas
table_pdg = Table(title="Comparación con PDG (Particle Data Group)", box=box.DOUBLE_EDGE)
table_pdg.add_column("Partícula", justify="center", style="cyan")
table_pdg.add_column("Masa PDG (GeV/c²)", justify="center", style="green")
table_pdg.add_column("Masa Observada", justify="center", style="yellow")
table_pdg.add_column("Diferencia", justify="center", style="red")
table_pdg.add_column("Descripción", justify="left", style="blue")

for nombre, masa_pdg in particulas_conocidas.items():
    # Buscar si hay pico cerca
    diferencias = st.abs(masas_picos - masa_pdg)
    if len(diferencias) > 0 and st.min(diferencias) < 2.0:
        idx_cercano = st.argmin(diferencias)
        masa_obs = masas_picos[idx_cercano]
        diff = masa_obs - masa_pdg
        table_pdg.add_row(
            nombre,
            f"{masa_pdg:.3f}",
            f"{masa_obs:.3f} ± 0.010",
            f"{diff:+.3f}",
            descripciones[nombre]
        )
    else:
        table_pdg.add_row(
            nombre,
            f"{masa_pdg:.3f}",
            "No detectada",
            "—",
            descripciones[nombre]
        )

cons.print(table_pdg)
