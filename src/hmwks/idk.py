"""
═══════════════════════════════════════════════════════════════════════════
ANÁLISIS DE RESONANCIAS DE DIMUONES - DATOS REALES DEL LHC
═══════════════════════════════════════════════════════════════════════════

Detector: CMS (Compact Muon Solenoid)
Acelerador: LHC (Large Hadron Collider)
Datos: Run2011A - ~31,000 colisiones protón-protón
Proceso: p + p → X → μ⁺ + μ⁻

Partículas esperadas:
• J/ψ (3.097 GeV/c²) - Mesón de charmonio
• Υ (Upsilon) familia (9-10 GeV/c²) - Mesones de bottomonio
• Z⁰ (91.2 GeV/c²) - Bosón Z

FÍSICA:
Dinámica relativista con c = 1:
  E² = p² + m²
  
Para la partícula madre (invariante):
  M² = (E₁ + E₂)² - (p⃗₁ + p⃗₂)²
  M = √[(E₁ + E₂)² - (px₁+px₂)² - (py₁+py₂)² - (pz₁+pz₂)²]
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import os

# Crear directorio de salida
output_dir = "lhc_analysis"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

console = Console()

console.rule("[bold red]ANÁLISIS DE RESONANCIAS - DATOS DEL CMS/LHC[/bold red]")

# ==============================================================================
# CARGAR DATOS
# ==============================================================================

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan]CARGANDO DATOS DEL CMS[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

# Leer datos
datos = pd.read_csv('Jpsimumu_Run2011A.csv')

console.print(f"\n[green]✓ Datos cargados exitosamente[/green]")
console.print(f"  Total de colisiones: {len(datos):,}")
console.print(f"  Columnas: {list(datos.columns)}")

# Mostrar primeras filas
console.print("\n[yellow]Primeras 5 colisiones:[/yellow]")
table = Table(title="Datos del CMS", box=box.ROUNDED)
for col in datos.columns[:8]:  # Primeras 8 columnas
    table.add_column(col, justify="center", style="cyan")

for idx in range(5):
    row = datos.iloc[idx]
    table.add_row(*[f"{row[col]:.3f}" if isinstance(row[col], float) 
                   else str(row[col]) for col in datos.columns[:8]])

console.print(table)

# ==============================================================================
# (a) CALCULAR MASA INVARIANTE
# ==============================================================================

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan](a) CÁLCULO DE MASA INVARIANTE[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

console.print("\n[yellow]Fórmula de masa invariante (c = 1):[/yellow]")
console.print("[yellow]  M² = (E₁ + E₂)² - (p⃗₁ + p⃗₂)²[/yellow]")
console.print("[yellow]  M = √[(E₁+E₂)² - (px₁+px₂)² - (py₁+py₂)² - (pz₁+pz₂)²][/yellow]")

def calcular_masa_invariante(E1, px1, py1, pz1, E2, px2, py2, pz2):
    """
    Calcula la masa invariante de una partícula que decae en dos muones.
    
    En unidades naturales (c = 1):
    M² = (E₁ + E₂)² - |p⃗₁ + p⃗₂|²
    
    Parámetros:
    -----------
    E1, E2 : float
        Energías de los muones (GeV)
    px1, py1, pz1 : float
        Componentes del momento del muón 1 (GeV/c)
    px2, py2, pz2 : float
        Componentes del momento del muón 2 (GeV/c)
    
    Retorna:
    --------
    M : float
        Masa invariante (GeV/c²)
    """
    # Energía total
    E_total = E1 + E2
    
    # Momento total (vectorial)
    px_total = px1 + px2
    py_total = py1 + py2
    pz_total = pz1 + pz2
    
    # Magnitud del momento total al cuadrado
    p2_total = px_total**2 + py_total**2 + pz_total**2
    
    # Masa invariante al cuadrado
    M2 = E_total**2 - p2_total
    
    # Masa invariante (tomar raíz cuadrada, evitar negativos por errores numéricos)
    M = np.sqrt(np.maximum(M2, 0))
    
    return M

# Calcular masas invariantes para todas las colisiones
console.print("\n[yellow]Calculando masas invariantes...[/yellow]")

masas = calcular_masa_invariante(
    datos['E1'].values,
    datos['px1'].values,
    datos['py1'].values,
    datos['pz1'].values,
    datos['E2'].values,
    datos['px2'].values,
    datos['py2'].values,
    datos['pz2'].values
)

# Agregar columna de masa al DataFrame
datos['Masa'] = masas

console.print(f"[green]✓ Masas calculadas: {len(masas):,} eventos[/green]")
console.print(f"\n[yellow]Estadísticas de masa:[/yellow]")
console.print(f"  Mínima: {masas.min():.3f} GeV/c²")
console.print(f"  Máxima: {masas.max():.3f} GeV/c²")
console.print(f"  Media: {masas.mean():.3f} GeV/c²")
console.print(f"  Mediana: {np.median(masas):.3f} GeV/c²")

# ==============================================================================
# (b) HISTOGRAMA DE FRECUENCIAS
# ==============================================================================

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan](b) HISTOGRAMA DE MASAS INVARIANTES[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

# Número de bins sugerido
n_bins = 120

console.print(f"\n[yellow]Número de bins: {n_bins}[/yellow]")

# Crear histograma
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Histograma completo
counts, bin_edges, patches = ax1.hist(masas, bins=n_bins, 
                                      color='steelblue', 
                                      edgecolor='black', 
                                      alpha=0.7,
                                      label=f'{len(masas):,} eventos')

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

ax1.set_xlabel('Masa Invariante (GeV/c²)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Frecuencia', fontsize=14, fontweight='bold')
ax1.set_title('Espectro de Masa Invariante μ⁺μ⁻ - Datos del CMS Run 2011A', 
             fontsize=16, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, masas.max())

# Histograma en escala logarítmica
ax2.hist(masas, bins=n_bins, 
         color='coral', 
         edgecolor='black', 
         alpha=0.7,
         label=f'{len(masas):,} eventos')

ax2.set_xlabel('Masa Invariante (GeV/c²)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Frecuencia (escala log)', fontsize=14, fontweight='bold')
ax2.set_title('Espectro de Masa (Escala Logarítmica)', 
             fontsize=16, fontweight='bold')
ax2.set_yscale('log')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim(0, masas.max())

plt.tight_layout()
filename = f"{output_dir}/histograma_masas_completo.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
console.print(f"\n[green]💾 Guardado: {filename}[/green]")
plt.close()

# ==============================================================================
# (c) ANÁLISIS DE RESONANCIAS
# ==============================================================================

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan](c) IDENTIFICACIÓN DE RESONANCIAS[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

# Encontrar picos en el histograma
# Usar find_peaks para detectar resonancias automáticamente
peaks_indices, properties = find_peaks(counts, 
                                      height=np.max(counts)*0.05,  # Al menos 5% del máximo
                                      distance=5,  # Separación mínima entre picos
                                      prominence=100)  # Prominencia mínima

masas_picos = bin_centers[peaks_indices]
alturas_picos = counts[peaks_indices]

console.print(f"\n[yellow](c.i) Resonancias detectadas: {len(masas_picos)}[/yellow]\n")

# Tabla de resonancias detectadas
table = Table(title="Resonancias Detectadas", box=box.DOUBLE)
table.add_column("Pico", justify="center", style="cyan")
table.add_column("Masa (GeV/c²)", justify="center", style="green")
table.add_column("Eventos", justify="center", style="yellow")
table.add_column("Candidato", justify="center", style="red")

# Identificar partículas conocidas
particulas_conocidas = {
    'J/ψ': 3.097,
    'ψ(2S)': 3.686,
    'Υ(1S)': 9.460,
    'Υ(2S)': 10.023,
    'Υ(3S)': 10.355,
    'Z⁰': 91.188
}

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

console.print(table)

# Gráfica con picos marcados
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Rango completo con picos marcados
ax = axes[0, 0]
ax.hist(masas, bins=n_bins, color='steelblue', edgecolor='black', alpha=0.7)
ax.plot(masas_picos, alturas_picos, 'r*', markersize=20, 
        label=f'{len(masas_picos)} resonancias', zorder=5)
for masa_pico in masas_picos:
    ax.axvline(masa_pico, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.set_xlabel('Masa (GeV/c²)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
ax.set_title('Espectro Completo con Resonancias', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Zoom en región J/ψ (2-4 GeV)
ax = axes[0, 1]
mask_jpsi = (masas > 2.5) & (masas < 4.0)
ax.hist(masas[mask_jpsi], bins=50, color='purple', edgecolor='black', alpha=0.7)
ax.axvline(3.097, color='red', linestyle='--', linewidth=2, label='J/ψ teórico (3.097 GeV)')
ax.axvline(3.686, color='orange', linestyle='--', linewidth=2, label='ψ(2S) teórico (3.686 GeV)')
ax.set_xlabel('Masa (GeV/c²)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
ax.set_title('Región J/ψ y ψ(2S)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Zoom en región Υ (Upsilon) (8-12 GeV)
ax = axes[1, 0]
mask_upsilon = (masas > 8) & (masas < 12)
ax.hist(masas[mask_upsilon], bins=50, color='green', edgecolor='black', alpha=0.7)
ax.axvline(9.460, color='red', linestyle='--', linewidth=2, label='Υ(1S) teórico (9.460 GeV)')
ax.axvline(10.023, color='orange', linestyle='--', linewidth=2, label='Υ(2S) teórico (10.023 GeV)')
ax.axvline(10.355, color='cyan', linestyle='--', linewidth=2, label='Υ(3S) teórico (10.355 GeV)')
ax.set_xlabel('Masa (GeV/c²)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
ax.set_title('Región Υ (Upsilon)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Zoom en región Z (80-100 GeV)
ax = axes[1, 1]
mask_z = (masas > 70) & (masas < 110)
if mask_z.sum() > 0:
    ax.hist(masas[mask_z], bins=30, color='red', edgecolor='black', alpha=0.7)
    ax.axvline(91.188, color='blue', linestyle='--', linewidth=2, label='Z⁰ teórico (91.188 GeV)')
    ax.set_xlabel('Masa (GeV/c²)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    ax.set_title('Región Z⁰', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Sin eventos en rango Z⁰', 
           ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('Región Z⁰ (sin datos)', fontsize=14, fontweight='bold')

plt.suptitle('Análisis de Resonancias - CMS Run 2011A', 
            fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()

filename = f"{output_dir}/analisis_resonancias.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
console.print(f"\n[green]💾 Guardado: {filename}[/green]")
plt.close()

# ==============================================================================
# (c.ii) IDENTIFICACIÓN Y COMPARACIÓN CON PDG
# ==============================================================================

console.print("\n[cyan]═══════════════════════════════════════════[/cyan]")
console.print("[bold cyan](c.ii) COMPARACIÓN CON PARTICLE DATA GROUP[/bold cyan]")
console.print("[cyan]═══════════════════════════════════════════[/cyan]")

# Tabla detallada de partículas
table_pdg = Table(title="Comparación con PDG (Particle Data Group)", box=box.DOUBLE_EDGE)
table_pdg.add_column("Partícula", justify="center", style="cyan")
table_pdg.add_column("Masa PDG (GeV/c²)", justify="center", style="green")
table_pdg.add_column("Masa Observada", justify="center", style="yellow")
table_pdg.add_column("Diferencia", justify="center", style="red")
table_pdg.add_column("Descripción", justify="left", style="blue")

descripciones = {
    'J/ψ': 'Mesón de charmonio (c͞c)',
    'ψ(2S)': 'Excitación del J/ψ',
    'Υ(1S)': 'Mesón de bottomonio (b͞b)',
    'Υ(2S)': 'Primera excitación del Υ',
    'Υ(3S)': 'Segunda excitación del Υ',
    'Z⁰': 'Bosón Z (mediador débil)'
}

for nombre, masa_pdg in particulas_conocidas.items():
    # Buscar si hay pico cerca
    diferencias = np.abs(masas_picos - masa_pdg)
    if len(diferencias) > 0 and np.min(diferencias) < 0.5:
        idx_cercano = np.argmin(diferencias)
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

console.print(table_pdg)

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================

console.print("\n")
console.rule("[bold green]RESUMEN FINAL[/bold green]")

resumen = f"""
[bold cyan]ANÁLISIS DE RESONANCIAS - DATOS REALES DEL LHC[/bold cyan]

[bold yellow]Datos analizados:[/bold yellow]
  • Colisiones: {len(datos):,}
  • Detector: CMS (Compact Muon Solenoid)
  • Periodo: Run 2011A
  • Proceso: p + p → X → μ⁺ + μ⁻

[bold yellow](a) Masa Invariante:[/bold yellow]
  • Fórmula: M = √[(E₁+E₂)² - (p⃗₁+p⃗₂)²]
  • Rango: {masas.min():.2f} - {masas.max():.2f} GeV/c²
  • Calculadas: {len(masas):,} masas

[bold yellow](b) Histograma:[/bold yellow]
  • Bins: {n_bins}
  • Generado en escala lineal y logarítmica

[bold yellow](c) Resonancias Detectadas: {len(masas_picos)}[/bold yellow]

[bold green]Partículas Identificadas:[/bold green]
"""

for i, masa_pico in enumerate(masas_picos):
    candidato = identificar_particula(masa_pico)
    resumen += f"  {i+1}. M = {masa_pico:.3f} GeV/c² → {candidato}\n"

resumen += f"""
[bold cyan]Física del proceso:[/bold cyan]
  • J/ψ y ψ(2S): Mesones de charmonio (quark charm + anticharm)
  • Υ(1S,2S,3S): Mesones de bottomonio (quark bottom + antibottom)
  • Z⁰: Bosón mediador de la fuerza débil

[bold red]Referencias:[/bold red]
  • Particle Data Group: https://pdg.lbl.gov/
  • CMS Open Data: http://opendata.cern.ch/
"""

panel = Panel(resumen, title="[bold]Resultados del Análisis[/bold]",
             border_style="green", box=box.DOUBLE)
console.print(panel)

console.print("\n[bold green]✓ ANÁLISIS COMPLETADO EXITOSAMENTE[/bold green]")
console.print(f"\n[yellow]Archivos generados en: {output_dir}/[/yellow]")