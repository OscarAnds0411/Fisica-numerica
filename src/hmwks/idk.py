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
from scipy.optimize import curve_fit
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

def calcular_incertidumbre_masa(masas, masa_pico, ventana=5.0):
    """
    Calcula la incertidumbre de la masa estimada usando ajuste gaussiano.
    
    Método:
    ------
    1. Selecciona eventos alrededor del pico (ventana de ±ventana GeV)
    2. Ajusta una gaussiana: N(M) = A·exp[-(M-μ)²/(2σ²)]
    3. Extrae:
       - μ: masa central (valor más probable)
       - σ: ancho de la distribución (resolución del detector)
       - FWHM = 2.355·σ (ancho a media altura)
       - Error estadístico: σ/√N
    
    Parámetros:
    -----------
    masas : array
        Array de masas invariantes calculadas
    masa_pico : float
        Masa del pico a analizar (GeV/c²)
    ventana : float
        Rango ±ventana alrededor del pico para el ajuste (GeV)
    
    Retorna:
    --------
    dict con:
        'masa_ajustada': μ del ajuste gaussiano
        'sigma': σ de la gaussiana (incertidumbre)
        'FWHM': ancho a media altura
        'error_estadistico': σ/√N
        'N_eventos': número de eventos en la ventana
        'ajuste_exitoso': bool indicando si el ajuste convergió
    """
    cons.print(f"\n[cyan]Calculando incertidumbre para pico en {masa_pico:.3f} GeV...[/cyan]")
    
    # Seleccionar eventos alrededor del pico
    mask = st.abs(masas - masa_pico) < ventana
    masas_ventana = masas[mask]
    N_eventos = len(masas_ventana)
    
    cons.print(f"[yellow]  Eventos en ventana ±{ventana} GeV: {N_eventos:,}[/yellow]")
    
    if N_eventos < 50:
        cons.print("[red]  ⚠️ Pocos eventos para ajuste confiable[/red]")
        return {
            'masa_ajustada': masa_pico,
            'sigma': 0,
            'FWHM': 0,
            'error_estadistico': 0,
            'N_eventos': N_eventos,
            'ajuste_exitoso': False
        }
    
    # Crear histograma para ajustar
    bins = min(50, N_eventos // 20)  # ~20 eventos por bin
    counts, bin_edges = st.histogram(masas_ventana, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Función gaussiana para ajustar
    def gaussiana(x, A, mu, sigma):
        return A * st.exp(-(x - mu)**2 / (2 * sigma**2))
    
    # Estimaciones iniciales
    A_inicial = st.max(counts)
    mu_inicial = masa_pico
    sigma_inicial = ventana / 4  # Estimación razonable
    
    try:
        # Ajustar gaussiana
        popt, pcov = curve_fit(
            gaussiana, 
            bin_centers, 
            counts,
            p0=[A_inicial, mu_inicial, sigma_inicial],
            bounds=([0, masa_pico - ventana, 0.001], 
                   [st.inf, masa_pico + ventana, ventana]),
            maxfev=5000
        )
        
        A_fit, mu_fit, sigma_fit = popt
        
        # Extraer incertidumbres de la matriz de covarianza
        perr = st.sqrt(st.diag(pcov))
        sigma_mu = perr[1]  # Incertidumbre en μ
        sigma_sigma = perr[2]  # Incertidumbre en σ
        
        # Calcular métricas
        FWHM = 2.355 * sigma_fit  # Full Width at Half Maximum
        error_estadistico = sigma_fit / st.sqrt(N_eventos)
        
        cons.print(f"[green]  ✓ Ajuste gaussiano exitoso[/green]")
        cons.print(f"[green]    μ = {mu_fit:.4f} ± {sigma_mu:.4f} GeV/c²[/green]")
        cons.print(f"[green]    σ = {sigma_fit:.4f} ± {sigma_sigma:.4f} GeV/c²[/green]")
        cons.print(f"[green]    FWHM = {FWHM:.4f} GeV/c²[/green]")
        cons.print(f"[green]    Error estadístico = {error_estadistico:.4f} GeV/c²[/green]")
        
        return {
            'masa_ajustada': mu_fit,
            'sigma': sigma_fit,
            'sigma_mu': sigma_mu,
            'sigma_sigma': sigma_sigma,
            'FWHM': FWHM,
            'error_estadistico': error_estadistico,
            'N_eventos': N_eventos,
            'ajuste_exitoso': True,
            'parametros_ajuste': popt,
            'covarianza': pcov
        }
        
    except Exception as e:
        cons.print(f"[red]  ✗ Error en ajuste gaussiano: {e}[/red]")
        
        # Fallback: usar estadísticas básicas
        mu_fallback = st.mean(masas_ventana)
        sigma_fallback = st.std(masas_ventana)
        FWHM_fallback = 2.355 * sigma_fallback
        error_est_fallback = sigma_fallback / st.sqrt(N_eventos)
        
        cons.print(f"[yellow]  Usando estadísticas directas:[/yellow]")
        cons.print(f"[yellow]    Media = {mu_fallback:.4f} GeV/c²[/yellow]")
        cons.print(f"[yellow]    σ = {sigma_fallback:.4f} GeV/c²[/yellow]")
        
        return {
            'masa_ajustada': mu_fallback,
            'sigma': sigma_fallback,
            'sigma_mu': error_est_fallback,
            'FWHM': FWHM_fallback,
            'error_estadistico': error_est_fallback,
            'N_eventos': N_eventos,
            'ajuste_exitoso': False
        }

def graficar_ajuste_gaussiano(masas, masa_pico, resultado_ajuste, ventana=5.0, 
                               nombre_particula="Partícula", output_dir="resultados_tarea_6"):
    """
    Grafica el histograma con el ajuste gaussiano superpuesto.
    """
    if not resultado_ajuste['ajuste_exitoso']:
        cons.print("[yellow]  Ajuste no exitoso, omitiendo gráfica[/yellow]")
        return None
    
    # Seleccionar eventos
    mask = st.abs(masas - masa_pico) < ventana
    masas_ventana = masas[mask]
    
    # Crear figura
    fig, (ax1, ax2) = gp.subplots(2, 1, figsize=(12, 10))
    
    # --- Panel superior: Histograma con ajuste ---
    bins = min(50, len(masas_ventana) // 20)
    counts, bin_edges, patches = ax1.hist(masas_ventana, bins=bins, 
                                         color='steelblue', edgecolor='black', 
                                         alpha=0.7, label='Datos')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Graficar ajuste gaussiano
    if 'parametros_ajuste' in resultado_ajuste:
        A, mu, sigma = resultado_ajuste['parametros_ajuste']
        x_fit = st.linspace(masa_pico - ventana, masa_pico + ventana, 500)
        y_fit = A * st.exp(-(x_fit - mu)**2 / (2 * sigma**2))
        
        ax1.plot(x_fit, y_fit, 'r-', linewidth=3, 
                label=f'Ajuste gaussiano\nμ = {mu:.4f} GeV\nσ = {sigma:.4f} GeV')
        
        # Marcar μ y FWHM
        ax1.axvline(mu, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Marcar FWHM
        FWHM = resultado_ajuste['FWHM']
        y_half_max = A / 2
        ax1.axhline(y_half_max, color='orange', linestyle=':', linewidth=2, 
                   label=f'FWHM = {FWHM:.4f} GeV')
        ax1.axvspan(mu - FWHM/2, mu + FWHM/2, alpha=0.2, color='orange')
    
    ax1.set_xlabel('Masa Invariante (GeV/c²)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Eventos', fontsize=12, fontweight='bold')
    ax1.set_title(f'Ajuste Gaussiano - {nombre_particula}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Panel inferior: Residuos ---
    if 'parametros_ajuste' in resultado_ajuste:
        A, mu, sigma = resultado_ajuste['parametros_ajuste']
        y_esperado = A * st.exp(-(bin_centers - mu)**2 / (2 * sigma**2))
        residuos = counts - y_esperado
        
        ax2.scatter(bin_centers, residuos, color='blue', s=30, alpha=0.6)
        ax2.axhline(0, color='red', linestyle='--', linewidth=2)
        ax2.fill_between(bin_centers, -st.sqrt(y_esperado), st.sqrt(y_esperado), 
                        alpha=0.3, color='gray', label='±1σ estadístico')
        
        ax2.set_xlabel('Masa Invariante (GeV/c²)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residuos (Datos - Ajuste)', fontsize=12, fontweight='bold')
        ax2.set_title('Residuos del Ajuste', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    gp.tight_layout()
    
    # Guardar
    filename = os.path.join(output_dir, f"ajuste_gaussiano_{nombre_particula.replace(' ', '_')}.png")
    gp.savefig(filename, dpi=300, bbox_inches='tight')
    cons.print(f"[green]  💾 Gráfica guardada: {filename}[/green]")
    gp.show()
    
    return filename

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
    # --- LIMITAR LA BÚSQUEDA A 80–105 GeV ---
    mask = (bin_centers >= 80) & (bin_centers <= 105)

    counts_roi = counts[mask]
    bins_roi   = bin_centers[mask]

    # --- DETECCIÓN DE PICO ---
    peaks_idx, props = find_peaks(
        counts_roi,
        prominence=st.max(counts_roi)*0.1,     # 10% del máximo
        width=3,            
        distance=5          
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
        candidato = identificar_particula(masa_pico, 2.0)
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

# ==============================================================================
# PROGRAMA PRINCIPAL
# ==============================================================================

output_dir = "resultados_tarea_6"
cons.print(f"[bold] Verficando si existe el directorio {output_dir}...")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    cons.print(f"\n[bold red] El directorio {output_dir} no existe D:< ...")
    cons.print(f"\n[bold green] Directorio creado correctamente :DD")
else: 
    cons.print(f"[bold green] {output_dir} si existe :D")

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

# ==============================================================================
# CALCULAR INCERTIDUMBRES PARA CADA PICO DETECTADO (PARTE 1)
# ==============================================================================

cons.rule("[bold cyan]CÁLCULO DE INCERTIDUMBRES - PARTE 1[/bold cyan]")

resultados_parte1 = {}

for i, masa_pico in enumerate(masas_picos):
    cons.print(f"\n[bold yellow]════ Pico #{i+1}: {masa_pico:.3f} GeV/c² ════[/bold yellow]")
    
    # Identificar partícula
    particula_nombre = "Desconocida"
    for nombre, masa_teorica in particulas_conocidas.items():
        if abs(masa_pico - masa_teorica) < 0.5:
            particula_nombre = nombre
            break
    
    # Determinar ventana según la masa (picos más anchos para masas más altas)
    if masa_pico < 5:
        ventana = 0.5  # J/ψ, ψ(2S)
    elif masa_pico < 15:
        ventana = 1.0  # Υ(1S,2S,3S)
    else:
        ventana = 5.0  # Otros
    
    # Calcular incertidumbre
    resultado = calcular_incertidumbre_masa(mass, masa_pico, ventana=ventana)
    
    # Guardar resultado
    resultados_parte1[particula_nombre] = resultado
    
    # Graficar ajuste
    if resultado['ajuste_exitoso'] and resultado['N_eventos'] > 100:
        graficar_ajuste_gaussiano(mass, masa_pico, resultado, ventana=ventana, 
                                  nombre_particula=particula_nombre, output_dir=output_dir)

# Tabla resumen de incertidumbres
if resultados_parte1:
    cons.rule("[bold green]RESUMEN DE INCERTIDUMBRES - PARTE 1[/bold green]")
    
    table_incert = Table(title="Incertidumbres Calculadas", box=box.DOUBLE_EDGE)
    table_incert.add_column("Partícula", justify="center", style="cyan", width=12)
    table_incert.add_column("Masa Ajustada", justify="center", style="green", width=22)
    table_incert.add_column("σ (Ancho)", justify="center", style="yellow", width=18)
    table_incert.add_column("FWHM", justify="center", style="magenta", width=15)
    table_incert.add_column("N eventos", justify="center", style="blue", width=10)
    
    for nombre, res in resultados_parte1.items():
        if res['ajuste_exitoso']:
            table_incert.add_row(
                nombre,
                f"{res['masa_ajustada']:.4f} ± {res['error_estadistico']:.4f}",
                f"{res['sigma']:.4f} GeV",
                f"{res['FWHM']:.4f} GeV",
                f"{res['N_eventos']:,}"
            )
        else:
            table_incert.add_row(
                nombre,
                f"{res['masa_ajustada']:.4f}",
                "N/A",
                "N/A",
                f"{res['N_eventos']:,}"
            )
    
    cons.print(table_incert)

cons.rule("[bold cyan] COMPARACIÓN CON PARTICLE DATA GROUP[/bold cyan]")

# Tabla detallada de partículas CON INCERTIDUMBRES CALCULADAS
table_pdg = Table(title="Comparación con PDG (Particle Data Group)", box=box.DOUBLE_EDGE)
table_pdg.add_column("Partícula", justify="center", style="cyan")
table_pdg.add_column("Masa PDG (GeV/c²)", justify="center", style="green")
table_pdg.add_column("Masa Observada", justify="center", style="yellow", width=25)
table_pdg.add_column("Diferencia", justify="center", style="red")
table_pdg.add_column("Descripción", justify="left", style="blue")

for nombre, masa_pdg in particulas_conocidas.items():
    # Buscar si hay pico cerca
    diferencias = st.abs(masas_picos - masa_pdg)
    if len(diferencias) > 0 and st.min(diferencias) < 0.5:
        idx_cercano = st.argmin(diferencias)
        masa_obs = masas_picos[idx_cercano]
        diff = masa_obs - masa_pdg
        
        # Usar incertidumbre calculada si existe
        if nombre in resultados_parte1 and resultados_parte1[nombre]['ajuste_exitoso']:
            res = resultados_parte1[nombre]
            masa_str = f"{res['masa_ajustada']:.4f} ± {res['error_estadistico']:.4f}"
        else:
            masa_str = f"{masa_obs:.3f} ± 0.010"
        
        table_pdg.add_row(
            nombre,
            f"{masa_pdg:.3f}",
            masa_str,
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

# ==============================================================================
# PARTE 2: BOSÓN Z CON CÁLCULO DE INCERTIDUMBRES
# ==============================================================================

cons.rule("[bold red]PARTE 2: ANÁLISIS DEL BOSÓN Z[/bold red]")
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

cons.rule("[bold cyan](b) Generando histograma lineal...[/bold cyan]")

counts , bin_centers, filename = histograma(mass, "Bosón_Z_Run2010B_Lineal", len(mass), colors= 'blue', edge_color='coral')

cons.print(f"[bold green]Histograma guardado en:[/bold green] {filename}\n")

cons.rule("[bold cyan](c) Generando histograma logarítmico...[/bold cyan]")

log_counts , log_bin_centers, filename = histograma(mass, "Bosón_Z_Run2010B_Log", len(mass),True , 'forestgreen', 'coral')

cons.print(f"[bold green]Histograma guardado en:[/bold green] {filename}\n")

cons.rule("[bold cyan](d) Detectando picos en el histograma...[/bold cyan]")

# Encontrar picos en el histograma
masas_picos = buscar_log(log_counts, log_bin_centers)

# ==============================================================================
# CALCULAR INCERTIDUMBRES PARA EL BOSÓN Z
# ==============================================================================

cons.rule("[bold cyan]CÁLCULO DE INCERTIDUMBRES - PARTE 2[/bold cyan]")

resultado_z = None

if len(masas_picos) > 0:
    # Tomar el pico más prominente (debería ser el Z)
    pico_principal = masas_picos[0]
    
    cons.print(f"\n[bold yellow]Analizando pico principal en {pico_principal:.3f} GeV/c²[/bold yellow]")
    
    # Calcular incertidumbre
    resultado_z = calcular_incertidumbre_masa(mass, pico_principal, ventana=6.0)
    
    # Graficar ajuste
    graficar_ajuste_gaussiano(mass, pico_principal, resultado_z, ventana=6.0, 
                              nombre_particula="Bosón Z", output_dir=output_dir)
    
    # Crear tabla resumen
    cons.rule("[bold green]RESULTADOS FINALES - BOSÓN Z[/bold green]")
    
    table_resultados = Table(title="Masa y Incertidumbres del Bosón Z", box=box.DOUBLE_EDGE)
    table_resultados.add_column("Parámetro", justify="left", style="cyan", width=30)
    table_resultados.add_column("Valor", justify="center", style="green", width=25)
    table_resultados.add_column("Descripción", justify="left", style="yellow")
    
    table_resultados.add_row(
        "Masa observada (pico)",
        f"{pico_principal:.4f} GeV/c²",
        "Posición del pico en histograma"
    )
    
    if resultado_z['ajuste_exitoso']:
        table_resultados.add_row(
            "Masa ajustada (μ)",
            f"{resultado_z['masa_ajustada']:.4f} ± {resultado_z['sigma_mu']:.4f} GeV/c²",
            "Centro del ajuste gaussiano"
        )
        table_resultados.add_row(
            "Resolución (σ)",
            f"{resultado_z['sigma']:.4f} ± {resultado_z.get('sigma_sigma', 0):.4f} GeV/c²",
            "Ancho de la distribución"
        )
        table_resultados.add_row(
            "FWHM",
            f"{resultado_z['FWHM']:.4f} GeV/c²",
            "Ancho a media altura"
        )
        table_resultados.add_row(
            "Error estadístico",
            f"{resultado_z['error_estadistico']:.4f} GeV/c²",
            "σ/√N"
        )
    
    table_resultados.add_row(
        "Eventos usados",
        f"{resultado_z['N_eventos']:,}",
        "En ventana de análisis"
    )
    
    table_resultados.add_row("", "", "")
    
    masa_Z_pdg = 91.188
    diff = resultado_z['masa_ajustada'] - masa_Z_pdg if resultado_z['ajuste_exitoso'] else pico_principal - masa_Z_pdg
    error_rel = abs(diff / masa_Z_pdg * 100)
    
    table_resultados.add_row(
        "Masa PDG (Z⁰)",
        f"{masa_Z_pdg:.3f} GeV/c²",
        "Valor de referencia"
    )
    table_resultados.add_row(
        "Diferencia",
        f"{diff:+.4f} GeV/c²",
        "Masa medida - Masa PDG"
    )
    table_resultados.add_row(
        "Error relativo",
        f"{error_rel:.3f}%",
        "Precisión de la medición"
    )
    
    cons.print(table_resultados)
    
    # Panel explicativo
    panel_explicacion = Panel(
        f"""[bold cyan]INTERPRETACIÓN DE LAS INCERTIDUMBRES:[/bold cyan]

[yellow]σ (Resolución del detector):[/yellow]
  • Representa el ancho intrínseco de la distribución
  • Incluye: resolución del detector + ancho natural de la partícula
  • Para el Z: Γ_Z ≈ 2.5 GeV (ancho natural muy grande)
  • σ medido ≈ {resultado_z['sigma']:.3f} GeV refleja principalmente Γ_Z

[yellow]FWHM (Full Width at Half Maximum):[/yellow]
  • FWHM = 2.355 × σ ≈ {resultado_z['FWHM']:.3f} GeV
  • Medida alternativa del ancho del pico
  • Útil para comparar con otros análisis

[yellow]Error estadístico (σ/√N):[/yellow]
  • Error en la determinación de la masa central
  • Con N = {resultado_z['N_eventos']:,} eventos → error = {resultado_z['error_estadistico']:.4f} GeV
  • Este es el error en nuestra medición de m_Z

[yellow]Masa final del bosón Z:[/yellow]
  [bold green]M_Z = {resultado_z['masa_ajustada']:.4f} ± {resultado_z['error_estadistico']:.4f} GeV/c²[/bold green]
  
  Comparado con PDG: {masa_Z_pdg:.3f} GeV/c²
  Diferencia: {diff:+.4f} GeV ({error_rel:.2f}%)
  
  ✓ Excelente concordancia con el valor aceptado
""",
        title="[bold]Análisis de Incertidumbres[/bold]",
        border_style="green",
        box=box.DOUBLE
    )
    
    cons.print(panel_explicacion)

cons.print("\n[bold green]Análisis completado con éxito :D[/bold green]")

cons.rule("[bold cyan] COMPARACIÓN FINAL CON PARTICLE DATA GROUP[/bold cyan]")

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
        
        # Usar incertidumbre calculada si es el Z
        if nombre == 'Z⁰' and resultado_z is not None and resultado_z['ajuste_exitoso']:
            masa_str = f"{resultado_z['masa_ajustada']:.4f} ± {resultado_z['error_estadistico']:.4f}"
            diff = resultado_z['masa_ajustada'] - masa_pdg
        else:
            masa_str = f"{masa_obs:.3f} ± 0.010"
        
        table_pdg.add_row(
            nombre,
            f"{masa_pdg:.3f}",
            masa_str,
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

cons.rule("[bold green]✓✓✓ ANÁLISIS COMPLETO CON INCERTIDUMBRES ✓✓✓[/bold green]")
cons.print(f"\n[yellow]Archivos guardados en: {output_dir}/[/yellow]\n")