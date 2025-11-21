import numpy as lol
import pandas as pf
import matplotlib.pyplot as gf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import os

output_dir = "exam_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
cns = Console()
cns.rule("[bold red] Ejercicio 1: Decaimiento Radiactivo")
cns.print("[bold cyan] Vamos a generar las gráficas del decaimiento con los datos que se inventó Claude.")
cns.print("\n [bold cyan] Los datos son los siguientes: \n")

# Datos dados
ttime = lol.array([0,2,4,6,8,10,12,14,16,18])
cuentas = lol.array([1000, 820 , 670 , 550,  450 , 370 , 300 , 245 , 200 , 165])
sigma_n = lol.array([30  , 25  , 22 ,  20 ,  18  , 16   ,15  , 14  , 13   ,12])
ln_cuentas = lol.log(cuentas)
sigma_ln_n = lol.array(sigma_n)/lol.array(cuentas)

tabular = Table(title="Datos del experimento", box= box.DOUBLE_EDGE)
tabular.add_column("Tiempo (horas)", justify="center", style="magenta", no_wrap=True)
tabular.add_column("N (Cuentas)", justify="center", style="magenta", no_wrap=True)
tabular.add_column("sigma_N", justify="center", style="magenta", no_wrap=True)

for i in range(len(ttime)):
    tabular.add_row(str(ttime[i]), str(cuentas[i]), str(sigma_n[i]))
cns.print(tabular)

cns.print("[bold yellow] Vamos a generar las gráficas solicitadas ... :D")
gf.figure(figsize=(10,8))
gf.errorbar(ttime, cuentas, yerr=sigma_n, fmt='o', label='Datos experimentales con barras de error', color = 'r', ecolor='black', capsize=5)
gf.title('Decaimiento de particulas radiactivas', fontsize=16)
gf.xlabel('Tiempo (horas)', fontsize=14)
gf.ylabel('N (Cuentas)', fontsize=14)
gf.grid(True)
gf.legend()
filename = f"{output_dir}/N_vs_T.png"
gf.savefig(filename, dpi=300, bbox_inches="tight")
cns.print(Panel(f"Gráfica guardada como [bold]{filename}[/bold]", title="Éxto", subtitle="Generado por Rich", box=box.HEAVY))
gf.show()

gf.figure(figsize=(10,8))
gf.errorbar(ttime, ln_cuentas, yerr=sigma_ln_n, fmt='o', label='Datos experimentales con barras de error', color = 'r', ecolor='black', capsize=5)
gf.title('Decaimiento de particulas radiactivas con el cambio de parametros', fontsize=16)
gf.xlabel('Tiempo (horas)', fontsize=14)
gf.ylabel('ln(N) (Cuentas)', fontsize=14)
gf.grid(True)
gf.legend()
filename = f"{output_dir}/ln_N_vs_T.png"
gf.savefig(filename, dpi=300, bbox_inches="tight")
cns.print(Panel(f"Gráfica guardada como [bold]{filename}[/bold]", title="Éxto", subtitle="Generado por Rich", box=box.HEAVY))
gf.show()

cns.print(f"[bold yellow] :D Gráficas generadas con éxto y guardadas en {output_dir}/")
cns.rule("[bold red] Ajustes por mínimos cuadrados")

cns.print("\n[bold cyan] Vamos a ajustar el modelo N(t) = N₀·e^(-λt) por mínimos cuadrados")

# ajuste por realizar

def modelo_ajuste(t, N0, lam):
    """
    Modelamos el decaimiento exponencial para las particulas radiactivas:
    N(t) = N₀·e^(-λt)

    Parámetros:
    -----------
    t : float o array
        Tiempo
    N0 : float
        Población inicial (en t=0)
    Lam : float
        Constante de decaimiento λ
    
    Retorna:
    --------
    N : float o array
        Población en el tiempo t
    """
    return N0 * lol.exp(- lam*t)
def min_cuad_lineal(x , y):
    """
    Realizamos el ajuste según lo visto en clase :D
    Ajustamos una recta de la forma 
    y = mx + b
    Parámetros:
    -----------
    lon : int
          cantidad de datos a ajustar
    x : array
        los datos del eje x
    y : array
        los datos del eje y

    Retorna:
    --------
    m : float 
        la pendiente de nuestro ajuste
    b : float
        la ordenada al origen de nuestro ajuste
    """
    # x_aux = x.to_numpy()
    # y_aux = y.to_numpy()
    lon = len(x)

    b =(sum(x*x)*sum(y)-sum(x)*sum(x*y))/(lon*sum(x*x)-(sum(x))**2)
    m =(lon*sum(x*y)-sum(x)*sum(y)) / (lon*sum(x*x)-(sum(x))**2)

    return m,b
def calculo_incertidumbres(sig, x, y):
    """
    
    """
    def S(sig):
        return lol.sum(1/(sig**2))

    def S_x(sig,x):
        return lol.sum(x/(sig**2))

    def S_y(sig,y):
        return lol.sum(y/(sig**2))

    def S_xx(sig,x):
        return lol.sum((x**2)/(sig**2))

    def S_xy(sig,x,y):
        return lol.sum((x*y)/(sig**2))

    def delta(s,s_xx,s_x):
        return s*s_xx-(s_x**2)

    def sigma_a1(s_xx,delt):
        """
        Sigma para ordenada al origen (No)
        """
        return s_xx/delt
    def sigma_a2(s,delt):
        """"
        Sigma para pendiente (lambda)
        """
        return s/delt
    s = S(sig)
    s_x = S_x(sig,x)
    s_xx = S_xx(sig,x)
    delt = delta(s, s_xx, s_x)

    sigma_b = s_xx / delt
    sigma_m = s / delt
    return sigma_m, sigma_b

def calculo_vida_media_pm(lambsa, sigma):
    """
    
    """
    t_half= lol.log(2)/lambsa
    sigma_t_half = (lol.log(2)/(lambsa**2))*sigma
    return t_half, sigma_t_half

def calculo_chis_cuadrado(y,m,x,b,sig_y):
    """
    
    """
    chi_cuadrada = lol.sum(((y-(m*x+b))**2)/(sig_y**2))
    chi_red = chi_cuadrada/(len(x)-2)
    return chi_cuadrada, chi_red

cns.print("\n[bold cyan] Del modelo: N(t) = N₀·e^(-λt) tenemos que:\n"
          "ln(N)=ln(N₀)-λt \t Por lo que y = ln(N), x=t, m = - λ, b= ln(N₀)\n")
cns.print("\n[bold yellow] Realizando el ajuste ... D:\n")

m_linear, b_linear = min_cuad_lineal(ttime, ln_cuentas)

N_0 = lol.exp(b_linear)
lam = - m_linear
sigma_m, sigma_b = calculo_incertidumbres(sigma_ln_n, ttime, ln_cuentas)
sigma_lambda = sigma_m
sigma_N_0 = N_0*sigma_b

# Vida media
t_media, sigma_t_media = calculo_vida_media_pm(lam, sigma_lambda)

# Chi cuadrado
chi2 , chi2_red = calculo_chis_cuadrado(ln_cuentas,m_linear,ttime, b_linear,sigma_ln_n)


cns.rule("[bold green] Resultados del Ajuste")

cns.print(f"[bold yellow]N₀ = {N_0:.2f} ± {sigma_N_0:.2f}")
cns.print(f"[bold yellow]λ = {lam:.5f} ± {sigma_lambda:.5f}")
cns.rule("[bold green] Resultados del Ajuste")

cns.print(f"[bold yellow]N₀ = {N_0:.2f} ± {sigma_N_0:.2f}")
cns.print(f"[bold yellow]λ = {lam:.5f} ± {sigma_lambda:.5f}")
cns.print(f"[bold yellow]t_1/2 = {t_media:.3f} ± {sigma_t_media:.3f} horas")
cns.print(f"[bold yellow]χ² = {chi2:.3f}")
cns.print(f"[bold yellow]χ² reducido = {chi2_red:.3f}")

# curva ajustada
t_suave = lol.linspace(0, 18, 1000)
ajuste = modelo_ajuste(t_suave, N_0, lam)

gf.figure(figsize=(10,8))
gf.errorbar(ttime, cuentas, yerr=sigma_n, fmt='o', color='r', capsize=5, label='Datos')
gf.plot(t_suave, ajuste, label=f"Ajuste exponencial N(t)={N_0:.2f}exp(-{lam:.2f}*t)", linewidth=2)
gf.xlabel("Tiempo (h)")
gf.title("Curva ajustada de nuestro decaimiento")
gf.ylabel("N")
gf.legend()
gf.grid()
filename = f"{output_dir}/Ajuste_exponencial.png"
gf.savefig(filename, dpi=300, bbox_inches="tight")
cns.print(Panel(f"Curva ajustada guardada como [bold]{filename}[/bold]", 
                title="Éxito", box=box.HEAVY))
gf.show()