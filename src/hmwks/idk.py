"""
PRUEBAS DE ALEATORIEDAD Y UNIFORMIDAD
Implementación de tests estadísticos según:
Knuth, D.E. (1997). The Art of Computer Programming, Vol. 2: Seminumerical Algorithms

Autor: Implementación educativa
Fecha: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammainc, gamma
import seaborn as sns
from typing import Tuple, Dict, List
import warnings

# Configuración estética
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10


# =============================================================================
# 1. TEST DE FRECUENCIA (CHI-CUADRADO)
# =============================================================================


def test_frecuencia_chi_cuadrado(
    numeros: np.ndarray, k: int = 10, alpha: float = 0.05
) -> Dict:
    """
    Test de Chi-cuadrado para uniformidad (Knuth, Sección 3.3.1)

    Hipótesis nula (H0): Los números siguen una distribución uniforme

    Parámetros:
    -----------
    numeros : array
        Números a probar (en [0, 1) o se normalizan)
    k : int
        Número de intervalos (bins) a dividir [0, 1)
    alpha : float
        Nivel de significancia (típicamente 0.05)

    Retorna:
    --------
    resultado : dict
        - chi2_stat: Estadístico Chi-cuadrado calculado
        - chi2_critico: Valor crítico de Chi-cuadrado
        - p_value: Valor p del test
        - pasa_test: True si pasa el test
        - observados: Frecuencias observadas
        - esperados: Frecuencias esperadas

    Teoría:
    -------
    χ² = Σ[(O_i - E_i)² / E_i]

    donde:
    - O_i = frecuencia observada en bin i
    - E_i = frecuencia esperada = n/k (uniforme)
    - Grados de libertad = k - 1
    """
    # Normalizar a [0, 1) si es necesario
    if numeros.max() > 1.0 or numeros.min() < 0.0:
        numeros = (numeros - numeros.min()) / (numeros.max() - numeros.min())

    n = len(numeros)

    # Frecuencias observadas
    observados, _ = np.histogram(numeros, bins=k, range=(0, 1))

    # Frecuencias esperadas (uniforme)
    esperados = np.full(k, n / k)

    # Estadístico Chi-cuadrado
    chi2_stat = np.sum((observados - esperados) ** 2 / esperados)

    # Grados de libertad
    gl = k - 1

    # Valor crítico y p-value
    chi2_critico = stats.chi2.ppf(1 - alpha, gl)
    p_value = 1 - stats.chi2.cdf(chi2_stat, gl)

    # Decisión
    pasa_test = chi2_stat < chi2_critico

    return {
        "chi2_stat": chi2_stat,
        "chi2_critico": chi2_critico,
        "p_value": p_value,
        "pasa_test": pasa_test,
        "grados_libertad": gl,
        "observados": observados,
        "esperados": esperados,
        "alpha": alpha,
    }


# =============================================================================
# 2. TEST DE KOLMOGOROV-SMIRNOV
# =============================================================================


def test_kolmogorov_smirnov(numeros: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Test de Kolmogorov-Smirnov para uniformidad (Knuth, Sección 3.3.1)

    Mide la máxima discrepancia entre la función de distribución empírica
    y la teórica (uniforme).

    Parámetros:
    -----------
    numeros : array
        Números a probar (en [0, 1) o se normalizan)
    alpha : float
        Nivel de significancia

    Retorna:
    --------
    resultado : dict
        - D_stat: Estadístico D de KS (máxima discrepancia)
        - D_critico: Valor crítico
        - p_value: Valor p del test
        - pasa_test: True si pasa el test

    Teoría:
    -------
    D = max|F_n(x) - F(x)|

    donde:
    - F_n(x) = función de distribución empírica
    - F(x) = función de distribución teórica (uniforme = x)
    """
    # Normalizar a [0, 1)
    if numeros.max() > 1.0 or numeros.min() < 0.0:
        numeros = (numeros - numeros.min()) / (numeros.max() - numeros.min())

    # Aplicar test de KS contra distribución uniforme
    D_stat, p_value = stats.kstest(numeros, "uniform")

    # Valor crítico aproximado: D_α = c(α) / √n
    n = len(numeros)
    # Para α=0.05, c(α) ≈ 1.36; para α=0.01, c(α) ≈ 1.63
    c_alpha = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}.get(alpha, 1.36)
    D_critico = c_alpha / np.sqrt(n)

    pasa_test = D_stat < D_critico

    return {
        "D_stat": D_stat,
        "D_critico": D_critico,
        "p_value": p_value,
        "pasa_test": pasa_test,
        "n": n,
        "alpha": alpha,
    }


# =============================================================================
# 3. TEST DE CORRELACIÓN SERIAL
# =============================================================================


def test_correlacion_serial(
    numeros: np.ndarray, lag: int = 1, alpha: float = 0.05
) -> Dict:
    """
    Test de correlación serial (Knuth, Sección 3.3.3)

    Verifica si existe correlación entre números consecutivos.

    Parámetros:
    -----------
    numeros : array
        Secuencia de números
    lag : int
        Desfase para calcular correlación (típicamente 1)
    alpha : float
        Nivel de significancia

    Retorna:
    --------
    resultado : dict
        - correlacion: Coeficiente de correlación
        - z_stat: Estadístico Z normalizado
        - p_value: Valor p (bilateral)
        - pasa_test: True si pasa (correlación ≈ 0)

    Teoría:
    -------
    Para números aleatorios independientes:
    ρ ≈ 0, con σ_ρ ≈ 1/√n
    """
    n = len(numeros)

    # Calcular correlación con lag
    x1 = numeros[:-lag]
    x2 = numeros[lag:]

    correlacion = np.corrcoef(x1, x2)[0, 1]

    # Para muestras grandes, ρ ~ N(0, 1/n)
    # Estadístico Z = ρ * √n
    z_stat = correlacion * np.sqrt(n - lag)

    # P-value bilateral
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Valor crítico para correlación
    z_critico = stats.norm.ppf(1 - alpha / 2)
    rho_critico = z_critico / np.sqrt(n - lag)

    pasa_test = abs(correlacion) < rho_critico

    return {
        "correlacion": correlacion,
        "z_stat": z_stat,
        "p_value": p_value,
        "pasa_test": pasa_test,
        "rho_critico": rho_critico,
        "alpha": alpha,
        "lag": lag,
    }


# =============================================================================
# 4. TEST DE POKER
# =============================================================================


def test_poker(numeros: np.ndarray, d: int = 5, alpha: float = 0.05) -> Dict:
    """
    Test del Poker (Knuth, Sección 3.3.2)

    Divide números en grupos de d dígitos y clasifica según
    patrones (ej: todos diferentes, pares, tríos, etc.)

    Parámetros:
    -----------
    numeros : array
        Números en [0, 1)
    d : int
        Número de dígitos por mano (típicamente 5)
    alpha : float
        Nivel de significancia

    Retorna:
    --------
    resultado : dict
        - chi2_stat: Estadístico Chi-cuadrado
        - p_value: Valor p
        - pasa_test: True si pasa
        - categorias: Conteos por categoría

    Teoría:
    -------
    Para d=5 dígitos decimales, hay 7 categorías:
    1. Todos diferentes (ej: 12345)
    2. Un par (ej: 11234)
    3. Dos pares (ej: 11223)
    4. Tres iguales (ej: 11123)
    5. Full house (ej: 11122)
    6. Cuatro iguales (ej: 11112)
    7. Todos iguales (ej: 11111)
    """
    # Convertir a dígitos
    digitos = (numeros * 10**d).astype(int) % (10**d)

    # Extraer dígitos individuales
    def contar_patron(numero):
        """Clasifica patrón de dígitos"""
        digitos_str = str(numero).zfill(d)
        conteos = np.bincount([int(dig) for dig in digitos_str])
        conteos_unicos = sorted(conteos[conteos > 0], reverse=True)

        # Clasificar según patrón
        if conteos_unicos == [d]:  # Todos iguales
            return 6
        elif conteos_unicos == [d - 1, 1]:  # 4 iguales
            return 5
        elif conteos_unicos == [3, 2]:  # Full house
            return 4
        elif conteos_unicos == [3, 1, 1]:  # 3 iguales
            return 3
        elif conteos_unicos == [2, 2, 1]:  # 2 pares
            return 2
        elif conteos_unicos == [2, 1, 1, 1]:  # 1 par
            return 1
        else:  # Todos diferentes
            return 0

    # Clasificar todas las manos
    categorias = np.array([contar_patron(num) for num in digitos])
    observados = np.bincount(categorias, minlength=7)

    # Probabilidades teóricas para d=5 (fórmulas de Knuth)
    n_manos = len(digitos)
    if d == 5:
        # Probabilidades exactas para 5 dígitos decimales
        p_teoricas = np.array(
            [
                0.3024,  # Todos diferentes
                0.5040,  # Un par
                0.1080,  # Dos pares
                0.0720,  # Tres iguales
                0.0090,  # Full house
                0.0045,  # Cuatro iguales
                0.0001,  # Todos iguales
            ]
        )
    else:
        # Aproximación para otros valores de d
        warnings.warn(
            f"Probabilidades exactas solo para d=5, usando aproximación para d={d}"
        )
        # Usar distribución observada normalizada
        p_teoricas = observados / observados.sum()

    esperados = n_manos * p_teoricas

    # Evitar divisiones por cero
    esperados[esperados < 1] = 1

    # Chi-cuadrado
    chi2_stat = np.sum((observados - esperados) ** 2 / esperados)
    gl = len(p_teoricas) - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, gl)

    chi2_critico = stats.chi2.ppf(1 - alpha, gl)
    pasa_test = chi2_stat < chi2_critico

    categorias_nombres = [
        "Todos diferentes",
        "Un par",
        "Dos pares",
        "Tres iguales",
        "Full house",
        "Cuatro iguales",
        "Todos iguales",
    ]

    return {
        "chi2_stat": chi2_stat,
        "chi2_critico": chi2_critico,
        "p_value": p_value,
        "pasa_test": pasa_test,
        "observados": observados,
        "esperados": esperados,
        "categorias_nombres": categorias_nombres,
        "d": d,
        "alpha": alpha,
    }


# =============================================================================
# 5. TEST DE RUNS (RACHAS)
# =============================================================================


def test_runs_arriba_abajo(numeros: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Test de Runs - Rachas arriba y abajo (Knuth, Sección 3.3.2)

    Cuenta rachas de números ascendentes o descendentes.

    Parámetros:
    -----------
    numeros : array
        Secuencia de números
    alpha : float
        Nivel de significancia

    Retorna:
    --------
    resultado : dict
        - total_runs: Número total de rachas
        - media_esperada: Media esperada bajo H0
        - varianza_esperada: Varianza esperada
        - z_stat: Estadístico Z
        - p_value: Valor p
        - pasa_test: True si pasa

    Teoría:
    -------
    Para n números aleatorios:
    - Media: μ = (2n - 1) / 3
    - Varianza: σ² = (16n - 29) / 90
    """
    n = len(numeros)

    # Detectar rachas ascendentes
    diferencias = np.diff(numeros)
    cambios = np.diff(np.sign(diferencias))

    # Contar rachas (cambios de signo + 1)
    total_runs = np.sum(cambios != 0) + 1

    # Media y varianza esperadas (fórmulas de Knuth)
    media_esperada = (2 * n - 1) / 3
    varianza_esperada = (16 * n - 29) / 90

    # Estadístico Z
    z_stat = (total_runs - media_esperada) / np.sqrt(varianza_esperada)

    # P-value bilateral
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    z_critico = stats.norm.ppf(1 - alpha / 2)
    pasa_test = abs(z_stat) < z_critico

    return {
        "total_runs": total_runs,
        "media_esperada": media_esperada,
        "varianza_esperada": varianza_esperada,
        "z_stat": z_stat,
        "p_value": p_value,
        "pasa_test": pasa_test,
        "z_critico": z_critico,
        "alpha": alpha,
    }


# =============================================================================
# 6. TEST DE GAP
# =============================================================================


def test_gap(
    numeros: np.ndarray, a: float = 0.3, b: float = 0.7, alpha: float = 0.05
) -> Dict:
    """
    Test de Gap - Espacios entre ocurrencias (Knuth, Sección 3.3.2)

    Mide la distancia entre números que caen en un intervalo [a, b).

    Parámetros:
    -----------
    numeros : array
        Números en [0, 1)
    a, b : float
        Intervalo de interés [a, b)
    alpha : float
        Nivel de significancia

    Retorna:
    --------
    resultado : dict
        - chi2_stat: Estadístico Chi-cuadrado
        - p_value: Valor p
        - pasa_test: True si pasa
        - gaps: Longitudes de gaps observados

    Teoría:
    -------
    Si p = b - a (probabilidad del intervalo):
    P(gap = r) = (1-p)^r · p
    """
    # Normalizar
    if numeros.max() > 1.0 or numeros.min() < 0.0:
        numeros = (numeros - numeros.min()) / (numeros.max() - numeros.min())

    # Identificar números en [a, b)
    en_intervalo = (numeros >= a) & (numeros < b)
    indices = np.where(en_intervalo)[0]

    # Calcular gaps
    if len(indices) < 2:
        return {
            "chi2_stat": np.inf,
            "p_value": 0.0,
            "pasa_test": False,
            "gaps": [],
            "mensaje": "Muy pocos eventos en intervalo",
        }

    gaps = np.diff(indices) - 1  # -1 porque gap=0 significa consecutivos

    # Agrupar gaps (0, 1, 2, ..., k, ≥k+1)
    k_max = 10  # Máximo gap a considerar individualmente
    observados = np.zeros(k_max + 1)
    for gap in gaps:
        if gap <= k_max:
            observados[gap] += 1
        else:
            observados[k_max] += 1  # Agrupar gaps grandes

    # Probabilidades teóricas
    p = b - a
    esperados = np.zeros(k_max + 1)
    for r in range(k_max):
        esperados[r] = len(gaps) * ((1 - p) ** r) * p
    esperados[k_max] = len(gaps) * ((1 - p) ** k_max)  # P(gap ≥ k_max)

    # Chi-cuadrado (solo categorías con esperado > 1)
    mask = esperados >= 1
    chi2_stat = np.sum((observados[mask] - esperados[mask]) ** 2 / esperados[mask])

    gl = np.sum(mask) - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, gl)

    chi2_critico = stats.chi2.ppf(1 - alpha, gl)
    pasa_test = chi2_stat < chi2_critico

    return {
        "chi2_stat": chi2_stat,
        "chi2_critico": chi2_critico,
        "p_value": p_value,
        "pasa_test": pasa_test,
        "gaps": gaps,
        "observados": observados,
        "esperados": esperados,
        "intervalo": (a, b),
        "alpha": alpha,
    }


# =============================================================================
# 7. SUITE COMPLETA DE TESTS
# =============================================================================


def suite_completa_tests(
    numeros: np.ndarray,
    nombre: str = "Generador",
    alpha: float = 0.05,
    verbose: bool = True,
) -> Dict:
    """
    Ejecuta todos los tests de aleatoriedad y genera reporte completo.

    Parámetros:
    -----------
    numeros : array
        Secuencia de números a probar
    nombre : str
        Nombre del generador (para reporte)
    alpha : float
        Nivel de significancia
    verbose : bool
        Si True, imprime reporte detallado

    Retorna:
    --------
    resultados : dict
        Diccionario con resultados de todos los tests
    """
    # Normalizar a [0, 1) si es necesario
    if numeros.max() > 1.0 or numeros.min() < 0.0:
        nums_norm = (numeros - numeros.min()) / (numeros.max() - numeros.min())
    else:
        nums_norm = numeros.copy()

    resultados = {}

    # Test 1: Chi-cuadrado
    resultados["chi_cuadrado"] = test_frecuencia_chi_cuadrado(nums_norm, alpha=alpha)

    # Test 2: Kolmogorov-Smirnov
    resultados["ks"] = test_kolmogorov_smirnov(nums_norm, alpha=alpha)

    # Test 3: Correlación serial
    resultados["correlacion"] = test_correlacion_serial(nums_norm, alpha=alpha)

    # Test 4: Poker
    resultados["poker"] = test_poker(nums_norm, alpha=alpha)

    # Test 5: Runs
    resultados["runs"] = test_runs_arriba_abajo(nums_norm, alpha=alpha)

    # Test 6: Gap
    resultados["gap"] = test_gap(nums_norm, alpha=alpha)

    # Resumen
    tests_pasados = sum(
        [
            resultados["chi_cuadrado"]["pasa_test"],
            resultados["ks"]["pasa_test"],
            resultados["correlacion"]["pasa_test"],
            resultados["poker"]["pasa_test"],
            resultados["runs"]["pasa_test"],
            resultados["gap"]["pasa_test"],
        ]
    )

    resultados["resumen"] = {
        "tests_pasados": tests_pasados,
        "total_tests": 6,
        "porcentaje": (tests_pasados / 6) * 100,
        "calificacion": (
            "EXCELENTE"
            if tests_pasados == 6
            else (
                "BUENO"
                if tests_pasados >= 5
                else "REGULAR" if tests_pasados >= 4 else "MALO"
            )
        ),
    }

    if verbose:
        imprimir_reporte(resultados, nombre, alpha)

    return resultados


def imprimir_reporte(resultados: Dict, nombre: str, alpha: float):
    """Imprime reporte detallado de los tests"""
    print("\n" + "=" * 80)
    print(f"REPORTE DE PRUEBAS DE ALEATORIEDAD - {nombre}")
    print(f"Nivel de significancia: α = {alpha}")
    print("=" * 80)

    # Test 1
    r = resultados["chi_cuadrado"]
    print(f"\n1. TEST DE CHI-CUADRADO (Frecuencia)")
    print(f"   χ² = {r['chi2_stat']:.4f}, crítico = {r['chi2_critico']:.4f}")
    print(f"   p-value = {r['p_value']:.4f}")
    print(f"   Resultado: {'✅ PASA' if r['pasa_test'] else '❌ FALLA'}")

    # Test 2
    r = resultados["ks"]
    print(f"\n2. TEST DE KOLMOGOROV-SMIRNOV")
    print(f"   D = {r['D_stat']:.6f}, crítico = {r['D_critico']:.6f}")
    print(f"   p-value = {r['p_value']:.4f}")
    print(f"   Resultado: {'✅ PASA' if r['pasa_test'] else '❌ FALLA'}")

    # Test 3
    r = resultados["correlacion"]
    print(f"\n3. TEST DE CORRELACIÓN SERIAL")
    print(f"   ρ = {r['correlacion']:.6f}, crítico = ±{r['rho_critico']:.6f}")
    print(f"   p-value = {r['p_value']:.4f}")
    print(f"   Resultado: {'✅ PASA' if r['pasa_test'] else '❌ FALLA'}")

    # Test 4
    r = resultados["poker"]
    print(f"\n4. TEST DEL POKER (d={r['d']})")
    print(f"   χ² = {r['chi2_stat']:.4f}, crítico = {r['chi2_critico']:.4f}")
    print(f"   p-value = {r['p_value']:.4f}")
    print(f"   Resultado: {'✅ PASA' if r['pasa_test'] else '❌ FALLA'}")

    # Test 5
    r = resultados["runs"]
    print(f"\n5. TEST DE RUNS (Rachas)")
    print(f"   Runs = {r['total_runs']}, esperado = {r['media_esperada']:.2f}")
    print(f"   Z = {r['z_stat']:.4f}, p-value = {r['p_value']:.4f}")
    print(f"   Resultado: {'✅ PASA' if r['pasa_test'] else '❌ FALLA'}")

    # Test 6
    r = resultados["gap"]
    print(f"\n6. TEST DE GAP (Espacios)")
    print(f"   χ² = {r['chi2_stat']:.4f}, crítico = {r['chi2_critico']:.4f}")
    print(f"   p-value = {r['p_value']:.4f}")
    print(f"   Resultado: {'✅ PASA' if r['pasa_test'] else '❌ FALLA'}")

    # Resumen
    r = resultados["resumen"]
    print(f"\n" + "=" * 80)
    print(
        f"RESUMEN: {r['tests_pasados']}/{r['total_tests']} tests pasados ({r['porcentaje']:.1f}%)"
    )
    print(f"CALIFICACIÓN: {r['calificacion']}")
    print("=" * 80 + "\n")


# =============================================================================
# 8. VISUALIZACIÓN
# =============================================================================


def visualizar_tests(numeros: np.ndarray, nombre: str = "Generador"):
    """
    Genera visualizaciones diagnósticas para los tests.
    """
    # Normalizar
    if numeros.max() > 1.0 or numeros.min() < 0.0:
        nums_norm = (numeros - numeros.min()) / (numeros.max() - numeros.min())
    else:
        nums_norm = numeros.copy()

    fig = plt.figure(figsize=(16, 12))

    # 1. Histograma de frecuencias
    ax1 = plt.subplot(3, 3, 1)
    plt.hist(nums_norm, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    plt.axhline(
        len(nums_norm) / 50, color="red", linestyle="--", label="Uniforme esperado"
    )
    plt.title("1. Distribución de Frecuencias")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. QQ-Plot
    ax2 = plt.subplot(3, 3, 2)
    stats.probplot(nums_norm, dist="uniform", plot=plt)
    plt.title("2. QQ-Plot (vs Uniforme)")
    plt.grid(True, alpha=0.3)

    # 3. Función de distribución empírica
    ax3 = plt.subplot(3, 3, 3)
    nums_sorted = np.sort(nums_norm)
    y_empirica = np.arange(1, len(nums_sorted) + 1) / len(nums_sorted)
    plt.plot(nums_sorted, y_empirica, "b-", linewidth=2, label="Empírica")
    plt.plot([0, 1], [0, 1], "r--", linewidth=2, label="Teórica (uniforme)")
    plt.title("3. Función de Distribución (KS Test)")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Test de correlación serial
    ax4 = plt.subplot(3, 3, 4)
    plt.scatter(nums_norm[:-1], nums_norm[1:], s=1, alpha=0.3)
    plt.title("4. Correlación Serial (lag=1)")
    plt.xlabel(r"$x_i$")
    plt.ylabel(r"$x_{i+1}$")
    plt.grid(True, alpha=0.3)

    # 5. Autocorrelación
    ax5 = plt.subplot(3, 3, 5)
    lags = range(1, min(50, len(nums_norm) // 10))
    autocorr = [np.corrcoef(nums_norm[:-lag], nums_norm[lag:])[0, 1] for lag in lags]
    plt.stem(lags, autocorr, basefmt=" ")
    plt.axhline(
        1.96 / np.sqrt(len(nums_norm)), color="r", linestyle="--", label="95% límite"
    )
    plt.axhline(-1.96 / np.sqrt(len(nums_norm)), color="r", linestyle="--")
    plt.title("5. Autocorrelación")
    plt.xlabel("Lag")
    plt.ylabel("Correlación")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. Distribución 2D (pares)
    ax6 = plt.subplot(3, 3, 6)
    plt.hexbin(nums_norm[::2], nums_norm[1::2], gridsize=30, cmap="Blues")
    plt.title("6. Distribución 2D (Test Visual)")
    plt.xlabel(r"$x_{2i}$")
    plt.ylabel(r"$x_{2i+1}$")
    plt.colorbar(label="Frecuencia")

    # 7. Serie temporal
    ax7 = plt.subplot(3, 3, 7)
    plt.plot(nums_norm[:1000], "o-", markersize=2, linewidth=0.5, alpha=0.6)
    plt.title("7. Serie Temporal (primeros 1000)")
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)

    # 8. Test de Runs (visual)
    ax8 = plt.subplot(3, 3, 8)
    diffs = np.diff(nums_norm[:500])
    colores = ["green" if d > 0 else "red" for d in diffs]
    plt.bar(range(len(diffs)), np.abs(diffs), color=colores, alpha=0.6)
    plt.title("8. Rachas Ascendentes/Descendentes")
    plt.xlabel("Índice")
    plt.ylabel("|Diferencia|")
    plt.grid(True, alpha=0.3)

    # 9. Test del Poker (distribución de patrones)
    ax9 = plt.subplot(3, 3, 9)
    r_poker = test_poker(nums_norm, d=5)
    x_pos = np.arange(len(r_poker["categorias_nombres"]))
    plt.bar(
        x_pos - 0.2,
        r_poker["observados"],
        0.4,
        label="Observado",
        alpha=0.7,
        color="steelblue",
    )
    plt.bar(
        x_pos + 0.2,
        r_poker["esperados"],
        0.4,
        label="Esperado",
        alpha=0.7,
        color="orange",
    )
    plt.xticks(x_pos, ["TD", "1P", "2P", "3I", "FH", "4I", "TI"], rotation=45)
    plt.title("9. Test del Poker (d=5)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Pruebas de Aleatoriedad - {nombre}", fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Ejemplo de uso
    print("Módulo de pruebas de aleatoriedad cargado correctamente.")
    print("Usa suite_completa_tests(numeros) para ejecutar todos los tests.")
