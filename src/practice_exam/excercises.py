"""
═══════════════════════════════════════════════════════════════════════════
    10 EJERCICIOS PARA PREPARACIÓN DE EXAMEN DE FÍSICA NUMÉRICA
═══════════════════════════════════════════════════════════════════════════

Basado en los temas que has trabajado (ajustes por mínimos cuadrados, 
análisis de datos, ecuaciones diferenciales, etc.)

Autor: Claude
Fecha: Noviembre 2024
═══════════════════════════════════════════════════════════════════════════
"""

# ==============================================================================
# EJERCICIO 1: AJUSTE LINEAL Y χ²
# ==============================================================================
"""
EJERCICIO 1: Decaimiento Radioactivo
=====================================

Se midió la actividad de una muestra radioactiva en función del tiempo:

t (horas):  0    2    4    6    8    10   12   14   16   18
N (cuentas): 1000 820  670  550  450  370  300  245  200  165
σ_N:        30   25   22   20   18   16   15   14   13   12

a) Graficar N vs t y ln(N) vs t
b) Ajustar el modelo N(t) = N₀·e^(-λt) por mínimos cuadrados
c) Determinar N₀, λ y sus incertidumbres
d) Calcular la vida media t_1/2 = ln(2)/λ
e) Calcular χ² y evaluar la bondad del ajuste
f) Graficar datos y curva ajustada

Conceptos clave:
• Linearización de exponenciales
• Propagación de errores
• Chi-cuadrado reducido
• Transformación de variables
"""

# ==============================================================================
# EJERCICIO 2: INTERPOLACIÓN
# ==============================================================================
"""
EJERCICIO 2: Interpolación de Datos Experimentales
===================================================

Tienes mediciones de temperatura vs tiempo durante un experimento:

t (s):  0    5    10   15   20   25   30
T (°C): 20   35   48   59   68   75   80

a) Interpolar usando polinomios de Lagrange para t = 7.5 s
b) Interpolar usando splines cúbicos para t = 7.5 s
c) Comparar ambos resultados
d) Estimar la velocidad de calentamiento dT/dt en t = 10 s
e) Graficar datos originales y curvas interpoladas
f) ¿Cuál método es más apropiado y por qué?

Conceptos clave:
• Interpolación de Lagrange
• Splines cúbicos
• Derivación numérica
• Comparación de métodos
"""

# ==============================================================================
# EJERCICIO 3: INTEGRACIÓN NUMÉRICA
# ==============================================================================
"""
EJERCICIO 3: Trabajo Realizado por una Fuerza Variable
=======================================================

Una fuerza F(x) actúa sobre un objeto. Datos experimentales:

x (m):  0.0  0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0
F (N):  10   12   15   19   24   30   37   45   54

El trabajo realizado es W = ∫F(x)dx de 0 a 4 m.

a) Calcular W usando la regla del trapecio
b) Calcular W usando la regla de Simpson
c) Calcular W usando integración de Monte Carlo (1000 puntos)
d) Comparar los tres resultados
e) Estimar el error en cada método
f) Graficar F(x) y el área bajo la curva

Conceptos clave:
• Regla del trapecio
• Regla de Simpson
• Método de Monte Carlo
• Análisis de errores numéricos
"""

# ==============================================================================
# EJERCICIO 4: ECUACIONES DIFERENCIALES - CAÍDA LIBRE CON FRICCIÓN
# ==============================================================================
"""
EJERCICIO 4: Caída Libre con Resistencia del Aire
==================================================

Un objeto de masa m = 2 kg cae desde el reposo con resistencia del aire
proporcional a v²:

dv/dt = g - (b/m)·v²

donde g = 9.8 m/s², b = 0.1 kg/m

Condiciones iniciales: v(0) = 0, y(0) = 100 m

a) Resolver usando el método de Euler (Δt = 0.01 s)
b) Resolver usando Runge-Kutta 4 (Δt = 0.1 s)
c) Comparar ambas soluciones
d) Calcular la velocidad terminal v_terminal = √(mg/b)
e) Determinar el tiempo de caída hasta y = 0
f) Graficar v(t) y y(t)
g) Comparar con caída libre sin fricción

Conceptos clave:
• Método de Euler
• Runge-Kutta de orden 4
• Velocidad terminal
• Comparación de métodos numéricos
"""

# ==============================================================================
# EJERCICIO 5: AJUSTE NO LINEAL - OSCILADOR AMORTIGUADO
# ==============================================================================
"""
EJERCICIO 5: Oscilador Armónico Amortiguado
============================================

Un péndulo amortiguado tiene posición angular θ(t) medida:

t (s):  0.0  0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0  4.5  5.0
θ (°):  30   20   10   2   -4   -7   -8   -7   -5   -3   -1
σ_θ (°): 1    1    1   1    1    1    1    1    1    1    1

Modelo: θ(t) = A·e^(-γt)·cos(ωt + φ)

a) Estimar valores iniciales de A, γ, ω, φ
b) Ajustar el modelo usando curve_fit
c) Determinar los parámetros y sus incertidumbres
d) Calcular el coeficiente de amortiguamiento γ
e) Calcular la frecuencia natural ω₀ = √(ω² + γ²)
f) Calcular χ² y evaluar el ajuste
g) Graficar datos y modelo ajustado

Conceptos clave:
• Ajuste no lineal
• Oscilaciones amortiguadas
• Estimación de parámetros iniciales
• Propagación de errores en funciones compuestas
"""

# ==============================================================================
# EJERCICIO 6: ANÁLISIS DE FOURIER
# ==============================================================================
"""
EJERCICIO 6: Análisis de Frecuencias en Señal Periódica
========================================================

Una señal f(t) fue muestreada a 100 Hz durante 2 segundos:

f(t) = 2·sin(2π·5·t) + 0.5·sin(2π·15·t) + ruido

Los datos están en el archivo "señal.txt"

a) Graficar la señal en el dominio del tiempo
b) Calcular la Transformada de Fourier usando FFT
c) Graficar el espectro de potencias vs frecuencia
d) Identificar las frecuencias dominantes
e) Filtrar el ruido usando un filtro pasa-bajos
f) Reconstruir la señal filtrada
g) Calcular SNR (Signal-to-Noise Ratio)

Conceptos clave:
• Transformada rápida de Fourier (FFT)
• Espectro de potencias
• Filtrado de señales
• Análisis espectral
"""

# ==============================================================================
# EJERCICIO 7: RAÍCES DE ECUACIONES
# ==============================================================================
"""
EJERCICIO 7: Ecuación de Van der Waals
=======================================

Para un gas real, la presión P está dada por la ecuación de Van der Waals:

(P + a/V²)(V - b) = RT

donde para CO₂: a = 3.658 atm·L²/mol², b = 0.04267 L/mol

Dadas: T = 300 K, P = 10 atm, R = 0.08206 atm·L/(mol·K)

a) Escribir la ecuación en la forma f(V) = 0
b) Graficar f(V) para identificar las raíces
c) Encontrar V usando el método de bisección
d) Encontrar V usando el método de Newton-Raphson
e) Encontrar V usando el método de la secante
f) Comparar número de iteraciones y precisión
g) Comparar con el gas ideal: V = RT/P

Conceptos clave:
• Método de bisección
• Método de Newton-Raphson
• Método de la secante
• Convergencia de métodos
"""

# ==============================================================================
# EJERCICIO 8: AJUSTE DE DISTRIBUCIÓN ESTADÍSTICA
# ==============================================================================
"""
EJERCICIO 8: Distribución de Velocidades de Maxwell-Boltzmann
==============================================================

Se midieron las velocidades de N partículas de gas:

Rangos (m/s): [0-50] [50-100] [100-150] [150-200] [200-250] [250-300]
Frecuencias:    15      45       80        65        30        10

La distribución de Maxwell-Boltzmann es:

f(v) = 4π·n·(m/2πkT)^(3/2)·v²·exp(-mv²/2kT)

donde m = 6.63×10⁻²⁷ kg (helio)

a) Crear histograma normalizado de velocidades
b) Ajustar la distribución de Maxwell-Boltzmann
c) Determinar la temperatura T del gas
d) Calcular la velocidad más probable v_p = √(2kT/m)
e) Calcular la velocidad promedio <v> = √(8kT/πm)
f) Calcular la velocidad RMS v_rms = √(3kT/m)
g) Graficar histograma y distribución teórica
h) Calcular χ² para evaluar el ajuste

Conceptos clave:
• Distribuciones de probabilidad
• Histogramas
• Ajuste de distribuciones
• Física estadística
"""

# ==============================================================================
# EJERCICIO 9: SISTEMA DE ECUACIONES DIFERENCIALES
# ==============================================================================
"""
EJERCICIO 9: Modelo Presa-Depredador (Lotka-Volterra)
======================================================

Población de conejos (presa) y zorros (depredador):

dx/dt = αx - βxy    (conejos)
dy/dt = δxy - γy    (zorros)

Parámetros: α = 0.1, β = 0.01, γ = 0.1, δ = 0.002
Condiciones iniciales: x(0) = 100, y(0) = 10

a) Resolver el sistema usando Runge-Kutta 4 para t ∈ [0, 200]
b) Graficar x(t) y y(t) en la misma gráfica
c) Crear diagrama de fase (x vs y)
d) Identificar el comportamiento cíclico
e) Calcular el periodo de oscilación
f) Analizar el punto de equilibrio
g) Variar condiciones iniciales y observar cambios

Conceptos clave:
• Sistemas de EDOs acopladas
• Diagramas de fase
• Comportamiento dinámico
• Puntos de equilibrio
"""

# ==============================================================================
# EJERCICIO 10: PROBLEMA COMPLETO - PÉNDULO NO LINEAL
# ==============================================================================
"""
EJERCICIO 10: Péndulo Simple No Lineal (EJERCICIO INTEGRADOR)
==============================================================

Un péndulo de longitud L = 1 m oscila con amplitud grande:

d²θ/dt² + (g/L)·sin(θ) = 0

Condiciones: θ(0) = π/3, dθ/dt(0) = 0

PARTE A: SOLUCIÓN NUMÉRICA
a) Convertir a sistema de dos EDOs de primer orden
b) Resolver usando RK4 con Δt = 0.01 s para t ∈ [0, 10]
c) Graficar θ(t) y ω(t) = dθ/dt
d) Crear diagrama de fase (θ vs ω)
e) Verificar conservación de energía: E = ½mL²ω² + mgL(1-cos(θ))

PARTE B: COMPARACIÓN CON APROXIMACIÓN LINEAL
f) Resolver aproximación lineal: d²θ/dt² + (g/L)·θ = 0
g) Comparar periodos: no lineal vs lineal
h) Calcular T_lineal = 2π√(L/g) y comparar con T_numérico

PARTE C: DEPENDENCIA CON AMPLITUD
i) Repetir para θ₀ = π/6, π/4, π/3, π/2
j) Graficar T vs θ₀
k) Ajustar T(θ₀) ≈ 2π√(L/g)·[1 + (θ₀²/16) + ...]

PARTE D: CAOS
l) Agregar amortiguamiento y forzamiento periódico
m) Explorar comportamiento caótico para ciertos parámetros

Conceptos clave:
• EDOs no lineales
• Conservación de energía
• Diagramas de fase
• Análisis de periodo
• Caos determinista
"""

# ==============================================================================
# RESUMEN DE CONCEPTOS IMPORTANTES
# ==============================================================================
"""
═══════════════════════════════════════════════════════════════════════════
RESUMEN DE TÉCNICAS Y CONCEPTOS CLAVE
═══════════════════════════════════════════════════════════════════════════

1. AJUSTES POR MÍNIMOS CUADRADOS
   ✓ Lineal: y = mx + b
   ✓ No lineal: curve_fit de scipy
   ✓ Chi-cuadrado: χ² = Σ[(y_i - y_fit)²/σ_i²]
   ✓ χ²_reducido = χ²/(n - p)
   ✓ Propagación de errores

2. INTERPOLACIÓN
   ✓ Lagrange
   ✓ Splines cúbicos
   ✓ Ventajas y desventajas

3. INTEGRACIÓN NUMÉRICA
   ✓ Trapecio: ∫f(x)dx ≈ h·[f(a)/2 + Σf(x_i) + f(b)/2]
   ✓ Simpson: ∫f(x)dx ≈ (h/3)·[f(a) + 4Σf_impar + 2Σf_par + f(b)]
   ✓ Monte Carlo

4. DERIVACIÓN NUMÉRICA
   ✓ Diferencias finitas adelantadas
   ✓ Diferencias finitas centradas
   ✓ Diferencias finitas atrasadas

5. RAÍCES DE ECUACIONES
   ✓ Bisección (robusto, lento)
   ✓ Newton-Raphson (rápido, requiere derivada)
   ✓ Secante (compromiso)

6. ECUACIONES DIFERENCIALES ORDINARIAS
   ✓ Euler: y_{n+1} = y_n + h·f(x_n, y_n)
   ✓ RK2 (punto medio)
   ✓ RK4 (método estándar)
   ✓ Sistemas de EDOs

7. ANÁLISIS DE FOURIER
   ✓ FFT (Fast Fourier Transform)
   ✓ Espectro de potencias
   ✓ Filtrado de señales

8. ESTADÍSTICA Y DISTRIBUCIONES
   ✓ Histogramas
   ✓ Ajuste de distribuciones
   ✓ Pruebas de bondad de ajuste

9. ANÁLISIS DE ERRORES
   ✓ Error absoluto vs relativo
   ✓ Propagación de incertidumbres
   ✓ Errores numéricos (truncamiento, redondeo)

10. VISUALIZACIÓN
    ✓ Gráficas de datos con barras de error
    ✓ Diagramas de fase
    ✓ Mapas de contorno
    ✓ Animaciones

═══════════════════════════════════════════════════════════════════════════
CONSEJOS PARA EL EXAMEN
═══════════════════════════════════════════════════════════════════════════

📝 ANTES DEL EXAMEN:
   1. Repasa todos los métodos numéricos vistos en clase
   2. Practica la implementación en Python/MATLAB
   3. Entiende CUÁNDO usar cada método
   4. Domina el análisis de χ² y residuos
   5. Repasa propagación de errores

💻 DURANTE EL EXAMEN:
   1. Lee todo el problema antes de empezar
   2. Identifica qué método(s) necesitas
   3. Comenta tu código claramente
   4. Verifica dimensiones y unidades
   5. Grafica siempre que sea posible
   6. Interpreta los resultados físicamente

✓ CHECKLIST PARA CADA PROBLEMA:
   □ ¿Entiendo el problema físico?
   □ ¿Qué método numérico es apropiado?
   □ ¿Implementé el método correctamente?
   □ ¿Verifiqué casos límite?
   □ ¿Calculé errores/incertidumbres?
   □ ¿Hice gráficas apropiadas?
   □ ¿Interpreté los resultados?
   □ ¿Tiene sentido físico la respuesta?

═══════════════════════════════════════════════════════════════════════════
FÓRMULAS IMPORTANTES
═══════════════════════════════════════════════════════════════════════════

Chi-cuadrado reducido:
    χ²_red = (1/(n-p)) · Σ[(y_i - y_fit)²/σ_i²]
    
    0.5 ≤ χ²_red ≤ 2.0 → Buen ajuste
    χ²_red >> 1 → Errores subestimados o modelo incorrecto
    χ²_red << 1 → Errores sobrestimados

Propagación de errores:
    f(x,y) → σ_f = √[(∂f/∂x·σ_x)² + (∂f/∂y·σ_y)²]

Runge-Kutta 4:
    k1 = h·f(x_n, y_n)
    k2 = h·f(x_n + h/2, y_n + k1/2)
    k3 = h·f(x_n + h/2, y_n + k2/2)
    k4 = h·f(x_n + h, y_n + k3)
    y_{n+1} = y_n + (k1 + 2k2 + 2k3 + k4)/6

Simpson:
    ∫_a^b f(x)dx ≈ (h/3)[f(x_0) + 4f(x_1) + 2f(x_2) + ... + f(x_n)]
    donde h = (b-a)/n, n debe ser par

═══════════════════════════════════════════════════════════════════════════
"""

print(__doc__)
print("\n✓ Lista de ejercicios generada")
print("\nEstos ejercicios cubren todos los temas principales de física numérica.")
print("¡Buena suerte en tu examen! 🎯")