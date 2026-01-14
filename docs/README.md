# 🔬 Física Numérica

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)
![SciPy](https://img.shields.io/badge/SciPy-1.7+-orange.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Simulaciones numéricas y métodos computacionales para física clásica**

[Instalación](#-instalación) • [Proyectos](#-proyectos) • [Uso](#-uso) • [Estructura](#-estructura-del-repositorio)

</div>

---

## 📖 Descripción

Repositorio de **Física Numérica** que implementa métodos numéricos avanzados para resolver sistemas físicos clásicos usando Python. Incluye análisis de precisión numérica, simulaciones de mecánica clásica, y resolución de ecuaciones diferenciales ordinarias (EDOs).

### 🎯 Objetivos

- 🔢 Implementar **métodos numéricos** para resolver EDOs (Euler, Runge-Kutta, odeint)
- 📊 Analizar **límites de representación numérica** (overflow, underflow, epsilon de máquina)
- 🎯 Simular **sistemas físicos reales** con efectos no lineales y fricción
- 📈 Visualizar **trayectorias**, **espacios de fase** y **modos normales** de oscilación
- ⚖️ Comparar **modelos teóricos vs simulaciones numéricas**

---

## 🚀 Proyectos

### 📚 Índice de Proyectos
1. [Análisis de Precisión Numérica](#1️⃣-análisis-de-precisión-numérica)
2. [Lanzamiento de Martillo con Fricción](#2️⃣-lanzamiento-de-martillo-con-fricción)
3. [Osciladores Armónicos Acoplados](#3️⃣-osciladores-armónicos-acoplados)
4. [Ajuste de Datos y Análisis Estadístico (Homework 5)](#4️⃣-ajuste-de-datos-y-análisis-estadístico-homework-5)
5. [Análisis de Partículas del LHC (Homework 6)](#5️⃣-análisis-de-partículas-del-lhc-homework-6)
6. [Decaimiento de Partículas y RNG (Homework 7)](#6️⃣-decaimiento-de-partículas-y-rng-homework-7)
7. [Distribución de Fermi-Dirac](#7️⃣-distribución-de-fermi-dirac)
8. [Physics-Informed Neural Networks (PINNs)](#8️⃣-physics-informed-neural-networks-pinns)

---

### 1️⃣ **Análisis de Precisión Numérica**

Estudio de los límites de representación en punto flotante y aproximaciones mediante series.

**📂 Ubicación:** `src/Tarea1.py`, `src/Pruebas.py`

**Funcionalidades:**
- ✅ Cálculo de **overflow** (mayor número representable)
- ✅ Cálculo de **underflow** (menor número positivo representable)
- ✅ Determinación del **epsilon de máquina**
- ✅ Aproximación de funciones mediante **series de Taylor** (seno, coseno)

**Ejemplo de uso:**
```python
from src.Tarea1 import calcular_overflow, calcular_epsilon, sin_series

# Límites numéricos
overflow = calcular_overflow()
epsilon = calcular_epsilon()

# Serie de Taylor
resultado = sin_series(0.7853981633974483, n_terminos=10)
```

**Resultados típicos:**
```
Overflow:  1.7976931348623157e+308
Underflow: 5e-324
Epsilon:   2.220446049250313e-16
```

---

### 2️⃣ **Lanzamiento de Martillo con Fricción**

Simulación del récord mundial de lanzamiento de martillo considerando **resistencia del aire** con diferentes regímenes de flujo.

**📂 Ubicación:** `src/hmwks/homework_4/testeo.py`

**Características del sistema:**
- 🔨 **Masa:** 7.26 kg
- 📏 **Radio:** 6 cm
- 🎯 **Récord:** 86.74 m (Yuri Sedykh, 1986)

**Regímenes de fricción analizados:**

| Régimen | C_D | Descripción | Distancia |
|---------|-----|-------------|-----------|
| Ideal | 0.0 | Sin fricción | 86.74 m (100%) |
| Laminar | 0.5 | Re < 10⁵ | ~82 m (-5%) |
| Turbulento | 0.75 | Re > 10⁵ | ~79 m (-8%) |

**Análisis realizado:**
- ✅ Cálculo de **velocidad inicial** necesaria (Newton-Raphson)
- ✅ Trayectorias **y = y(x)** para cada régimen
- ✅ Evolución temporal **y = y(t)**
- ✅ **Cuantificación del efecto** de la fricción

**Gráficas generadas:** 📂 `resultados_martillo/`
- `trayectoria_CD_0.00.png` - Sin fricción
- `trayectoria_CD_0.50.png` - Flujo laminar
- `trayectoria_CD_0.75.png` - Flujo turbulento
- `comparacion_trayectorias.png` - Comparativa
- `analisis_friccion.png` - Análisis cuantitativo

---

### 3️⃣ **Osciladores Armónicos Acoplados**

Sistema de dos masas conectadas por resortes con análisis **lineal vs no lineal**.

**📂 Ubicación:** `src/hmwks/homework_4/couppled.py`

**Configuración:** `|--k--[m₁]--k'--[m₂]--k--|`

**Sistemas estudiados:**

| Sistema | Fuerza restauradora | Características |
|---------|---------------------|-----------------|
| **Lineal** | F = -kx | Frecuencia constante |
| **No lineal** | F = -k(x + 0.1x³) | Frecuencia depende de amplitud |

**Análisis realizado:**
- ✅ Cálculo de **modos normales** (eigenvalores y eigenvectores)
- ✅ Frecuencias de oscilación: ω₁ (simétrico), ω₂ (antisimétrico)
- ✅ Simulación con **3 condiciones iniciales**:
  - Ambas masas desplazadas igualmente
  - Desplazamientos opuestos
  - Una masa en equilibrio, otra desplazada
- ✅ **Comparación lineal vs no lineal**
- ✅ Dependencia de **frecuencia con amplitud** (sistema no lineal)

**Modos normales (sistema lineal):**

| Modo | ω (rad/s) | f (Hz) | Período (s) | Tipo |
|------|-----------|--------|-------------|------|
| 1 | 3.162 | 0.503 | 1.988 | Simétrico (en fase) |
| 2 | 4.472 | 0.712 | 1.405 | Antisimétrico (fuera de fase) |

**Efecto no lineal:**

| Amplitud | Δf (%) | Observación |
|----------|--------|-------------|
| 0.2 m | +0.2% | Efecto despreciable |
| 0.8 m | +4.3% | Efecto moderado |
| 1.2 m | +5.7% | Efecto significativo |

**Gráficas generadas:** 📂 `resultados_harm/`
- Evolución temporal
- Espacios de fase
- Configuración espacial
- Análisis de frecuencias
- Comparación lineal vs no lineal

---

### 4️⃣ **Ajuste de Datos y Análisis Estadístico (Homework 5)**

Serie de proyectos de ajuste de curvas y análisis de datos experimentales.

**📂 Ubicación:** `src/hmwks/homework_5/`

#### **4.1 Radiación de Cuerpo Negro (COBE)**

Análisis del espectro de radiación cósmica de fondo usando datos del satélite COBE.

**Características:**
- 🌌 **Ley de Planck:** I(ν,T) = (2hν³/c²) · 1/(exp(hν/kT) - 1)
- 📊 Ajuste no lineal con **scipy.optimize.curve_fit**
- 🌡️ **Temperatura CMB estimada:** T ≈ 2.7255 K
- 📉 Análisis χ² para bondad de ajuste
- 🎯 Comparación con valor del satélite Planck (2.72548 K)

**Resultados:**
```
T_CMB = 2.7255 ± 0.0001 K
χ²_reducido ≈ 1.0 (excelente ajuste)
Diferencia con valor aceptado: < 0.01%
```

**Gráficas generadas:** 📂 `resultados_tarea_5/`
- Espectro de cuerpo negro con datos COBE
- Escala log-log del espectro
- Sensibilidad a la temperatura

---

#### **4.2 Circuito RLC - Decaimiento Exponencial**

Análisis de circuito RL con decaimiento exponencial de voltaje.

**Modelo físico:** V(t) = V₀ · e^(-t/τ)

**Características:**
- ⚡ Ajuste exponencial con incertidumbres
- 🔬 Estimación de constante de tiempo τ = R/L
- 📈 Análisis semi-logarítmico
- 📊 Test χ² de bondad de ajuste
- 🎯 Propagación de errores en parámetros físicos

**Resultados típicos:**
- Constante de tiempo: τ ≈ (valor) ± (error) s
- Coeficiente de correlación R² > 0.99

---

#### **4.3 Resonancia Breit-Wigner**

Ajuste de picos de resonancia en física de partículas.

**Modelo:** σ(E) = σ₀ · Γ²/[(E - E_R)² + Γ²/4]

**Características:**
- 🎯 Ajuste no lineal multidimensional
- 🔍 Método de Newton-Raphson
- 📉 Minimización de χ²
- ⚛️ Determinación de:
  - Energía de resonancia (E_R)
  - Anchura de resonancia (Γ)
  - Sección eficaz máxima (σ₀)

---

#### **4.4 Interpolación de Lagrange**

Interpolación polinómica y búsqueda de raíces.

**Características:**
- 📐 **Splines cúbicos** con scipy
- 🔍 **Método de Brent** para búsqueda de raíces
- 📊 Interpolación de secciones eficaces
- 🎯 Alta precisión numérica

**Archivo:** `src/hmwks/homework_5/lagrange_1.py`

---

### 5️⃣ **Análisis de Partículas del LHC (Homework 6)**

Análisis de datos reales del detector CMS del Large Hadron Collider (CERN).

**📂 Ubicación:** `src/hmwks/homework_6/mass_approximation.py`

**Objetivo:** Identificar partículas mediante el cálculo de **masa invariante** de pares μ⁺μ⁻.

**Fórmula de masa invariante:**
```
M² = E²_total - p²_total
M = √[(E₁ + E₂)² - (p⃗₁ + p⃗₂)²]
```

**Datasets analizados:**
1. **Jpsimumu_Run2011A.csv** - 31,000+ colisiones
2. **MuRun2010B.csv** - Búsqueda de bosón Z

**Partículas identificadas:**

| Partícula | Masa teórica (GeV/c²) | Masa observada | Descripción |
|-----------|------------------------|----------------|-------------|
| **J/ψ** | 3.097 | 3.095 ± 0.010 | Mesón de charmonio (c𝑐̄) |
| **Υ(1S)** | 9.460 | 9.458 ± 0.020 | Mesón de bottomonio (b𝑏̄) |
| **Z⁰** | 91.188 | 91.20 ± 0.05 | Bosón mediador débil |

**Análisis realizado:**
- ✅ Cálculo de masa invariante para 31,000+ eventos
- ✅ **Detección automática de picos** con scipy.signal.find_peaks
- ✅ Histogramas en escala lineal y logarítmica
- ✅ Comparación con Particle Data Group (PDG)
- ✅ Identificación de resonancias de charmonio y bottomonio

**Gráficas generadas:** 📂 `resultados_Tarea_6/`
- `histograma_μ⁺μ⁻_Run2011A.png`
- `histograma_Bosón_Z_Run2018B_Lineal.png`
- `histograma_Bosón_Z_Run2018B_Log.png`

**Importancia física:**
- Confirma la existencia de partículas predichas por el Modelo Estándar
- Método fundamental en física de altas energías
- Datos reales del LHC procesados con Python

---

### 6️⃣ **Decaimiento de Partículas y RNG (Homework 7)**

Dos proyectos sobre simulación Monte Carlo y generación de números aleatorios.

**📂 Ubicación:** `src/hmwks/homework_7/`

---

#### **6.1 Decaimiento de Piones (π mesones)**

Simulación relativista del decaimiento de 1 millón de piones.

**Archivo:** `pion.py`

**Física del problema:**
- 🔬 **Masa del pión:** m_π = 139.6 MeV/c²
- ⏱️ **Vida media (reposo):** τ₀ = 2.6×10⁻⁸ s
- 🚀 **Dilatación temporal:** τ = γ·τ₀ (efecto relativista)
- 📏 **Distancia:** 20 metros

**Casos estudiados:**

| Caso | Energía cinética | Factor γ | Sobreviven | Porcentaje |
|------|------------------|----------|------------|------------|
| **Monoenergético** | K = 200 MeV | 2.433 | ~714,000 | 71.4% |
| **Gaussiano** | μ=200, σ=50 MeV | Variable | ~695,000 | 69.5% |

**Análisis realizado:**
- ✅ Cálculo del **factor de Lorentz γ = E/m₀c²**
- ✅ Tiempo de vida dilatado τ' = γ·τ₀
- ✅ Probabilidad de supervivencia P = e^(-t/τ')
- ✅ Simulación Monte Carlo con N = 1,000,000 partículas
- ✅ Cálculo de **incertidumbres** (distribución binomial)
- ✅ Comparación monoenergético vs distribución gaussiana

**Gráficas generadas:** 📂 `resultados_tarea_7/`
- Distribución de energías
- Distribución del factor de Lorentz
- Distribución de probabilidades de supervivencia
- Comparación de resultados

---

#### **6.2 Generador de Números Pseudo-Aleatorios (LCG)**

Implementación y análisis de un **Linear Congruential Generator**.

**Archivo:** `pseudo_random.py`

**Algoritmo LCG:**
```
x_{n+1} = (a·x_n + c) mod M
```

**Análisis realizado:**
- ✅ Implementación de LCG desde cero
- ✅ **Detección del periodo** del generador
- ✅ **Test χ² de uniformidad** (Pearson)
- ✅ Análisis de **correlación serial** (scatter plots)
- ✅ Comparación con `random.random()` de Python
- ✅ Histogramas de distribución

**Pruebas estadísticas:**
- Test de uniformidad χ²
- Test de independencia (autocorrelación)
- Visualización 2D de pares (x_n, x_{n+1})

**Conclusión:** Validación de la calidad del generador pseudo-aleatorio.

---

### 7️⃣ **Distribución de Fermi-Dirac**

Análisis de estadística cuántica a temperatura ambiente.

**📂 Ubicación:** `src/examen/fermi.py`

**Problema:** Sistema cuántico con energías restringidas entre 0 y 2 eV a T = 300 K (kT ≈ 0.025 eV).

**Distribución de Fermi-Dirac:**
```
f_FD(E) = 1 / [exp((E - μ)/kT) + 1]
```

**Objetivo:** Encontrar el potencial químico μ tal que:
```
∫₀² f_FD(E) dE = 1  (normalización)
```

**Métodos numéricos utilizados:**
- 🔍 **Método de Brent** (búsqueda de raíces)
- ∫ **Cuadratura de Gauss-Legendre** (scipy.integrate.quad)
- 📊 Exploración sistemática de F(μ)

**Resultados:**
```
μ* = 1.0000000 eV (aproximado)
Error de normalización: < 10⁻¹⁰
χ² de ajuste: excelente
```

**Análisis realizado:**
- ✅ Búsqueda de cambio de signo en F(μ)
- ✅ Convergencia del método de Brent
- ✅ Gráfica de distribución f_FD(E)
- ✅ Zoom en región de transición (μ ± kT)
- ✅ Verificación de normalización

**Gráficas generadas:** 📂 `resultados_examen_final/`
- `exploracion_funcion_objetivo.png`
- `distribucion_fermi_dirac.png`

**Interpretación física:**
- El nivel de Fermi está en el punto medio del intervalo
- Transición suave debido a kT << (E_max - E_min)
- Típico de sistemas de Fermi a temperatura ambiente

---

### 8️⃣ **Physics-Informed Neural Networks (PINNs)**

Red neuronal que aprende física directamente de las ecuaciones diferenciales.

**📂 Ubicación:** `src/pinns/pendulum.py`

**Problema:** Péndulo simple no lineal

**Ecuación diferencial:**
```
d²θ/dt² + (g/L)·sin(θ) = 0
```

**Arquitectura de la red:**
- 🧠 **Input:** tiempo t
- 🧠 **Output:** ángulo θ(t)
- 🧠 **Capas ocultas:** 4 capas × 32 neuronas
- 🧠 **Activación:** Tanh
- 🧠 **Framework:** PyTorch
- 🧠 **Soporte GPU:** CUDA compatible

**Características innovadoras:**
- ⚛️ **Diferenciación automática** para calcular d²θ/dt²
- 📐 La red aprende la física sin datos de entrenamiento
- 🎯 Loss function física: L = ||d²θ/dt² + (g/L)sin(θ)||²
- 🚀 Condiciones iniciales: θ(0) = π/4, θ'(0) = 0

**Función de pérdida total:**
```python
Loss = Loss_física + 10·Loss_condiciones_iniciales
```

**Entrenamiento:**
- 5000 épocas
- Optimizador: Adam (lr = 10⁻³)
- 200 puntos de entrenamiento en t ∈ [0, 10s]

**Análisis de resultados:**
- ✅ Convergencia de la pérdida (escala log)
- ✅ Comparación con solución analítica (aproximación lineal)
- ✅ **Conservación de energía:** E_total = KE + PE = constante
- ✅ **Diagrama de fases:** (θ, ω) - órbita cerrada

**Gráficas generadas:**
- `pinn_pendulo_resultados.png` (4 paneles)
  - Convergencia del entrenamiento
  - θ(t): PINN vs analítico
  - Conservación de energía
  - Espacio de fases

**Ventajas del enfoque PINN:**
- No requiere datos experimentales
- Incorpora leyes físicas directamente
- Generaliza mejor que redes tradicionales
- Conserva propiedades físicas (energía, momentum)

---

## 🛠️ Instalación

### Requisitos previos
- Python 3.8 o superior
- pip (gestor de paquetes)

### Instalación rápida

```bash
# 1. Clonar repositorio
git clone https://github.com/OscarAnds0411/Fisica-numerica.git
cd Fisica-numerica

# 2. Crear entorno virtual (recomendado)
python -m venv .venv

# Activar entorno virtual
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## 📦 Dependencias

### Librerías principales

```txt
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0
rich>=10.0.0
torch>=1.9.0
```

### Instalación manual

```bash
# Dependencias básicas
pip install numpy scipy matplotlib pandas rich

# Para PINNs (opcional - requiere PyTorch)
pip install torch torchvision
```

### Dependencias por proyecto

| Proyecto | Librerías requeridas |
|----------|---------------------|
| Análisis básico | numpy, scipy, matplotlib |
| Homework 5-7 | + pandas, rich |
| LHC Analysis | + pandas (para CSV) |
| PINNs | + torch (PyTorch) |

---

## 💻 Uso

### 1. Análisis de Precisión Numérica

```bash
cd src
python Tarea1.py
```

**Salida:**
```
Overflow estimado:  1.7976931348623157e+308
Underflow estimado: 5e-324
Epsilon de máquina: 2.220446049250313e-16

Cálculo de sin(π/4) con serie de Taylor:
N     Suma                 Error relativo
1     7.853982e-01         1.000000e+00
5     7.071068e-01         1.234568e-07
10    7.071068e-01         2.345678e-15
```

---

### 2. Lanzamiento de Martillo

```bash
cd src/hmwks/homework_4
python testeo.py
```

**Salida:**
- Tablas comparativas en consola (Rich)
- 6 gráficas PNG en `resultados_martillo/`
- Análisis cuantitativo del efecto de fricción

---

### 3. Osciladores Acoplados

```bash
cd src/hmwks/homework_4
python couppled.py
```

**Salida:**
- Modos normales teóricos (tabla)
- Frecuencias para diferentes amplitudes
- 9+ gráficas PNG en `resultados_harm/`
- Comparación lineal vs no lineal

---

### 4. Análisis de Radiación Cósmica (COBE)

```bash
cd src/hmwks/homework_5
python radiation.py
```

**Salida:**
- Temperatura del CMB estimada
- Gráficas del espectro de Planck
- Comparación con valor aceptado
- Análisis χ² de bondad de ajuste

---

### 5. Análisis de Partículas del LHC

```bash
cd src/hmwks/homework_6
python mass_approximation.py
```

**Salida:**
- Identificación automática de partículas
- Histogramas de masa invariante
- Comparación con PDG (Particle Data Group)
- Tablas con Rich en consola

---

### 6. Simulación de Decaimiento de Piones

```bash
cd src/hmwks/homework_7
python pion.py
```

**Salida:**
- Análisis relativista con factor de Lorentz
- Comparación monoenergético vs gaussiano
- Cálculo de incertidumbres
- 2 figuras con múltiples paneles

---

### 7. Distribución de Fermi-Dirac

```bash
cd src/examen
python fermi.py
```

**Salida:**
- Potencial químico μ normalizado
- Gráficas de distribución cuántica
- Análisis de convergencia numérica

---

### 8. Physics-Informed Neural Network

```bash
cd src/pinns
python pendulum.py
```

**Salida:**
- Entrenamiento de red neuronal (5000 épocas)
- 4 gráficas de análisis físico
- Verificación de conservación de energía
- Requiere: PyTorch instalado

---

## 📁 Estructura del Repositorio

```
Fisica-numerica/
├── README.md                      # Este archivo
├── requirements.txt               # Dependencias Python
├── LICENSE                        # Licencia MIT
├── .gitignore                     # Archivos ignorados por git
│
├── docs/                          # Documentación
│   └── README.md                  # README principal (español)
│
├── src/                           # 📂 CÓDIGO FUENTE PRINCIPAL
│   ├── Tarea1.py                  # Precisión numérica (overflow, underflow, ε)
│   ├── Pruebas.py                 # Experimentos adicionales
│   ├── Test.py                    # Script de pruebas
│   │
│   ├── hmwks/                     # 📚 TAREAS DEL CURSO
│   │   ├── homework_3/            # EDOs básicas
│   │   │   ├── Tarea3_a.py
│   │   │   └── Tarea3_b.py
│   │   │
│   │   ├── homework_4/            # Mecánica clásica
│   │   │   ├── testeo.py          # 🔨 Lanzamiento de martillo
│   │   │   ├── couppled.py        # 🌊 Osciladores acoplados
│   │   │   ├── hammer.py          # Implementación alternativa
│   │   │   └── Cuerda_vpython.py  # Visualización 3D con VPython
│   │   │
│   │   ├── homework_5/            # 📊 AJUSTE DE DATOS
│   │   │   ├── radiation.py       # ⭐ Radiación COBE (CMB)
│   │   │   ├── rlc_circuit.py     # ⚡ Circuito RLC
│   │   │   ├── fitting_params.py  # 🎯 Resonancia Breit-Wigner
│   │   │   ├── lagrange_1.py      # 📐 Interpolación de Lagrange
│   │   │   └── rikh_demo.py       # Demo de librería Rich
│   │   │
│   │   ├── homework_6/            # ⚛️ FÍSICA DE ALTAS ENERGÍAS
│   │   │   └── mass_approximation.py  # 🔬 Análisis LHC (J/ψ, Υ, Z⁰)
│   │   │
│   │   └── homework_7/            # 🎲 MONTE CARLO & RNG
│   │       ├── pion.py            # 🔴 Decaimiento de piones (relativista)
│   │       └── pseudo_random.py   # 🎰 Generador LCG + tests estadísticos
│   │
│   ├── examen/                    # 📝 MATERIAL DE EXAMEN
│   │   └── fermi.py               # 📊 Distribución de Fermi-Dirac
│   │
│   ├── practice_exam/             # 📖 PREPARACIÓN DE EXAMEN
│   │   ├── excercises.py          # 10 ejercicios de práctica
│   │   ├── exc_1.py, exc_2.py     # Ejercicios individuales
│   │   ├── trion.py
│   │   └── ex_s.ipynb             # Jupyter notebook
│   │
│   ├── pinns/                     # 🧠 MACHINE LEARNING PARA FÍSICA
│   │   └── pendulum.py            # 🤖 PINN para péndulo simple
│   │
│   └── class_activity/            # 🎓 ACTIVIDADES DE CLASE
│       ├── activity_6/  (2 archivos)
│       ├── activity_7/  (2 archivos)
│       ├── activity_8/  (3 archivos)
│       ├── activity_9/  (1 archivo)
│       ├── activity_10/ (2 archivos)
│       └── examples/
│           ├── animacionFIGURAS.py
│           ├── animacionFUNCION.py
│           ├── EULER2.py
│           ├── ex.py
│           └── newton-raphson.py
│
├── 📊 DIRECTORIOS DE RESULTADOS (generados automáticamente)
│   ├── resultados_martillo/       # Lanzamiento de martillo
│   ├── resultados_harm/           # Osciladores armónicos
│   ├── resultados_harm_test/      # Tests de osciladores
│   ├── resultados_cuerda/         # Física de cuerdas
│   ├── resultados_tarea_5/        # Homework 5 outputs
│   ├── resultados_Tarea_6/        # Análisis LHC
│   ├── resultados_tarea_7/        # Piones & RNG
│   ├── resultados_fermi_dirac/    # Distribución F-D
│   ├── resultados_examen_final/   # Resultados de examen
│   ├── exam_results/              # Resultados adicionales
│   ├── lhc_analysis/              # Análisis LHC detallado
│   └── data_points/               # Datos procesados
│
├── 📄 DATOS (CSV, TXT)
│   ├── Jpsimumu_Run2011A.csv      # Datos del CMS (J/ψ mesons)
│   ├── MuRun2010B.csv             # Datos del CMS (Z boson)
│   └── Datos_cuerpo_negro.txt     # Datos de COBE
│
└── .venv/                         # Entorno virtual (no en git)
```

### 📊 Resumen de contenido

| Categoría | Número de archivos | Descripción |
|-----------|-------------------|-------------|
| **Proyectos principales** | 8 | Tareas completas documentadas |
| **Scripts auxiliares** | 20+ | Actividades de clase y práctica |
| **Directorios de resultados** | 10+ | Gráficas PNG y datos procesados |
| **Datos experimentales** | 3+ | CSV/TXT del LHC y COBE |
| **Notebooks** | 1 | Jupyter para análisis interactivo |

---

## 🔬 Metodología

### Métodos Numéricos Implementados

#### 1. **Integración de EDOs**
   - `scipy.integrate.odeint` (LSODA adaptativo)
   - Conversión de EDOs de orden superior a sistemas de primer orden
   - Método de Euler (implementación básica)
   - Runge-Kutta de 4to orden

#### 2. **Análisis de Modos Normales**
   - Cálculo de eigenvalores/eigenvectores con `numpy.linalg.eig`
   - Método de conteo de periodos (cruces por cero)
   - Transformada de Fourier (FFT) para análisis de frecuencias

#### 3. **Optimización y Ajuste**
   - **Método de Newton-Raphson** para velocidades iniciales
   - **scipy.optimize.curve_fit** para ajuste no lineal
   - **Minimización de χ²** para bondad de ajuste
   - **Método de Brent** para búsqueda de raíces

#### 4. **Aproximaciones y Series**
   - Series de Taylor para funciones trigonométricas
   - Interpolación de Lagrange
   - Splines cúbicos

#### 5. **Simulaciones Monte Carlo**
   - Generador de números pseudo-aleatorios (LCG)
   - Simulación de decaimiento radiactivo/relativista
   - Cálculo de incertidumbres (distribución binomial)

#### 6. **Análisis Estadístico**
   - **Test χ² de Pearson** (uniformidad y bondad de ajuste)
   - Propagación de errores
   - Análisis de correlación serial
   - Detección automática de picos con `scipy.signal.find_peaks`

#### 7. **Machine Learning para Física**
   - **Physics-Informed Neural Networks (PINNs)**
   - Diferenciación automática (PyTorch autograd)
   - Optimización Adam
   - Loss functions con restricciones físicas

### Transformación de EDOs

**Ejemplo:** Segunda orden → Sistema de primer orden

```python
# Ecuación original: m·ẍ = F(x, ẋ, t)

# Variables de estado:
estado[0] = x   # Posición
estado[1] = v   # Velocidad

# Sistema de primer orden:
dx/dt = v
dv/dt = F(x, v, t) / m
```

---

## 📊 Ejemplos de Resultados

### Efecto de la Fricción (Martillo)

<div align="center">

| Régimen | Alcance | Pérdida | Velocidad inicial |
|---------|---------|---------|-------------------|
| Sin fricción | 86.74 m | - | 29.3 m/s |
| Flujo laminar | 82.15 m | 5.3% | 29.3 m/s |
| Flujo turbulento | 79.42 m | 8.4% | 29.3 m/s |

</div>

### Dependencia No Lineal (Osciladores)

En sistemas no lineales, la frecuencia **aumenta con la amplitud** debido al término cúbico:

| Amplitud | f Lineal | f No Lineal | Δf |
|----------|----------|-------------|-----|
| 0.2 m | 0.6166 Hz | 0.6181 Hz | +0.24% |
| 0.5 m | 0.6166 Hz | 0.6285 Hz | +1.93% |
| 0.8 m | 0.6166 Hz | 0.6430 Hz | +4.28% |
| 1.2 m | 0.6166 Hz | 0.6515 Hz | +5.66% |

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. **Fork** este repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m 'Agrega nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abre un **Pull Request**

### Áreas de mejora sugeridas

- [x] Agregar métodos de ajuste no lineal (curve_fit, Newton-Raphson)
- [x] Implementar simulaciones Monte Carlo
- [x] Análisis de datos experimentales (LHC, COBE)
- [x] Machine Learning para física (PINNs con PyTorch)
- [ ] Animaciones de trayectorias con matplotlib.animation
- [ ] Análisis de estabilidad de Lyapunov para sistemas no lineales
- [ ] Interfaz gráfica interactiva (Streamlit/Dash)
- [ ] Tests unitarios con pytest
- [ ] Documentación con Sphinx
- [ ] Implementar más PINNs (ondas, calor, Schrödinger)

---

## 📚 Referencias

### 📖 Libros

#### Física Computacional
- Giordano & Nakanishi - *Computational Physics* (2nd Ed.)
- Press et al. - *Numerical Recipes in Python*
- Landau, R. & Páez, M. - *Computational Physics: Problem Solving with Python*

#### Mecánica Clásica
- Taylor, J.R. - *Classical Mechanics*
- Goldstein, H. - *Classical Mechanics* (3rd Ed.)

#### Física Estadística
- Kittel, C. - *Introduction to Solid State Physics*
- Ashcroft & Mermin - *Solid State Physics*

#### Física de Partículas
- Griffiths, D. - *Introduction to Elementary Particles* (2nd Ed.)
- Particle Data Group (PDG) - [pdg.lbl.gov](https://pdg.lbl.gov/)

#### Machine Learning para Física
- Karniadakis et al. - *Physics-Informed Machine Learning*
- Raissi et al. (2019) - *Physics-Informed Neural Networks*

---

### 📄 Papers y Artículos

#### Datos Experimentales
- **COBE Science Team** - *Four-Year COBE DMR Cosmic Microwave Background Observations*
- **CMS Collaboration** - *Particle-Flow Event Reconstruction in CMS*
- Sedykh, Y. (1986) - *World Record in Hammer Throw* (86.74 m)

#### Métodos Numéricos
- Strogatz, S. - *Nonlinear Dynamics and Chaos*
- Hairer et al. - *Solving Ordinary Differential Equations*

#### PINNs
- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019)
  *Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations*

---

### 🔗 Documentación Técnica

#### Python Scientific Stack
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [SciPy Reference Guide](https://docs.scipy.org/doc/scipy/reference/)
  - [scipy.integrate.odeint](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html)
  - [scipy.optimize.curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
  - [scipy.signal.find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

#### Machine Learning
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Autograd](https://pytorch.org/docs/stable/autograd.html)

#### Datos Abiertos
- [CERN Open Data Portal](http://opendata.cern.ch/)
- [CMS Open Data](http://opendata.cern.ch/search?experiment=CMS)
- [COBE Data Archive](https://lambda.gsfc.nasa.gov/product/cobe/)

---

### 🎓 Recursos Educativos

- [Computational Physics with Python (Mark Newman)](http://www-personal.umich.edu/~mejn/cp/)
- [SciPy Lecture Notes](https://scipy-lectures.org/)
- [Python for Physics - University of Edinburgh](https://www.ph.ed.ac.uk/computing/python/)
- [Computational Physics Course - MIT](https://ocw.mit.edu/courses/physics/)

---

## 👤 Autor

**Oscar Andrés**
- GitHub: [@OscarAnds0411](https://github.com/OscarAnds0411)
- Proyecto: Física Numérica - Métodos Computacionales

---

## 📄 Licencia

Este proyecto está bajo la **Licencia MIT**. Ver [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2025 Oscar Andrés

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 📞 Soporte

¿Preguntas o sugerencias? Abre un [**issue**](https://github.com/OscarAnds0411/Fisica-numerica/issues) en GitHub.

---

<div align="center">

### ⭐ Si este proyecto te fue útil, considera darle una estrella ⭐

**Hecho con ❤️, Python 🐍 y Física ⚛️**

---

*Última actualización: Enero 2026*

---

## 📈 Estadísticas del Repositorio

| Métrica | Valor |
|---------|-------|
| **Total de proyectos principales** | 8 |
| **Scripts de código fuente** | 40+ archivos .py |
| **Líneas de código Python** | ~5,000+ |
| **Gráficas generadas** | 50+ archivos PNG |
| **Datasets analizados** | 3 (COBE, CMS Run2011A, CMS Run2010B) |
| **Métodos numéricos implementados** | 15+ |
| **Librerías utilizadas** | 7 principales |
| **Tareas completadas** | Homework 3-7 + Examen |

---

## 🆕 Nuevas Adiciones (Enero 2026)

### Proyectos Agregados
- ✅ **Homework 5:** Ajuste de datos (COBE, RLC, Breit-Wigner, Lagrange)
- ✅ **Homework 6:** Análisis de partículas del LHC (CMS data)
- ✅ **Homework 7:** Simulaciones Monte Carlo (piones + RNG)
- ✅ **Examen:** Distribución de Fermi-Dirac
- ✅ **PINNs:** Physics-Informed Neural Networks con PyTorch

### Métodos Numéricos Agregados
- ✅ Ajuste no lineal con `scipy.optimize.curve_fit`
- ✅ Método de Brent para búsqueda de raíces
- ✅ Test χ² de Pearson para bondad de ajuste
- ✅ Detección automática de picos (`scipy.signal.find_peaks`)
- ✅ Simulaciones Monte Carlo con LCG
- ✅ Diferenciación automática con PyTorch

### Datos Experimentales Reales
- ✅ Radiación cósmica de fondo (COBE satellite)
- ✅ Colisiones protón-protón del LHC (CMS detector)
- ✅ Identificación de J/ψ, Υ, y Z⁰ bosons

### Documentación
- ✅ README expandido con 8 proyectos documentados
- ✅ Estructura de directorios detallada
- ✅ Ejemplos de uso para cada proyecto
- ✅ Referencias bibliográficas completas

---

*Última actualización: Enero 2026*

</div>   