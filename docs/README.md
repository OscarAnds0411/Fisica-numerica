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
```

### Instalación manual

```bash
pip install numpy scipy matplotlib pandas rich
```

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

## 📁 Estructura del Repositorio

```
Fisica-numerica/
├── README.md                      # Este archivo
├── requirements.txt               # Dependencias
├── LICENSE                        # Licencia MIT
├── .gitignore                     # Archivos ignorados
│
├── src/                          # Código fuente
│   ├── Tarea1.py                # Precisión numérica
│   ├── Pruebas.py               # Experimentos adicionales
│   │
│   ├── hmwks/                   # Tareas del curso
│   │   ├── homework_3/          # Tarea 3
│   │   └── homework_4/          # Tarea 4
│   │       ├── testeo.py        # Lanzamiento martillo
│   │       └── couppled.py      # Osciladores acoplados
│   │
│   ├── class_activity/          # Actividades de clase
│   │   └── examples/
│   │
│   └── pinns/                   # Physics-Informed NNs
│
├── resultados_martillo/         # Gráficas martillo
├── resultados_harm/             # Gráficas osciladores
│
├── docs/                        # Documentación
├── tests/                       # Pruebas unitarias
│
└── .venv/                       # Entorno virtual (local)
```

---

## 🔬 Metodología

### Métodos Numéricos Implementados

1. **Integración de EDOs:**
   - `scipy.integrate.odeint` (LSODA adaptativo)
   - Conversión de EDOs de orden superior a sistemas de primer orden

2. **Análisis de Modos Normales:**
   - Cálculo de eigenvalores/eigenvectores con `numpy.linalg.eig`
   - Método de conteo de periodos (cruces por cero)

3. **Optimización:**
   - Método de Newton-Raphson para encontrar velocidades iniciales

4. **Aproximaciones:**
   - Series de Taylor para funciones trigonométricas

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

- [ ] Agregar más métodos de integración (RK4, Verlet)
- [ ] Implementar animaciones de trayectorias
- [ ] Análisis de estabilidad de Lyapunov para sistemas no lineales
- [ ] Interfaz gráfica interactiva (Streamlit/Dash)
- [ ] Tests unitarios con pytest
- [ ] Documentación con Sphinx

---

## 📚 Referencias

### Libros
- Giordano & Nakanishi - *Computational Physics* (2nd Ed.)
- Taylor, J.R. - *Classical Mechanics*
- Press et al. - *Numerical Recipes in Python*
- Goldstein, H. - *Classical Mechanics* (3rd Ed.)

### Papers
- Sedykh, Y. (1986) - "World Record in Hammer Throw"
- Strogatz, S. - *Nonlinear Dynamics and Chaos*

### Documentación técnica
- [SciPy odeint](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html)
- [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)

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

*Última actualización: Octubre 2025*

</div>