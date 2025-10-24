# ============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================================
import numpy as np              # Para operaciones matemáticas y arrays
import torch                    # PyTorch - framework de deep learning
import torch.nn as nn          # Módulo de redes neuronales de PyTorch
import matplotlib.pyplot as plt # Para visualizaciones y gráficas
from matplotlib.animation import FuncAnimation  # Para animaciones (opcional)

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

# Detecta si hay GPU disponible, si no usa CPU
# Esto acelera el entrenamiento si tienes una GPU NVIDIA con CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# ============================================================================
# PARÁMETROS FÍSICOS DEL PÉNDULO SIMPLE
# ============================================================================
# El péndulo simple es una masa colgando de una cuerda que oscila
g = 9.81            # Aceleración de la gravedad en m/s²
L = 1.0             # Longitud del péndulo en metros
theta0 = np.pi / 4  # Ángulo inicial: π/4 radianes = 45 grados
omega0 = 0.0        # Velocidad angular inicial (lo soltamos desde el reposo)

# ============================================================================
# DEFINICIÓN DE LA RED NEURONAL (PINN)
# ============================================================================
class PINN(nn.Module):
    """
    Physics-Informed Neural Network para el péndulo simple
    
    ¿Qué hace?
    - Toma como entrada: tiempo (t)
    - Produce como salida: ángulo del péndulo (θ)
    - La red aprende a cumplir la ecuación: d²θ/dt² + (g/L)*sin(θ) = 0
    
    Esta es la ecuación diferencial que gobierna el movimiento del péndulo
    """
    def __init__(self, hidden_layers=4, neurons_per_layer=32):
        super(PINN, self).__init__()
        
        # Construimos la arquitectura de la red neuronal capa por capa
        layers = []
        
        # CAPA DE ENTRADA: convierte tiempo (1 valor) en 32 neuronas
        layers.append(nn.Linear(1, neurons_per_layer))
        layers.append(nn.Tanh())  # Función de activación no lineal
        
        # CAPAS OCULTAS: 4 capas intermedias de 32 neuronas cada una
        # Estas capas aprenden patrones complejos del movimiento
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())  # Tanh funciona bien para problemas físicos
        
        # CAPA DE SALIDA: convierte las 32 neuronas en 1 valor (el ángulo θ)
        layers.append(nn.Linear(neurons_per_layer, 1))
        
        # Juntamos todas las capas en una secuencia
        self.network = nn.Sequential(*layers)
        
    def forward(self, t):
        """
        Propagación hacia adelante: t → red neuronal → θ(t)
        
        Entrada: tiempo t
        Salida: ángulo predicho θ en ese tiempo
        """
        return self.network(t)

# ============================================================================
# FUNCIÓN DE PÉRDIDA FÍSICA (Lo más importante del PINN)
# ============================================================================
def physics_loss(model, t):
    """
    Calcula qué tan bien la red cumple la ecuación física del péndulo
    
    La ecuación del péndulo es: d²θ/dt² + (g/L)*sin(θ) = 0
    
    ¿Cómo funciona?
    1. La red predice θ(t)
    2. Calculamos dθ/dt usando diferenciación automática
    3. Calculamos d²θ/dt² usando diferenciación automática otra vez
    4. Verificamos si d²θ/dt² + (g/L)*sin(θ) ≈ 0
    5. Si es cercano a cero, la red cumple la física correctamente
    """
    # Activamos el cálculo de gradientes para poder derivar
    t.requires_grad = True
    
    # PASO 1: La red predice el ángulo θ en cada tiempo t
    theta = model(t)
    
    # PASO 2: Calculamos la PRIMERA DERIVADA dθ/dt (velocidad angular)
    # torch.autograd.grad calcula derivadas automáticamente
    theta_t = torch.autograd.grad(
        theta, t,  # Derivamos theta respecto a t
        grad_outputs=torch.ones_like(theta),  # Necesario para el cálculo
        create_graph=True  # Permite calcular derivadas de derivadas
    )[0]
    
    # PASO 3: Calculamos la SEGUNDA DERIVADA d²θ/dt² (aceleración angular)
    # Derivamos la velocidad angular respecto al tiempo
    theta_tt = torch.autograd.grad(
        theta_t, t,  # Derivamos theta_t respecto a t
        grad_outputs=torch.ones_like(theta_t),
        create_graph=True  # Importante para el backpropagation
    )[0]
    
    # PASO 4: Evaluamos la ecuación del péndulo
    # Si la red es correcta: theta_tt + (g/L)*sin(theta) debe ser ≈ 0
    physics_residual = theta_tt + (g/L) * torch.sin(theta)
    
    # PASO 5: Convertimos en una pérdida (error cuadrático medio)
    # Mientras menor sea este número, mejor cumple la física
    return torch.mean(physics_residual**2)

# ============================================================================
# FUNCIÓN DE PÉRDIDA PARA CONDICIONES INICIALES
# ============================================================================
def initial_condition_loss(model, t0, theta0_val, omega0_val):
    """
    Asegura que la red cumpla las condiciones iniciales del problema
    
    Condiciones iniciales:
    - En t=0: θ(0) = θ₀ (el ángulo inicial que le damos)
    - En t=0: dθ/dt(0) = ω₀ (la velocidad angular inicial)
    
    Es como decirle a la red: "El péndulo empieza en 45° y sin moverse"
    """
    # Creamos un tensor para t=0
    t0_tensor = torch.tensor([[t0]], dtype=torch.float32, requires_grad=True).to(device)
    
    # CONDICIÓN 1: Posición inicial
    # La red debe predecir θ(0) = θ₀
    theta_pred = model(t0_tensor)
    loss_theta0 = (theta_pred - theta0_val)**2  # Error cuadrático
    
    # CONDICIÓN 2: Velocidad inicial
    # La derivada dθ/dt en t=0 debe ser ω₀
    theta_t = torch.autograd.grad(
        theta_pred, t0_tensor,
        grad_outputs=torch.ones_like(theta_pred),
        create_graph=True
    )[0]
    loss_omega0 = (theta_t - omega0_val)**2  # Error cuadrático
    
    # Sumamos ambas condiciones
    return loss_theta0 + loss_omega0

# ============================================================================
# FUNCIÓN DE ENTRENAMIENTO
# ============================================================================
def train_pinn(model, epochs=5000, lr=1e-3):
    """
    Entrena la red neuronal para que aprenda el comportamiento del péndulo
    
    Parámetros:
    - epochs: número de iteraciones de entrenamiento (5000 veces)
    - lr: learning rate (tasa de aprendizaje) - qué tan rápido aprende
    
    El proceso:
    1. Generar puntos de tiempo para evaluar
    2. Para cada época:
       a) Calcular la pérdida física (¿cumple la ecuación?)
       b) Calcular la pérdida de condiciones iniciales
       c) Ajustar los pesos de la red para reducir el error
    """
    # Optimizador Adam: ajusta los pesos de la red para minimizar el error
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Generamos 200 puntos de tiempo entre 0 y 10 segundos
    # La red aprenderá el comportamiento del péndulo en estos puntos
    t_train = torch.linspace(0, 10, 200).reshape(-1, 1).to(device)
    
    # Lista para guardar el historial de pérdidas (para graficar después)
    losses = []
    
    # BUCLE DE ENTRENAMIENTO
    for epoch in range(epochs):
        # Paso 1: Reiniciar los gradientes del optimizador
        optimizer.zero_grad()
        
        # Paso 2: Calcular la pérdida FÍSICA
        # ¿Qué tan bien cumple la ecuación d²θ/dt² + (g/L)*sin(θ) = 0?
        loss_physics = physics_loss(model, t_train)
        
        # Paso 3: Calcular la pérdida de CONDICIONES INICIALES
        # ¿Empieza en el ángulo y velocidad correctos?
        loss_ic = initial_condition_loss(model, 0.0, theta0, omega0)
        
        # Paso 4: Combinar ambas pérdidas
        # Multiplicamos loss_ic por 10 para darle más importancia
        # (queremos que DEFINITIVAMENTE cumpla las condiciones iniciales)
        loss = loss_physics + 10 * loss_ic
        
        # Paso 5: BACKPROPAGATION
        # Calcula cómo cambiar los pesos para reducir el error
        loss.backward()
        
        # Paso 6: ACTUALIZACIÓN DE PESOS
        # El optimizador ajusta los pesos de la red
        optimizer.step()
        
        # Guardamos la pérdida para análisis posterior
        losses.append(loss.item())
        
        # Cada 500 épocas, mostramos el progreso
        if (epoch + 1) % 500 == 0:
            print(f'Época {epoch+1}/{epochs}, Pérdida: {loss.item():.6f}')
    
    return losses

# ============================================================================
# EJECUCIÓN PRINCIPAL DEL PROGRAMA
# ============================================================================

# PASO 1: Crear el modelo PINN
print("Creando modelo PINN...")
print("  - La red tiene 4 capas ocultas con 32 neuronas cada una")
print("  - Entrada: tiempo (t)")
print("  - Salida: ángulo (θ)")
model = PINN(hidden_layers=4, neurons_per_layer=32).to(device)

# PASO 2: Entrenar el modelo
print("\nEntrenando modelo...")
print("  - Esto puede tardar 1-2 minutos dependiendo de tu computadora")
print("  - La red está aprendiendo las leyes de la física del péndulo")
losses = train_pinn(model, epochs=5000, lr=1e-3)

# PASO 3: Hacer predicciones con el modelo entrenado
print("\nGenerando predicciones...")
# Creamos 500 puntos de tiempo entre 0 y 10 segundos (más que en entrenamiento)
t_test = torch.linspace(0, 10, 500).reshape(-1, 1).to(device)

# Desactivamos el cálculo de gradientes (no necesitamos entrenar, solo predecir)
with torch.no_grad():
    theta_pred = model(t_test).cpu().numpy()  # Movemos a CPU y convertimos a numpy

# Convertimos el tiempo a numpy también
t_test_np = t_test.cpu().numpy()

# ============================================================================
# SOLUCIÓN ANALÍTICA (para comparar con nuestro PINN)
# ============================================================================
# Para ángulos PEQUEÑOS, existe una solución matemática exacta del péndulo
# θ(t) = θ₀ * cos(ω*t), donde ω = sqrt(g/L)
# 
# NOTA: Para ángulos grandes (como 45°), esta aproximación no es perfecta
# La solución exacta requiere funciones elípticas complejas
omega = np.sqrt(g/L)  # Frecuencia angular del péndulo
theta_analytical = theta0 * np.cos(omega * t_test_np)

# ============================================================================
# VISUALIZACIÓN DE RESULTADOS (4 gráficas diferentes)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Cuadrícula de 2x2 gráficas

# ---------------------------------------------------------------------------
# GRÁFICA 1: Pérdida durante el entrenamiento (esquina superior izquierda)
# ---------------------------------------------------------------------------
# Muestra cómo el error disminuye mientras la red aprende
axes[0, 0].plot(losses, color='blue', linewidth=1.5)
axes[0, 0].set_xlabel('Época de entrenamiento', fontsize=11)
axes[0, 0].set_ylabel('Pérdida (error)', fontsize=11)
axes[0, 0].set_title('Convergencia del entrenamiento', fontsize=12, fontweight='bold')
axes[0, 0].set_yscale('log')  # Escala logarítmica para ver mejor la convergencia
axes[0, 0].grid(True, alpha=0.3)
# Interpretación: Si baja constantemente → la red está aprendiendo correctamente

# ---------------------------------------------------------------------------
# GRÁFICA 2: Comparación PINN vs Solución analítica (superior derecha)
# ---------------------------------------------------------------------------
# Comparamos lo que aprendió nuestra red vs la solución matemática conocida
axes[0, 1].plot(t_test_np, theta_pred, label='PINN', linewidth=2, color='red')
axes[0, 1].plot(t_test_np, theta_analytical, '--', label='Aproximación lineal', 
                linewidth=2, color='blue', alpha=0.7)
axes[0, 1].set_xlabel('Tiempo (segundos)', fontsize=11)
axes[0, 1].set_ylabel('Ángulo θ (radianes)', fontsize=11)
axes[0, 1].set_title('Ángulo del péndulo vs Tiempo', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)
# Interpretación: Si las curvas se parecen → nuestro PINN funciona bien

# ---------------------------------------------------------------------------
# GRÁFICA 3: Conservación de Energía (inferior izquierda)
# ---------------------------------------------------------------------------
# El péndulo es un sistema conservativo: energía total debe mantenerse constante

# Primero, calculamos la velocidad angular (dθ/dt) numéricamente
dt = t_test_np[1] - t_test_np[0]  # Diferencia de tiempo entre puntos
omega_pred = np.gradient(theta_pred.flatten(), dt)  # Derivada numérica

# ENERGÍA CINÉTICA: (1/2)*m*L²*ω²  (asumimos masa m=1 kg)
# Depende de qué tan rápido se mueve el péndulo
KE = 0.5 * L**2 * omega_pred**2

# ENERGÍA POTENCIAL: m*g*L*(1-cos(θ))  (asumimos m=1 kg)  
# Depende de qué tan alto está el péndulo
PE = g * L * (1 - np.cos(theta_pred.flatten()))

# ENERGÍA TOTAL: debe ser constante si la física es correcta
E_total = KE + PE

axes[1, 0].plot(t_test_np, KE, label='Energía Cinética', linewidth=2, color='green')
axes[1, 0].plot(t_test_np, PE, label='Energía Potencial', linewidth=2, color='orange')
axes[1, 0].plot(t_test_np, E_total, '--', label='Energía Total', linewidth=2, 
                color='black', alpha=0.7)
axes[1, 0].set_xlabel('Tiempo (segundos)', fontsize=11)
axes[1, 0].set_ylabel('Energía (Joules)', fontsize=11)
axes[1, 0].set_title('Conservación de Energía', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
# Interpretación: Si la energía total es plana → conservación perfecta

# ---------------------------------------------------------------------------
# GRÁFICA 4: Diagrama de Fases (inferior derecha)
# ---------------------------------------------------------------------------
# Muestra la relación entre posición (θ) y velocidad (ω)
# Es una órbita cerrada porque el péndulo es periódico
axes[1, 1].plot(theta_pred, omega_pred, linewidth=2, color='purple')
axes[1, 1].set_xlabel('Ángulo θ (radianes)', fontsize=11)
axes[1, 1].set_ylabel('Velocidad angular ω (rad/s)', fontsize=11)
axes[1, 1].set_title('Diagrama de Fases', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
# Interpretación: Debería ser una elipse cerrada → movimiento periódico

# Ajustar espaciado entre gráficas
plt.tight_layout()

# Guardar la figura en alta resolución
plt.savefig('pinn_pendulo_resultados.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Entrenamiento completado!")
print(f"✓ Pérdida final: {losses[-1]:.6f}")
print(f"✓ Gráficas guardadas como 'pinn_pendulo_resultados.png'")

# Información adicional
print("\n" + "="*50)
print("INFORMACIÓN DEL MODELO")
print("="*50)
print(f"Parámetros del péndulo:")
print(f"  - Longitud (L): {L} m")
print(f"  - Gravedad (g): {g} m/s²")
print(f"  - Ángulo inicial (θ₀): {theta0:.4f} rad ({np.degrees(theta0):.2f}°)")
print(f"  - Velocidad angular inicial (ω₀): {omega0} rad/s")
print(f"\nArquitectura de la red:")
print(f"  - Capas ocultas: 4")
print(f"  - Neuronas por capa: 32")
print(f"  - Función de activación: Tanh")
print(f"  - Total de parámetros: {sum(p.numel() for p in model.parameters())}")
print("="*50)