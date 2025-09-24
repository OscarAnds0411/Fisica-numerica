# limites_flotantes.py
import sys

def limites_iterativos():
    # Overflow por multiplicación
    x = 1.0
    while True:
        if x * 2.0 == float("inf"):
            break
        x *= 2.0
    overflow_est = x

    # Underflow por división
    y = 1.0
    while y / 2.0 > 0.0:
        y /= 2.0
    underflow_est = y

    return overflow_est, underflow_est

def limites_sys():
    max_float = sys.float_info.max
    min_float = sys.float_info.min  # mínimo positivo normalizado
    return max_float, min_float

if __name__ == "__main__":
    # Método iterativo
    overflow_est, underflow_est = limites_iterativos()

    # Valores de la máquina
    max_float, min_float = limites_sys()

    # Comparación
    print("==== MÉTODO ITERATIVO ====")
    print(f"Overflow estimado : {overflow_est}")
    print(f"Underflow estimado: {underflow_est}")

    print("\n==== VALORES DE sys.float_info ====")
    print(f"Máximo float      : {max_float}")
    print(f"Mínimo float      : {min_float}")

    print("\n==== COMPARACIÓN ====")
    print(f"Overflow (factor de diferencia): {max_float / overflow_est:.2f}")
    print(f"Underflow (factor de diferencia): {underflow_est / min_float:.2f}")