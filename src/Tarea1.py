# ----- Valencia Magaña  Oscar Andrés -----
# ----- Tarea 1 -----
# Funcion para determinar el overflow y underflow de mi pc:
def det_overflow():
    x = 1.0
    while x * 2 != float("inf"):
        x *= 2.0
    return x


def det_underflow():
    z = 1.0
    while z / 2.0 != 0.0:  # Continúa mientras z no sea 0
        z /= 2.0
    return z


opc1 = det_overflow()
print("El valor de overflow es: ", opc1)
opc2 = det_underflow()
print("El valor de underflow es: ", opc2)
