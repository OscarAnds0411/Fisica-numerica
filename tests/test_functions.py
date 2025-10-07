import math
import sys
from src.Pruebas import calcular_overflow, calcular_underflow, calcular_epsilon, sin_series

def test_calcular_overflow():
    # Verifica que el overflow esté cerca del máximo representable
    assert abs(calcular_overflow() - sys.float_info.max) < 1e300

def test_calcular_underflow():
    # Verifica que el underflow sea positivo y menor que el mínimo representable
    assert calcular_underflow() > 0
    assert calcular_underflow() < sys.float_info.min

def test_calcular_epsilon():
    # Verifica que el epsilon esté cerca del epsilon de máquina
    assert abs(calcular_epsilon() - sys.float_info.epsilon) < 1e-16

def test_sin_series():
    # Verifica que sin_series sea preciso para valores comunes
    tolerancia = 1e-8
    valores = [0, math.pi/6, math.pi/4, math.pi/2, math.pi]
    for x in valores:
        resultado, _ = sin_series(x, tol=tolerancia)
        assert abs(resultado - math.sin(x)) < tolerancia

def test_integration():
    # Prueba de integración entre las funciones
    overflow = calcular_overflow()
    underflow = calcular_underflow()
    resultado, _ = sin_series(math.pi/4, tol=1e-8)

    assert overflow > 1e300
    assert underflow < 1e-300
    assert abs(resultado - math.sin(math.pi/4)) < 1e-8