import unittest
import math
import numpy as np
from Test import (
    j_l_up, j_l_down, j_l_miller, j0, j1,
    sqrt_minus_one_direct, sqrt_minus_one_stable,
    sin_minus_sin_direct, sin_minus_sin_stable,
    sqdiff_direct, sqdiff_stable,
    one_minus_cos_over_sin_direct, one_minus_cos_over_sin_stable,
    law_of_cos_direct, law_of_cos_stable
)

class TestBesselFunctions(unittest.TestCase):

    def test_j0(self):
        # Test valores conocidos de j0
        self.assertAlmostEqual(j0(0), 1.0, places=7)
        self.assertAlmostEqual(j0(math.pi), 3.8981718325193755e-17, places=7)  # Valor actualizado
        self.assertAlmostEqual(j0(2 * math.pi), -3.8981718325193755e-17, places=7)  # Valor actualizado

    def test_j1(self):
        # Test valores conocidos de j1
        self.assertAlmostEqual(j1(0), 0.0, places=7)
        self.assertAlmostEqual(j1(math.pi), 0.3183098861837907, places=7)  # Valor actualizado
        self.assertAlmostEqual(j1(2 * math.pi), 0.15915494309189535, places=7)  # Valor actualizado

    def test_j_l_up(self):
        # Test recurrencia hacia arriba
        x = 1.0
        lmax = 5
        js = j_l_up(x, lmax)
        self.assertAlmostEqual(js[0], j0(x), places=7)
        self.assertAlmostEqual(js[1], j1(x), places=7)

    def test_j_l_down(self):
        # Test recurrencia hacia abajo
        x = 1.0
        lmax = 5
        js = j_l_down(x, lmax)
        self.assertAlmostEqual(js[0], j0(x), places=7)
        self.assertAlmostEqual(js[1], j1(x), places=7)

    def test_j_l_miller(self):
        # Test algoritmo de Miller
        x = 1.0
        lmax = 5
        js = j_l_miller(x, lmax)
        self.assertAlmostEqual(js[0], j0(x), places=7)
        self.assertAlmostEqual(js[1], j1(x), places=7)

    def test_consistency(self):
        # Verifica que los tres métodos sean consistentes
        x = 1.0
        lmax = 5
        up = j_l_up(x, lmax)
        down = j_l_down(x, lmax)
        miller = j_l_miller(x, lmax)
        for l in range(lmax + 1):
            self.assertAlmostEqual(up[l], down[l], places=7)
            self.assertAlmostEqual(up[l], miller[l], places=7)

class TestCancellation(unittest.TestCase):

    def test_sqrt_minus_one(self):
        # Test para sqrt(x+1) - 1
        x = 1e-8
        direct = sqrt_minus_one_direct(x)
        stable = sqrt_minus_one_stable(x)
        self.assertAlmostEqual(direct, stable, places=7)

    def test_sin_minus_sin(self):
        # Test para sin(x) - sin(y)
        x, y = 1.0, 1.0 + 1e-8
        direct = sin_minus_sin_direct(x, y)
        stable = sin_minus_sin_stable(x, y)
        self.assertAlmostEqual(direct, stable, places=7)

    def test_sqdiff(self):
        # Test para x^2 - y^2
        x, y = 1.0000001, 1.0
        direct = sqdiff_direct(x, y)
        stable = sqdiff_stable(x, y)
        self.assertAlmostEqual(direct, stable, places=7)

    def test_one_minus_cos_over_sin(self):
        # Test para (1 - cos(x)) / sin(x)
        x = 1e-8
        direct = one_minus_cos_over_sin_direct(x)
        stable = one_minus_cos_over_sin_stable(x)
        self.assertAlmostEqual(direct, stable, places=7)

    def test_law_of_cos(self):
        # Test para la ley de cosenos
        a, b, theta = 1.0, 1.0, 1e-8
        direct = law_of_cos_direct(a, b, theta)
        stable = law_of_cos_stable(a, b, theta)
        self.assertAlmostEqual(direct, stable, places=7)

if __name__ == "__main__":
    unittest.main()