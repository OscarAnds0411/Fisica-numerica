# AI Coding Agent Instructions for Fisica-numerica

## Overview
This project, **Fisica-numerica**, focuses on numerical methods and their applications in physics. The codebase includes Python scripts for calculating overflow, underflow, and approximations using Taylor series. The project is structured to facilitate experimentation and learning in numerical computation.

## Key Files and Their Roles
- **`Tarea1.py`**: Contains functions to calculate overflow and underflow boundaries of the system and demonstrates their usage.
- **`Pruebas.py`**: Includes additional experiments, such as calculating Taylor series approximations for trigonometric functions.
- **`README.md`**: Provides a brief overview of the project.

## Project-Specific Conventions
- **Numerical Precision**: The code relies on Python's floating-point arithmetic (IEEE 754). Ensure calculations handle edge cases like `float('inf')` and underflow to `0.0`.
- **Taylor Series**: Functions like `sin_series` implement Taylor series approximations. These functions include parameters for tolerance (`tol`) and maximum terms (`max_terms`) to control precision and performance.
- **Mathematical Constants**: Use `math.pi` and other constants from the `math` module for accuracy.

## Developer Workflows
### Running the Code
1. Ensure Python is installed (>=3.8).
2. Run scripts directly using:
   ```bash
   python Tarea1.py
   python Pruebas.py
   ```

### Testing Numerical Functions
- Modify input values (e.g., `x` in `sin_series`) to test different scenarios.
- Adjust parameters like `tol` and `max_terms` to observe their impact on precision and performance.

### Debugging
- Use `print` statements to inspect intermediate values, especially in iterative calculations.
- For edge cases, verify results against known mathematical values (e.g., `sin(math.pi/2) = 1`).

## Integration Points
- **External Libraries**: The project uses Python's standard library (`math`, `sys`, `numpy`). Ensure these are available in the environment.
- **Environment Setup**: If using a virtual environment, activate it before running scripts:
  ```bash
  .venv\Scripts\activate  # Windows
  source .venv/bin/activate  # macOS/Linux
  ```

## Examples of Patterns
### Overflow Calculation
```python
def calcular_overflow():
    x = 1.0
    while x * 2 != float('inf'):
        x *= 2
    return x
```
### Taylor Series Approximation
```python
def sin_series(x, tol=1e-8, max_terms=1000):
    term = x
    suma = term
    n = 0
    while abs(term) >= tol and n < max_terms:
        term = -term * x**2 / ((2*n+2)*(2*n+3))
        suma += term
        n += 1
    return suma
```

## Notes
- This project is designed for educational purposes. Ensure numerical results are validated for real-world applications.
- Contributions should follow the existing coding style and structure.