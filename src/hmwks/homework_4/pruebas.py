from numpy import *

A=array([[1,1,1],[2,1,2],[3,2,4]])

print("Nuestra matriz A es:\n", A)
print("="*70)
a_squared=dot(A,A)
print("Nuestra matriz A^2 es:\n", a_squared)
print("="*70)
a_inv=linalg.inv(A)
print("Nuestra matriz inversa de A es:\n", a_inv)
print("="*70)
print("Buscamos resolver el problema de eigenvalores de la matriz A")
a_eingen=linalg.eig(A)
print(a_eingen)
print("="*70)


C=array([[2j,-1+1j],[1+1j,3j]])
print("Nuestra matriz C es:\n", C)
print("="*70)
c_squared=dot(C,C)
print("Nuestra matriz C^2 es:\n", c_squared)
print("="*70)
c_inv=linalg.inv(C)
print("Nuestra matriz inversa de C es:\n", c_inv)
print("="*70)
print("Buscamos resolver el problema de eigenvalores de la matriz C")
c_eingen=linalg.eig(C)
print(c_eingen)


# Suppress scientific notation and set decimal precision globally for NumPy
set_printoptions(suppress=True, precision=3)

# Create a sample matrix
A = array([[4, -1, 6], [2, 1, 6], [2, -1, 8]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = linalg.eig(A)

print("Eigenvalues and Eigenvectors:")
print("-" * 35)

for i in range(len(eigenvalues)):
    # Extract the i-th eigenvalue and its corresponding eigenvector (the i-th column)
    eigenvalue = eigenvalues[i]
    eigenvector = eigenvectors[:, i]

    print(f"Eigenvalue {i+1}: {eigenvalue:.4f}")
    print(f"Eigenvector {i+1}: {eigenvector}")
    print("-" * 35)

# To print complex numbers neatly, handle them separately
B = array([[1, -1], [1, 1]])
complex_eigenvalues, complex_eigenvectors = linalg.eig(B)

print("\nComplex Eigenvalues and Eigenvectors:")
print("-" * 35)
for i in range(len(complex_eigenvalues)):
    eigenvalue = complex_eigenvalues[i]
    eigenvector = complex_eigenvectors[:, i]

    print(f"Eigenvalue {i+1}: {eigenvalue}")
    print(f"Eigenvector {i+1}: {eigenvector}")
    print("-" * 35)

from rich.console import Console
from rich.table import Table

# Create a sample matrix
A = array([[4, -1, 6], [2, 1, 6], [2, -1, 8]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = linalg.eig(A)

console = Console()

# Create a table for the output
table = Table(title="Eigenvalues and Eigenvectors")
table.add_column("Index", style="cyan")
table.add_column("Eigenvalue", style="magenta")
table.add_column("Eigenvector", style="green")

# Add rows to the table
for i in range(len(eigenvalues)):
    eigenvalue = eigenvalues[i]
    eigenvector = eigenvectors[:, i]
    table.add_row(
        str(i + 1),
        f"{eigenvalue:.4f}",
        str(eigenvector)
    )

console.print(table)



import pandas as pd

# Create a sample matrix
A = array([[4, -1, 6], [2, 1, 6], [2, -1, 8]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = linalg.eig(A)

# Create a DataFrame for easy formatting
eigen_df = pd.DataFrame(
    {
        'Eigenvalue': eigenvalues,
        'Eigenvector': [eigenvectors[:, i] for i in range(len(eigenvalues))],
    }
)

# Set pandas display options for a tidy presentation
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_colwidth', None)

print("Eigenvalues and Eigenvectors (using pandas):")
print(eigen_df)
