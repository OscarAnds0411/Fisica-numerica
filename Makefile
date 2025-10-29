# Crear entorno virtual e instalar dependencias
setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

# Ejecutar todos los notebooks
run:
	jupyter nbconvert --to notebook --execute notebooks/*.ipynb --inplace

# Formatear código
format:
	pre-commit run --all-files

# Limpiar archivos temporales
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
