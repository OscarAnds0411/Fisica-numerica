# Organización del Proyecto

El proyecto ha sido reorganizado para mejorar la claridad y la estructura. Aquí está la nueva organización:

- **`src/`**: Contiene el código fuente principal del proyecto.
  - `Tarea1.py`: Código relacionado con el cálculo de overflow y underflow.
  - `Pruebas.py`: Experimentos adicionales, como la serie de Taylor.
- **`docs/`**: Documentación del proyecto.
  - `README.md`: Información general del proyecto.
- **`tests/`**: Pruebas unitarias y de integración.
- **`LICENSE`**: Licencia del proyecto.
- **`requeriments.txt`**: Dependencias necesarias para ejecutar el proyecto.

# Fisica-numerica

## Descripción
**Fisica-numerica** es un proyecto educativo que explora métodos numéricos y su aplicación en física. El objetivo principal es calcular límites de representación numérica (overflow y underflow), el epsilon de máquina, y aproximaciones de funciones matemáticas como el seno utilizando series de Taylor.

## Archivos Principales
- **`src/Tarea1.py`**: Contiene las implementaciones de las funciones principales:
  - `calcular_overflow`: Determina el mayor número representable antes de un desbordamiento.
  - `calcular_underflow`: Encuentra el menor número positivo representable antes de convertirse en 0.
  - `calcular_epsilon`: Calcula el epsilon de máquina, el menor número que puede sumarse a 1.0 para obtener un resultado distinto.
  - `sin_series`: Calcula el seno de un número utilizando la serie de Taylor.

## Requisitos
- Python 3.8 o superior.
- Librerías necesarias:
  - `numpy`

## Instalación
1. Clona este repositorio:
   ```bash
   git clone <URL-del-repositorio>
   ```
2. Navega al directorio del proyecto:
   ```bash
   cd Fisica-numerica
   ```
3. (Opcional) Crea y activa un entorno virtual:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # En Windows
   source .venv/bin/activate  # En macOS/Linux
   ```
4. Instala las dependencias:
   ```bash
   pip install -r requeriments.txt
   ```

## Uso
Ejecuta el archivo `src/Tarea1.py` para calcular los valores y generar tablas:
```bash
python src/Tarea1.py
```

### Ejemplo de Salida
```
Overflow estimado: 1.7976931348623157e+308
Underflow estimado: 5e-324
Epsilon de máquina: 2.220446049250313e-16

Cálculo de sin(0.7853981633974483) con serie de Taylor:
N     Suma                 Error relativo      
1     7.853981633974e-01   1.000000000000e+00  
2     7.704311800845e-01   9.999999999999e-01  
...
```

## Contribuciones
Este proyecto está diseñado para fines educativos. Si deseas contribuir, sigue estos pasos:
1. Haz un fork del repositorio.
2. Crea una nueva rama para tus cambios:
   ```bash
   git checkout -b mi-nueva-funcionalidad
   ```
3. Realiza tus cambios y haz commit:
   ```bash
   git commit -m "Agrega nueva funcionalidad"
   ```
4. Envía tus cambios:
   ```bash
   git push origin mi-nueva-funcionalidad
   ```
5. Abre un Pull Request.

## Notas
- Asegúrate de validar los resultados numéricos contra valores conocidos para garantizar la precisión.
- Sigue las convenciones de estilo del código existentes.

## Licencia
Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.