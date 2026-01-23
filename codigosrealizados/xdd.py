import sys
import os

print("=" * 60)
print("--- 1. Python Executable (Verificando venv) ---")
print(sys.executable)
print("Debería estar dentro de 'D:\\Python_proyectos_2025\\GAIATECH\\.venv\\...'")
print("=" * 60)

print("\n--- 2. sys.path (Rutas de búsqueda de Python, en orden) ---")
for i, p in enumerate(sys.path):
    print(f"{i}: {p}")
print("=" * 60)

try:
    print("\n--- 3. Investigando la importación de 'concreteproperties' ---")
    import concreteproperties

    print("\n--- 4. ¡Paquete 'concreteproperties' ENCONTRADO! ---")
    print(f"Ruta del archivo __init__.py: {concreteproperties.__file__}")

    pkg_path = os.path.dirname(concreteproperties.__file__)
    print(f"Directorio del paquete: {pkg_path}")

    print("\n--- 5. Contenido de ese directorio (¿Contiene 'cross_section.py'?) ---")
    try:
        # Listamos los archivos para ver si 'cross_section.py' está ahí
        file_list = os.listdir(pkg_path)
        print(file_list)

        if 'cross_section.py' in file_list:
            print("\nVEREDICTO: 'cross_section.py' SÍ existe. El problema es otro.")
        else:
            print("\nVEREDICTO: ¡'cross_section.py' NO existe en este paquete!")
            print("Python está importando un paquete 'concreteproperties' INCORRECTO.")

    except Exception as e:
        print(f"No se pudo listar el directorio: {e}")

except ImportError as e:
    print(f"\n--- 3. ERROR DE IMPORTACIÓN ---")
    print(f"Falló el 'import concreteproperties' (Nivel superior): {e}")

print("=" * 60)
