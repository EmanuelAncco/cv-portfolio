import pandas as pd
import matplotlib.pyplot as plt

# Leer el CSV con autodetección de delimitador (coma o punto y coma)
with open("detecciones.csv", 'r', encoding='utf-8') as file:
    first_line = file.readline()
    delimiter = ';' if ';' in first_line else ','
    file.seek(0)
    df = pd.read_csv(file, delimiter=delimiter)

# Mostrar columnas para verificación (opcional)
print("Columnas detectadas:", df.columns.tolist())

# Validar que exista la columna 'Frame'
if 'Frame' not in df.columns:
    raise ValueError("El archivo debe contener una columna llamada 'Frame'.")

# Contar objetos por frame
conteo_detecciones = df['Frame'].value_counts().sort_index()

# Graficar detecciones por frame
plt.figure(figsize=(14, 6))
plt.plot(conteo_detecciones.index, conteo_detecciones.values, marker='o', linestyle='-')
plt.title('Cantidad de objetos detectados por frame')
plt.xlabel('Frame')
plt.ylabel('Cantidad de detecciones')
plt.grid(True)
plt.tight_layout()
plt.show()
