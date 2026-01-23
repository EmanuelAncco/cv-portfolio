# filtrar_dataset.py
import os
import shutil

def filtrar_y_remapear_dataset():
    # --- CONFIGURACIÓN ---
    base_path = r'D:\Python_proyectos_2025\SEGURIDAD2.0'
    
    # Verificar si la ruta base existe
    if not os.path.exists(base_path):
        print(f"Error: La ruta base '{base_path}' no existe.")
        return

    directorio_origen = os.path.join(base_path, 'dataset_original')
    directorio_destino = os.path.join(base_path, 'dataset_filtrado')

    # Verificar si el directorio de origen existe
    if not os.path.exists(directorio_origen):
        print(f"Error: El directorio de origen '{directorio_origen}' no existe.")
        return

    clases_deseadas = {
        3: 'Hardhat',
        8: 'NO-Hardhat',
        10: 'NO-Safety Vest',
        13: 'Safety Vest'
    }

    mapeo_clases_nuevo = {
        'Hardhat': 0,
        'Safety Vest': 1,
        'NO-Hardhat': 2,
        'NO-Safety Vest': 3
    }

    nombres_clases_nuevas = ['helmet', 'vest', 'no-helmet', 'no-vest']

    print(f"Iniciando el proceso de filtrado y re-mapeo en: {base_path}")

    for subdirectorio in ['train', 'valid', 'test']:
        print(f"\nProcesando subdirectorio: {subdirectorio}")

        origen_labels = os.path.join(directorio_origen, subdirectorio, 'labels')
        origen_images = os.path.join(directorio_origen, subdirectorio, 'images')

        # Verificar si los directorios de origen existen
        if not os.path.exists(origen_labels) or not os.path.exists(origen_images):
            print(f"Advertencia: Directorio de origen no encontrado: {origen_labels} o {origen_images}")
            continue

        destino_labels = os.path.join(directorio_destino, subdirectorio, 'labels')
        destino_images = os.path.join(directorio_destino, subdirectorio, 'images')

        os.makedirs(destino_labels, exist_ok=True)
        os.makedirs(destino_images, exist_ok=True)

        try:
            archivos_labels = os.listdir(origen_labels)
        except OSError as e:
            print(f"Error al listar archivos en {origen_labels}: {e}")
            continue

        for filename in archivos_labels:
            if not filename.endswith('.txt'):
                continue

            ruta_label = os.path.join(origen_labels, filename)
            if not os.path.exists(ruta_label):
                print(f"Advertencia: Archivo de etiquetas no encontrado: {ruta_label}")
                continue

            try:
                lineas_nuevas = []
                with open(ruta_label, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        class_id_original = int(parts[0])

                        if class_id_original in clases_deseadas:
                            nombre_clase_original = clases_deseadas[class_id_original]
                            class_id_nuevo = mapeo_clases_nuevo[nombre_clase_original]
                            nueva_linea = f"{class_id_nuevo} {' '.join(parts[1:])}"
                            lineas_nuevas.append(nueva_linea)

                if lineas_nuevas:
                    with open(os.path.join(destino_labels, filename), 'w') as f:
                        f.write('\n'.join(lineas_nuevas))

                    imagen_origen_path = os.path.join(origen_images, filename.replace('.txt', '.jpg'))
                    imagen_destino_path = os.path.join(destino_images, filename.replace('.txt', '.jpg'))
                    
                    if os.path.exists(imagen_origen_path):
                        shutil.copy2(imagen_origen_path, imagen_destino_path)
                    else:
                        print(f"Advertencia: Imagen no encontrada: {imagen_origen_path}")

            except Exception as e:
                print(f"Error procesando {filename}: {e}")

        print(f"Finalizado {subdirectorio}")

    yaml_path = os.path.join(directorio_destino, 'data.yaml')
    try:
        with open(yaml_path, 'w') as f:
            f.write(f"train: ./train/images\n")
            f.write(f"val: ./valid/images\n")
            f.write(f"test: ./test/images\n\n")
            f.write(f"nc: {len(nombres_clases_nuevas)}\n")
            f.write(f"names: {nombres_clases_nuevas}\n")
        print(f"\n¡Proceso completado! Dataset guardado en '{directorio_destino}'")
        print(f"Archivo de configuración creado en '{yaml_path}'")
    except Exception as e:
        print(f"Error creando archivo YAML: {e}")

if __name__ == '__main__':
    filtrar_y_remapear_dataset()