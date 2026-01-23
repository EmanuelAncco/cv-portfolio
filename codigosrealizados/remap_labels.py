import os
import yaml
import logging
from tkinter import filedialog, Tk

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("remap_labels.log", mode='w'),
        logging.StreamHandler()
    ]
)


def get_path_from_dialog(title):
    """Abre un diálogo para que el usuario seleccione una carpeta o archivo."""
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter
    if "directorio" in title.lower():
        path = filedialog.askdirectory(title=title)
    else:
        path = filedialog.askopenfilename(title=title)
    return path


def remap_labels():
    """
    Traduce los IDs de clase de un "paquete de corrección" para que coincidan
    con los IDs de un "dataset maestro".
    """
    logging.info("--- Iniciando el script de re-mapeo de etiquetas ---")

    # 1. Obtener las rutas del usuario
    correction_pack_dir = get_path_from_dialog(
        "Por favor, selecciona el directorio del 'Paquete de Corrección' descargado de Roboflow")
    if not correction_pack_dir:
        logging.warning("Operación cancelada. No se seleccionó el directorio del paquete de corrección.")
        return

    master_yaml_path = get_path_from_dialog(
        "Por favor, selecciona el archivo 'data.yaml' de tu dataset MAESTRO (ej. dataset_v3_rebalanced)")
    if not master_yaml_path:
        logging.warning("Operación cancelada. No se seleccionó el archivo YAML maestro.")
        return

    correction_yaml_path = os.path.join(correction_pack_dir, 'data.yaml')

    # 2. Validar que todos los archivos necesarios existan
    if not os.path.exists(correction_yaml_path):
        logging.error(
            f"Error: No se encontró 'data.yaml' en el directorio del paquete de corrección: {correction_pack_dir}")
        return

    logging.info(f"Paquete de corrección a procesar: {correction_pack_dir}")
    logging.info(f"Dataset maestro de referencia: {master_yaml_path}")

    try:
        # 3. Cargar los mapas de clases de ambos archivos YAML
        with open(master_yaml_path, 'r') as f:
            master_data = yaml.safe_load(f)
        master_class_names = master_data['names']
        logging.info(f"Mapa de clases maestro cargado con {len(master_class_names)} clases.")

        with open(correction_yaml_path, 'r') as f:
            correction_data = yaml.safe_load(f)
        correction_class_names = correction_data['names']
        logging.info(f"Mapa de clases de corrección cargado con {len(correction_class_names)} clases.")

        # 4. Crear el "mapa de traducción"
        # Esto crea un diccionario que mapea el ID antiguo al ID nuevo.
        # Ej: {0 (del paquete) -> 9 (del maestro), 1 -> 5, ...}
        translation_map = {
            old_id: master_class_names.index(name)
            for old_id, name in enumerate(correction_class_names)
            if name in master_class_names
        }
        logging.info(f"Mapa de traducción generado: {translation_map}")

        # 5. Iterar sobre las carpetas de etiquetas (train, valid, test)
        remapped_count = 0
        for subset in ['train', 'valid', 'test']:
            labels_dir = os.path.join(correction_pack_dir, 'labels', subset)
            if not os.path.isdir(labels_dir):
                logging.info(f"No se encontró el subdirectorio '{subset}/labels', omitiendo.")
                continue

            logging.info(f"Procesando etiquetas en: {labels_dir}")
            for filename in os.listdir(labels_dir):
                if filename.endswith('.txt'):
                    filepath = os.path.join(labels_dir, filename)
                    new_lines = []

                    with open(filepath, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split()
                        if not parts:
                            continue

                        old_class_id = int(parts[0])

                        # Traducir el ID de clase
                        if old_class_id in translation_map:
                            new_class_id = translation_map[old_class_id]
                            new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                            new_lines.append(new_line)
                            remapped_count += 1
                        else:
                            # Si una clase del paquete no está en el maestro, se omite.
                            logging.warning(
                                f"Clase con ID antiguo {old_class_id} no encontrada en el mapa maestro. Se omitirá en el archivo {filename}.")

                    # Re-escribir el archivo de etiquetas con los IDs corregidos
                    with open(filepath, 'w') as f:
                        f.writelines(new_lines)

        logging.info("--- Proceso de re-mapeo finalizado ---")
        logging.info(f"Se han re-mapeado un total de {remapped_count} etiquetas.")
        print("\n¡Éxito! Las etiquetas en el 'paquete_de_correccion' han sido actualizadas.")
        print("Ahora puedes copiar y pegar los archivos en tu dataset maestro.")

    except Exception as e:
        logging.error(f"Ocurrió un error inesperado durante el proceso: {e}", exc_info=True)


if __name__ == "__main__":
    remap_labels()
