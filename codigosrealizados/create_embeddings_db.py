import os
import pickle
import logging
from deepface import DeepFace
import numpy as np

# --- CONFIGURACIÓN ---
DATABASE_PATH = "database"
OUTPUT_FILE = "face_database.pkl"

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')


def create_embeddings_database():
    """
    Procesa la base de datos de imágenes de trabajadores, calcula sus "huellas
    digitales faciales" (vector embeddings) y las guarda en un archivo .pkl
    para un acceso ultra-rápido durante la verificación.
    """
    if not os.path.exists(DATABASE_PATH):
        logging.error(f"El directorio de la base de datos '{DATABASE_PATH}' no existe.")
        return

    face_database = []

    # Iterar sobre cada trabajador registrado en la base de datos
    for worker_name in os.listdir(DATABASE_PATH):
        worker_dir = os.path.join(DATABASE_PATH, worker_name)
        if os.path.isdir(worker_dir):
            logging.info(f"Procesando al trabajador: {worker_name}...")
            worker_embeddings = []

            # Iterar sobre cada foto de registro del trabajador
            for image_file in os.listdir(worker_dir):
                image_path = os.path.join(worker_dir, image_file)
                try:
                    # --- MEJORA (v2.1): Ser más estricto en el registro ---
                    # enforce_detection=True asegura que solo se procesen fotos donde
                    # se detecte una cara clara, mejorando la calidad de la base de datos.
                    embedding_objs = DeepFace.represent(
                        img_path=image_path,
                        model_name='ArcFace',
                        enforce_detection=True
                    )
                    worker_embeddings.append(embedding_objs[0]["embedding"])
                except ValueError as e:
                    logging.warning(
                        f"¡ATENCIÓN! No se pudo detectar un rostro en la foto de registro {image_path}. Esta foto será ignorada. Asegúrese de que las fotos de registro sean claras y de frente. Error: {e}")
                except Exception as e:
                    logging.error(f"Error procesando {image_path}: {e}")

            if worker_embeddings:
                # Calcular el embedding "promedio" para tener una representación más robusta
                master_embedding = np.mean(worker_embeddings, axis=0)
                face_database.append((worker_name, master_embedding))
                logging.info(f"Embedding maestro creado para {worker_name}.")

    # Guardar la lista de embeddings en un archivo .pkl
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(face_database, f)

    logging.info("=" * 50)
    logging.info(f"¡Éxito! Base de datos de embeddings creada en '{OUTPUT_FILE}'.")
    logging.info(f"Total de trabajadores registrados: {len(face_database)}")
    logging.info("=" * 50)


if __name__ == "__main__":
    create_embeddings_database()
