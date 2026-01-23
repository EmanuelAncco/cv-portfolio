#!/usr/bin/env python3
"""
potato_vit_inference_gui.py – Aplicación GUI para Inferencia con Vision Transformer

Permite seleccionar imágenes, realizar inferencia con el modelo ViT entrenado
y mostrar los resultados en una interfaz gráfica con historial.

Requiere:
- Python 3.10+
- TensorFlow 2.10+
- Pillow (pip install Pillow)
- El modelo entrenado guardado en formato SavedModel ('final_vit_model_tf' por defecto)
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # Importar layers para las clases personalizadas
import os

# ─────────────────────────── Configuración del Modelo ────────────────────────────
# Ruta donde se guardó el modelo en formato SavedModel
MODEL_PATH = "final_vit_model_tf" # ¡Asegúrate de que esta ruta sea correcta!

# Parámetros de preprocesamiento de imagen (deben coincidir con el entrenamiento)
IMG_SIZE = (224, 224)
# Rango de reescalado utilizado durante el entrenamiento (ej: Rescaling(1./255))
RESCALE_RANGE = 1.0 / 255.0

# Nombres de las clases en el orden que el modelo las predice (deben coincidir con train_ds.class_names)
# Basado en tu salida: ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# ─────────────────────────── Carga del Modelo ────────────────────────────
# Para cargar un SavedModel que contiene capas personalizadas,
# necesitamos definir esas clases personalizadas aquí.
# Estas definiciones deben coincidir exactamente con las utilizadas al construir el modelo.

# --- Definiciones de Capas Personalizadas (Copiadas del script de entrenamiento) ---
class Patches(layers.Layer):
    """Divide las imágenes en parches."""
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    """Codifica los parches y añade embeddings posicionales."""
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
            })
        return config
# --- Fin Definiciones de Capas Personalizadas ---

# Diccionario de objetos personalizados para cargar el modelo
custom_objects = {
    'Patches': Patches,
    'PatchEncoder': PatchEncoder,
    # Si la capa de aumento de datos fue una Sequential nombrada y necesitas cargarla
    # como un objeto personalizado, podrías necesitar algo como:
    # 'data_augmentation': keras.Sequential # O la clase específica si la definiste
}

# Intenta cargar el modelo
model = None # Inicializa el modelo como None
try:
    print(f"Intentando cargar el modelo desde: {MODEL_PATH}")
    # Cargar el modelo SavedModel, pasando los objetos personalizados
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("Modelo cargado exitosamente.")
    # Opcional: Imprimir un resumen del modelo cargado para verificar
    # model.summary()
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None # Asegúrate de que el modelo sea None si falla la carga
    # Muestra un cuadro de diálogo de error si la GUI se va a iniciar
    if __name__ == "__main__":
         messagebox.showerror("Error de Carga del Modelo",
                              f"No se pudo cargar el modelo desde {MODEL_PATH}.\n"
                              f"Detalle del error: {e}\n"
                              "La aplicación no puede iniciar sin el modelo.")


# ─────────────────────────── Clase de la Aplicación GUI ────────────────────────────

class InferenceApp:
    def __init__(self, root):
        self.root = root
        root.title("Clasificador de Tizón de la Papa con ViT")

        # Aplica un tema para una mejor apariencia visual
        style = ttk.Style()
        # Prueba con diferentes temas: 'clam', 'alt', 'default', 'classic'
        style.theme_use('clam')

        # --- Elementos de la GUI ---
        self.main_frame = ttk.Frame(root, padding="15")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Área para mostrar la imagen seleccionada
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Imagen Seleccionada", padding="15")
        # sticky=(tk.W, tk.E, tk.N, tk.S) permite que el frame se expanda en todas direcciones
        self.image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        # Label para mostrar la imagen. Inicialmente vacío.
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) # Permite que el label se expanda

        # Área para mostrar el resultado de la predicción
        self.result_frame = ttk.LabelFrame(self.main_frame, text="Resultado de Clasificación", padding="15")
        # sticky=(tk.W, tk.E, tk.N) permite que el frame se expanda horizontalmente y hacia arriba
        self.result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N), padx=10, pady=10)

        # Label para mostrar el nombre de la clase predicha
        self.prediction_label = ttk.Label(self.result_frame, text="Clasificación: N/A", font=('Arial', 14, 'bold'), wraplength=300) # wraplength para ajustar texto largo
        self.prediction_label.grid(row=0, column=0, sticky=(tk.W, tk.E)) # Permite que el label se expanda horizontalmente


        # Botón para seleccionar imagen
        self.select_button = ttk.Button(self.main_frame, text="Seleccionar Imagen para Clasificar", command=self.select_image)
        self.select_button.grid(row=1, column=0, columnspan=2, pady=15) # columnspan 2 para centrar el botón

        # Tabla para el historial de clasificaciones
        self.history_frame = ttk.LabelFrame(self.main_frame, text="Historial de Clasificaciones", padding="15")
        # sticky=(tk.W, tk.E, tk.N, tk.S) permite que el frame se expanda en todas direcciones
        self.history_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        # Scrollbar para la tabla
        self.history_scrollbar = ttk.Scrollbar(self.history_frame)
        self.history_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Treeview para mostrar el historial (actúa como tabla)
        self.history_tree = ttk.Treeview(self.history_frame, columns=("Filename", "Classification"), show="headings", yscrollcommand=self.history_scrollbar.set)
        self.history_tree.heading("Filename", text="Nombre del Archivo")
        self.history_tree.heading("Classification", text="Clasificación Predicha")

        # Ajustar el ancho de las columnas (opcional, se ajustarán automáticamente)
        self.history_tree.column("Filename", width=200, anchor=tk.W)
        self.history_tree.column("Classification", width=200, anchor=tk.W)

        self.history_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) # Permite que la tabla se expanda

        # Configurar el scrollbar para controlar la tabla
        self.history_scrollbar.config(command=self.history_tree.yview)

        # --- Configurar pesos para redimensionamiento ---
        # Permite que la ventana principal se expanda
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        # Permite que el frame principal se expanda
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1) # Permite que la fila de imagen/resultado se expanda
        self.main_frame.rowconfigure(2, weight=1) # Permite que la fila del historial se expanda
        # Permite que el frame de imagen y su contenido se expandan
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        # Permite que el frame del historial y su contenido se expandan
        self.history_frame.columnconfigure(0, weight=1)
        self.history_frame.rowconfigure(0, weight=1)


    def select_image(self):
        """
        Abre un diálogo para seleccionar un archivo de imagen,
        carga la imagen, realiza la inferencia y actualiza la GUI.
        """
        # Verifica si el modelo se cargó correctamente
        if model is None:
            messagebox.showerror("Error", "El modelo no está disponible. No se puede realizar la inferencia.")
            return

        # Abre el diálogo de selección de archivo
        file_path = filedialog.askopenfilename(
            initialdir=".", # Directorio inicial (puede ser "." para el directorio actual)
            title="Seleccionar archivo de imagen",
            filetypes=(("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.gif"), # Tipos de archivo permitidos
                       ("Todos los archivos", "*.*"))
        )

        # Si el usuario cancela el diálogo, no hacemos nada
        if not file_path:
            return

        try:
            # --- Cargar y mostrar la imagen ---
            img = Image.open(file_path)
            # Crea una copia para mostrar y redimensiona para que quepa en el label
            img_display = img.copy()
            # Redimensiona manteniendo la relación de aspecto, máximo 300x300 píxeles para mostrar
            img_display.thumbnail((300, 300))
            # Convierte la imagen de PIL a un objeto PhotoImage compatible con Tkinter
            img_tk = ImageTk.PhotoImage(img_display)
            # Actualiza el label de la imagen con la nueva imagen
            self.image_label.configure(image=img_tk)
            # Mantén una referencia a la imagen para evitar que sea eliminada por el recolector de basura
            self.image_label.image = img_tk

            # --- Preprocesar la imagen para la inferencia ---
            # Redimensiona la imagen al tamaño esperado por el modelo
            img_inference = img.resize(IMG_SIZE)
            # Convierte la imagen a un array NumPy y asegura el tipo de dato float32
            img_array = np.array(img_inference).astype(np.float32)
            # Añade una dimensión de batch al principio (el modelo espera un batch de imágenes)
            img_array = np.expand_dims(img_array, axis=0)
            # Normaliza los valores de píxeles si es necesario (debe coincidir con el entrenamiento)
            img_array = img_array * RESCALE_RANGE

            # --- Realizar la inferencia ---
            # Obtiene las predicciones del modelo (probabilidades para cada clase)
            predictions = model.predict(img_array)
            # Obtiene el índice de la clase con la mayor probabilidad
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            # Obtiene el nombre de la clase predicha usando el índice
            predicted_class_name = CLASS_NAMES[predicted_class_index]

            # --- Actualizar la GUI con el resultado ---
            self.prediction_label.configure(text=f"Clasificación: {predicted_class_name}")

            # --- Añadir el resultado al historial ---
            # Obtiene solo el nombre del archivo de la ruta completa
            filename = os.path.basename(file_path)
            # Inserta una nueva fila en la tabla con el nombre del archivo y la clasificación
            self.history_tree.insert("", tk.END, values=(filename, predicted_class_name))

        except Exception as e:
            # Muestra un cuadro de diálogo de error si algo sale mal durante el procesamiento
            messagebox.showerror("Error de procesamiento", f"No se pudo procesar la imagen: {e}")


# ─────────────────────────── Ejecución Principal ────────────────────────────

if __name__ == "__main__":
    # Solo inicia la aplicación GUI si el modelo se cargó exitosamente
    if model is not None:
        # Crea la ventana principal de Tkinter
        root = tk.Tk()
        # Crea una instancia de nuestra aplicación
        app = InferenceApp(root)
        # Inicia el bucle principal de eventos de Tkinter
        root.mainloop()
    else:
        # Si el modelo no se cargó, el mensaje de error ya se mostró durante la carga.
        # No es necesario hacer nada más aquí, el script terminará.
        pass
